"""Cleaning functions for the NYC 311 data (Phase 3).

The notebook notebooks/01_data_cleaning_eda.ipynb explains every decision.
The functions live here so that training and the API clean data the same way.

Two kinds of cleaning:
- FIXED rules need no data to be learned (for example "upper-case the text").
  They are in clean_features().
- LEARNED values come from the training period only (for example which rare
  categories become "Other"). They are learned by learn_cleaning_values(),
  saved to models/cleaning_values.json and applied by apply_rare_categories().

Rebuild the cleaned file from the project root:  python src/preprocessing.py
"""

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- settings
THRESHOLD_DAYS = 7          # target: resolved within 7 days
TRAIN_SHARE = 0.70          # oldest 70% of rows = training period
MAX_MISSING_SHARE = 0.95    # columns emptier than this are dropped
MIN_CATEGORY_ROWS = 50      # categories with fewer training rows become "Other"

UNKNOWN = "Unknown"         # used when a value is missing or a placeholder
OTHER = "Other"             # used for rare or never-seen categories

# Text that means "no value" in this dataset (compared in lower case)
PLACEHOLDERS = {"", "unspecified", "unknown", "n/a", "na", "none", "null"}

# Columns that are only known AFTER a request is filed. They must never be model
# inputs. closed_date is used once, to compute the target, and then dropped.
LEAKAGE_COLUMNS = ["closed_date", "status", "resolution_description",
                   "resolution_action_updated_date", "due_date"]

# Text columns kept in the cleaned file
CATEGORICAL_COLUMNS = ["agency", "complaint_type", "descriptor", "location_type",
                       "incident_zip", "address_type", "community_board",
                       "borough", "open_data_channel_type"]

# Column order of the cleaned file
CLEAN_COLUMNS = (["unique_key", "created_date"] + CATEGORICAL_COLUMNS
                 + ["latitude", "longitude", "duration_hours", "resolved_within_7d"])

# A box around New York City. Coordinates outside it are treated as wrong.
NYC_LAT_RANGE = (40.47, 40.93)
NYC_LON_RANGE = (-74.27, -73.68)

# The first 3 digits of every NYC ZIP code (Manhattan, Staten Island, Bronx,
# Queens and Brooklyn). "10000" has an NYC prefix but is not a real ZIP code.
NYC_ZIP_PREFIXES = {"100", "101", "102", "103", "104", "110", "111", "112", "113", "114", "116"}
FAKE_ZIPS = {"10000"}

RAW_FILE = Path("data/raw/nyc311_raw.parquet")
CLEAN_FILE = Path("data/processed/clean.parquet")
VALUES_FILE = Path("models/cleaning_values.json")


# ---------------------------------------------------------------- fixed rules
def standardize_text(series):
    """Trim spaces, collapse repeated spaces and upper-case. Placeholders become missing."""
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    is_placeholder = text.str.lower().isin(PLACEHOLDERS)
    # A value that is literally "other" joins the "Other" group used for rare
    # categories. This also means cleaning an already-clean value changes nothing.
    text = text.mask((text == OTHER.upper()).fillna(False), OTHER)
    return text.mask(is_placeholder)


def clean_zip(series):
    """Keep only 5-digit NYC ZIP codes as text. Anything else becomes missing."""
    zip_text = series.astype("string").str.strip()
    valid = (zip_text.str.fullmatch(r"\d{5}", na=False)
             & zip_text.str[:3].isin(NYC_ZIP_PREFIXES)
             & ~zip_text.isin(FAKE_ZIPS))
    # "Other" is the label given to rare ZIP codes by apply_rare_categories;
    # keeping it means cleaning an already-clean value changes nothing.
    return zip_text.where(valid | (zip_text == OTHER).fillna(False))


def clean_coordinates(df):
    """Turn latitude/longitude into numbers. Values outside NYC become missing."""
    df = df.copy()
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    inside = lat.between(*NYC_LAT_RANGE) & lon.between(*NYC_LON_RANGE)
    df["latitude"] = lat.where(inside)     # a point needs both numbers, so
    df["longitude"] = lon.where(inside)    # both are cleared together
    return df


def convert_dates(df, columns=("created_date", "closed_date")):
    """Turn date text into datetimes. Text that is not a date becomes missing."""
    df = df.copy()
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def clean_features(df):
    """Apply every FIXED rule to the feature columns that are present.

    Safe to call on one API request or on the whole dataset: it learns nothing
    from the data it is given.
    """
    df = df.copy()
    if "incident_zip" in df.columns:
        df["incident_zip"] = clean_zip(df["incident_zip"])
    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            # Missing text gets the label "Unknown" so the model can use
            # "this field was empty" as information.
            text = standardize_text(df[column])
            if column == "community_board":
                # values such as "Unspecified BROOKLYN" and "0 Unspecified" name no board
                text = text.mask(text.str.contains("UNSPECIFIED", na=False))
            df[column] = text.fillna(UNKNOWN).astype(object)
    if {"latitude", "longitude"} <= set(df.columns):
        df = clean_coordinates(df)   # missing coordinates stay missing (NaN)
    return df


def add_target(df):
    """Add duration_hours and the target resolved_within_7d (needs datetime columns)."""
    df = df.copy()
    df["duration_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    df["resolved_within_7d"] = (df["duration_hours"] <= THRESHOLD_DAYS * 24).astype(int)
    return df


def drop_unusable_rows(df):
    """Drop rows with no closed_date or a negative duration (no valid target)."""
    usable = df["duration_hours"].notna() & (df["duration_hours"] >= 0)
    return df[usable].copy()


def training_period(df):
    """Return the oldest 70% of rows by created_date (df must be sorted already)."""
    return df.iloc[: int(len(df) * TRAIN_SHARE)]


def missing_table(df):
    """Percent missing per column, counting placeholder text as missing too."""
    rows = []
    for column in df.columns:
        is_null = df[column].isna()
        text = df[column].astype("string").str.strip().str.lower()
        is_placeholder = text.isin(PLACEHOLDERS) & ~is_null
        rows.append({"column": column,
                     "null_pct": 100 * is_null.mean(),
                     "placeholder_pct": 100 * is_placeholder.mean(),
                     "missing_pct": 100 * (is_null | is_placeholder).mean()})
    return pd.DataFrame(rows).set_index("column").sort_values("missing_pct", ascending=False)


# ---------------------------------------------------------------- learned values
def learn_cleaning_values(train):
    """Learn, from TRAINING rows only, which categories are common enough to keep."""
    kept = {}
    for column in CATEGORICAL_COLUMNS:
        counts = train[column].value_counts()
        kept[column] = sorted(counts[counts >= MIN_CATEGORY_ROWS].index)
    return {
        "learned_from": "training period only (oldest 70% of rows by created_date)",
        "train_rows": int(len(train)),
        "train_start": str(train["created_date"].min()),
        "train_end": str(train["created_date"].max()),
        "threshold_days": THRESHOLD_DAYS,
        "min_category_rows": MIN_CATEGORY_ROWS,
        "fill_value_missing_text": UNKNOWN,
        "fill_value_rare_category": OTHER,
        "kept_categories": kept,
    }


def apply_rare_categories(df, values):
    """Replace every category that is not in the learned list with "Other".

    A category the model never saw (for example a new complaint type sent to the
    API) is also not in the list, so it becomes "Other" too.
    """
    df = df.copy()
    for column, kept in values["kept_categories"].items():
        if column in df.columns:
            keep = df[column].isin(kept) | (df[column] == UNKNOWN)
            df[column] = df[column].where(keep, OTHER)
    return df


def save_cleaning_values(values, path=VALUES_FILE):
    Path(path).write_text(json.dumps(values, indent=2), encoding="utf-8")


def load_cleaning_values(path=VALUES_FILE):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- full run
def make_clean_dataset(raw):
    """Run every cleaning step in order. Returns (clean table, learned values)."""
    df = raw.drop_duplicates()                              # fully identical rows
    df = df.drop_duplicates(subset="unique_key")            # repeated request ids
    df = convert_dates(df)
    df = clean_features(df)
    df = add_target(df)
    df = drop_unusable_rows(df)
    df = df.sort_values(["created_date", "unique_key"]).reset_index(drop=True)

    values = learn_cleaning_values(training_period(df))     # training rows only
    df = apply_rare_categories(df, values)

    # Keeping only CLEAN_COLUMNS drops the leakage columns and unused columns.
    clean = df[CLEAN_COLUMNS]
    assert not set(LEAKAGE_COLUMNS) & set(clean.columns), "a leakage column was kept"
    return clean, values


if __name__ == "__main__":
    raw_table = pd.read_parquet(RAW_FILE)
    clean_table, learned = make_clean_dataset(raw_table)
    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_table.to_parquet(CLEAN_FILE, index=False)
    save_cleaning_values(learned)
    print(f"Raw rows: {len(raw_table):,}  ->  clean rows: {len(clean_table):,}, {clean_table.shape[1]} columns")
    print(f"Saved {CLEAN_FILE} and {VALUES_FILE}")
