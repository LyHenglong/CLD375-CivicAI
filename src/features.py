"""Features, the chronological split and the model pipeline (Phase 4).

The pipeline has three steps:

    clean   RequestCleaner      raw request fields -> cleaned feature table
    encode  ColumnTransformer   feature table -> numbers (one-hot, scaling)
    model   any classifier      added in Phase 5

The first step accepts the fields exactly as the API receives them, so the
same saved pipeline is used for training and for serving.

Import this file as a package module (from the project root):
    from src import features
Make the split files:
    python -m src.features
"""

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from src import preprocessing as pp

# ---------------------------------------------------------------- feature lists
TARGET = "resolved_within_7d"

# Every candidate input, in three groups (each group is encoded differently)
CATEGORICAL_CANDIDATES = ["complaint_type", "descriptor", "location_type", "borough",
                          "incident_zip", "open_data_channel_type", "agency",
                          "address_type", "community_board"]
COORDINATE_FEATURES = ["latitude", "longitude"]
TIME_FEATURES = ["hour", "weekday", "month", "is_weekend"]     # all derived from created_date
CANDIDATE_FEATURES = CATEGORICAL_CANDIDATES + COORDINATE_FEATURES + TIME_FEATURES

# The features the final model uses. Chosen in notebooks/02_feature_selection.ipynb.
# Dropped: agency, address_type and community_board (a person filing a request cannot supply them).
FINAL_FEATURES = ["complaint_type", "descriptor", "location_type", "borough", "incident_zip",
                  "open_data_channel_type", "latitude", "longitude",
                  "hour", "weekday", "month", "is_weekend"]

# Columns that must NEVER reach the model.
EXCLUDED_COLUMNS = [
    "unique_key",                        # an id, carries no meaning
    "created_date", "created_at",        # the raw timestamp (only hour, weekday, month, weekend are used)
    "year",                              # would let the model memorise a period instead of a pattern
    "duration_hours",                    # the answer, in hours
    "resolved_within_7d",                # the answer
    "closed_date", "status", "resolution_description",      # only known after the request is handled
    "resolution_action_updated_date", "due_date",
]

# The raw fields the pipeline reads. Fields that are absent are treated as missing.
RAW_INPUT_COLUMNS = CATEGORICAL_CANDIDATES + COORDINATE_FEATURES + ["created_date"]

TRAIN_SHARE, VAL_SHARE = 0.70, 0.15      # the newest 15% is the test split
PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------- helpers
def assert_no_excluded(columns):
    """Stop with an error if any excluded column is among the model inputs."""
    leaked = sorted(set(columns) & set(EXCLUDED_COLUMNS))
    assert not leaked, f"excluded columns reached the model: {leaked}"


def add_time_features(df):
    """Add hour, weekday, month and is_weekend from created_date (a datetime column)."""
    df = df.copy()
    created = df["created_date"]
    df["hour"] = created.dt.hour                               # 0 to 23
    df["weekday"] = created.dt.dayofweek                       # 0 = Monday ... 6 = Sunday
    df["month"] = created.dt.month                             # 1 to 12
    df["is_weekend"] = (created.dt.dayofweek >= 5).astype(int)   # 1 = Saturday or Sunday
    return df


def chronological_split(clean):
    """Split by time: oldest 70% train, next 15% validation, newest 15% test."""
    clean = clean.sort_values(["created_date", "unique_key"]).reset_index(drop=True)
    train_end = int(len(clean) * TRAIN_SHARE)
    val_end = int(len(clean) * (TRAIN_SHARE + VAL_SHARE))
    return clean.iloc[:train_end], clean.iloc[train_end:val_end], clean.iloc[val_end:]


def describe_splits(splits):
    """One row per split: rows, date range and share resolved within 7 days."""
    rows = [{"split": name, "rows": len(part), "share_of_rows": len(part) / sum(len(p) for p in splits.values()),
             "from": part["created_date"].min(), "to": part["created_date"].max(),
             "share_resolved_within_7d": part[TARGET].mean()} for name, part in splits.items()]
    return pd.DataFrame(rows).set_index("split")


def load_splits(folder=PROCESSED_DIR):
    """Read the saved train, validation and test files."""
    return {name: pd.read_parquet(Path(folder) / f"{name}.parquet") for name in ["train", "val", "test"]}


# ---------------------------------------------------------------- pipeline step 1
class RequestCleaner(BaseEstimator, TransformerMixin):
    """Pipeline step 1: raw request fields in, cleaned feature table out.

    fit()        learns, from the rows it is given (the training split), which
                 categories are common enough to keep.
    transform()  applies the Phase 3 cleaning rules and adds the time features,
                 then returns ONLY the feature columns. Because nothing else is
                 returned, an excluded column can never reach the model.
    """

    def __init__(self, features=None):
        self.features = features          # None means FINAL_FEATURES

    def _clean(self, X):
        """The cleaning rules that need nothing learned from data."""
        X = pd.DataFrame(X).copy()
        if "created_at" in X.columns and "created_date" not in X.columns:
            X = X.rename(columns={"created_at": "created_date"})   # the API calls it created_at
        for column in RAW_INPUT_COLUMNS:
            if column not in X.columns:
                X[column] = None          # an optional field that was not sent
        X = pp.convert_dates(X, columns=["created_date"])
        return pp.clean_features(X)

    def fit(self, X, y=None):
        self.values_ = pp.learn_cleaning_values(self._clean(X))
        return self

    def transform(self, X):
        X = self._clean(X)
        X = pp.apply_rare_categories(X, self.values_)
        X = add_time_features(X)
        features = self.features or FINAL_FEATURES
        assert_no_excluded(features)
        return X[features]


# ---------------------------------------------------------------- pipeline step 2
def build_binary_encoder(features=None):
    """Like build_encoder, but every output column is 0 or 1.

    Used for Bernoulli Naive Bayes, which expects yes/no features:
    - text columns: one 0/1 column per category (as before);
    - time features: one 0/1 column per hour, per weekday, per month and for the weekend flag;
    - latitude and longitude: missing values filled with the training median, then each
      cut into 10 bands holding equal numbers of training rows, one 0/1 column per band.
    """
    features = features or FINAL_FEATURES
    assert_no_excluded(features)
    categorical = [f for f in features if f in CATEGORICAL_CANDIDATES]
    coordinates = [f for f in features if f in COORDINATE_FEATURES]
    time = [f for f in features if f in TIME_FEATURES]

    steps = []
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
    if coordinates:
        bands = KBinsDiscretizer(n_bins=10, encode="onehot", strategy="quantile",
                                 quantile_method="averaged_inverted_cdf")
        steps.append(("coord", Pipeline([("impute", SimpleImputer(strategy="median")), ("bands", bands)]), coordinates))
    if time:
        steps.append(("time", OneHotEncoder(handle_unknown="ignore", sparse_output=True), time))
    return ColumnTransformer(steps, remainder="drop", sparse_threshold=1.0)


def build_encoder(features=None):
    """Pipeline step 2: turn the cleaned feature table into numbers."""
    features = features or FINAL_FEATURES
    assert_no_excluded(features)
    categorical = [f for f in features if f in CATEGORICAL_CANDIDATES]
    coordinates = [f for f in features if f in COORDINATE_FEATURES]
    time = [f for f in features if f in TIME_FEATURES]

    steps = []
    if categorical:
        # One 0/1 column per category. A category never seen in training gets all zeros.
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical))
    if coordinates:
        # Missing coordinates are filled with the training median, and an extra
        # 0/1 column records that the value was missing. Then both are scaled.
        steps.append(("coord", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                         ("scale", StandardScaler())]), coordinates))
    if time:
        steps.append(("time", StandardScaler(), time))
    # sparse_threshold=1.0 keeps the output sparse (mostly zeros, stored compactly)
    return ColumnTransformer(steps, remainder="drop", sparse_threshold=1.0)


def build_pipeline(model=None, features=None, binary=False):
    """The full pipeline. Without a model it stops after encoding.

    binary=True uses the all-0/1 encoder (for Bernoulli Naive Bayes).
    """
    features = list(features or FINAL_FEATURES)
    encoder = build_binary_encoder(features) if binary else build_encoder(features)
    steps = [("clean", RequestCleaner(features=features)), ("encode", encoder)]
    if model is not None:
        steps.append(("model", model))
    return Pipeline(steps)


def get_stack(model):
    """Return the StackingClassifier inside a saved pipeline.

    In the v2 model the stack is wrapped in a FixedThresholdClassifier (which only
    changes the cut-off used by predict), so the stack itself is one level down.
    """
    step = model["stack"]
    return getattr(step, "estimator_", step)


# ---------------------------------------------------------------- make the split files
if __name__ == "__main__":
    clean = pd.read_parquet(pp.CLEAN_FILE)
    train, val, test = chronological_split(clean)
    splits = {"train": train, "val": val, "test": test}
    for name, part in splits.items():
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    with pd.option_context("display.width", 200):
        print(describe_splits(splits).to_string())
    print(f"Saved train.parquet, val.parquet and test.parquet to {PROCESSED_DIR}")
