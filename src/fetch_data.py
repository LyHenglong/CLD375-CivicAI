"""Phase 1: pull a sample of closed NYC 311 requests from the Socrata API.

Run from the project root:  python src/fetch_data.py

A later period can be pulled the same way into its own files (used in Phase 6b
for the forward test set), for example:
    python src/fetch_data.py --start 2026-09-01 --end 2026-10-29 --name forward

What it does
1. Asks the API how many requests there are per status (saved to status_counts.csv).
2. Downloads a 1-in-48 sample of the CLOSED requests, in 4 date windows per month.
   Each window is saved as its own file, so an interrupted run can resume.
3. Joins the pieces into data/raw/nyc311_raw.parquet without changing any value.
4. Prints row count, date range, rows per month, column count and nulls per column.

Nothing is cleaned here. Cleaning happens in Phase 3.
"""

import argparse
import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------- settings
BASE_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9"
START = "2024-01-01"          # first created_date included
END = "2026-09-01"            # first created_date NOT included (so Aug 31 is the last day)
WINDOW_START_DAYS = [1, 9, 17, 25]   # 4 windows per month, covering the whole month
PAGE_SIZE = 50_000            # most rows asked for in one request
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 5

# Sampling: there are about 9.58 million closed requests in the date range, far
# more than we need. We keep every request whose unique_key divides evenly by 48
# (9.58M / 48 = about 200,000). unique_key is just an id number, so this is a
# fair sample of every day and every hour, and it gives the same rows every run.
SAMPLE_EVERY = 48
SAMPLE_FILTER = f"unique_key::number % {SAMPLE_EVERY} = 0"

MIN_ROWS = 150_000            # print a warning if the pull returns fewer rows than this

RAW_DIR = Path("data/raw")
PARTS_DIR = RAW_DIR / "parts"
RAW_FILE = RAW_DIR / "nyc311_raw.parquet"
STATUS_FILE = RAW_DIR / "status_counts.csv"


# ---------------------------------------------------------------- helpers
def get(url, params):
    """GET with retries. Returns the response text. Raises after MAX_RETRIES."""
    headers = {}
    token = os.environ.get("SOCRATA_APP_TOKEN")  # optional; never printed
    if token:
        headers["X-App-Token"] = token

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=120)
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            # Show only the error type and status code, never headers or the token.
            status = getattr(error.response, "status_code", "no response")
            print(f"  attempt {attempt}/{MAX_RETRIES} failed ({type(error).__name__}, {status})")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY_SECONDS)


def make_windows():
    """Return (start, end) date strings: 4 windows for every month in the range."""
    windows = []
    for month_start in pd.date_range(START, END, freq="MS", inclusive="left"):
        next_month = month_start + pd.offsets.MonthBegin(1)
        starts = [month_start.replace(day=d) for d in WINDOW_START_DAYS]
        ends = starts[1:] + [next_month]
        for s, e in zip(starts, ends):
            e = min(e, pd.Timestamp(END))          # a range may stop in the middle of a month
            if s < e:
                windows.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return windows


def count_rows(csv_path):
    """Number of data rows in a saved piece (dtype=str so nothing is converted)."""
    return len(pd.read_csv(csv_path, dtype=str, usecols=["unique_key"]))


# ---------------------------------------------------------------- steps
def save_status_counts():
    """Count ALL requests in the date range by status (not only the closed ones)."""
    params = {
        "$select": "status, count(*) as n",
        "$where": f"created_date >= '{START}T00:00:00' AND created_date < '{END}T00:00:00'",
        "$group": "status",
        "$order": "n DESC",
    }
    counts = pd.read_csv(io.StringIO(get(BASE_URL + ".csv", params)))
    counts["share_pct"] = (100 * counts["n"] / counts["n"].sum()).round(3)
    counts.to_csv(STATUS_FILE, index=False)
    print(f"Saved {STATUS_FILE}")
    print(counts.to_string(index=False))


def download_pieces():
    """Download every window, page by page. Pieces already on disk are skipped."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    windows = make_windows()
    for number, (start, end) in enumerate(windows, 1):
        page = 0
        while True:
            piece = PARTS_DIR / f"{start}_{end}_p{page}.csv"
            if piece.exists():
                rows = count_rows(piece)
                note = "already downloaded"
            else:
                params = {
                    "$where": (
                        f"status = 'Closed' "
                        f"AND created_date >= '{start}T00:00:00' "
                        f"AND created_date < '{end}T00:00:00' "
                        f"AND {SAMPLE_FILTER}"
                    ),
                    "$order": "unique_key",       # fixed order so paging is stable
                    "$limit": PAGE_SIZE,
                    "$offset": page * PAGE_SIZE,
                }
                text = get(BASE_URL + ".csv", params)
                # Write to a temporary name first, so a crash never leaves a half file.
                temp = piece.with_suffix(".tmp")
                temp.write_text(text, encoding="utf-8", newline="")
                temp.replace(piece)
                rows = count_rows(piece)
                note = "downloaded"
            print(f"[{number}/{len(windows)}] {start} to {end} page {page}: {rows} rows ({note})")
            if rows < PAGE_SIZE:   # a short page means the window is finished
                break
            page += 1


def combine_pieces():
    """Join all pieces into one parquet file. Every column stays text, unchanged."""
    frames = [
        # keep_default_na=False: only truly empty cells become null,
        # so text such as "N/A" is kept exactly as the API sent it.
        pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])
        for p in sorted(PARTS_DIR.glob("*.csv"))
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(RAW_FILE, index=False)
    print(f"Saved {RAW_FILE}")
    return raw


def print_summary(raw):
    created = pd.to_datetime(raw["created_date"])   # only for the printout
    print("\n===== SUMMARY =====")
    print(f"Rows:    {len(raw):,}")
    print(f"Columns: {raw.shape[1]}")
    print(f"created_date range: {created.min()}  to  {created.max()}")
    print(f"Status values: {raw['status'].value_counts().to_dict()}")
    print(f"Duplicate unique_key values: {raw['unique_key'].duplicated().sum()}")
    print("\nRows per month:")
    print(created.dt.to_period("M").value_counts().sort_index().to_string())
    print("\nNull count per column:")
    print(raw.isna().sum().to_string())
    if len(raw) < MIN_ROWS:
        print(f"\nWARNING: fewer than {MIN_ROWS:,} rows. The pull is SHORT of the target.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull a 1-in-48 sample of closed NYC 311 requests.")
    parser.add_argument("--start", default=START, help="first created_date included (YYYY-MM-DD)")
    parser.add_argument("--end", default=END, help="first created_date NOT included (YYYY-MM-DD)")
    parser.add_argument("--name", default=None, help="save under this name instead of the main raw file")
    parser.add_argument("--min-rows", type=int, default=MIN_ROWS, help="warn below this row count")
    args = parser.parse_args()
    START, END, MIN_ROWS = args.start, args.end, args.min_rows      # the functions above read these
    if args.name:                                                   # keep the main raw files untouched
        PARTS_DIR = RAW_DIR / f"parts_{args.name}"
        RAW_FILE = RAW_DIR / f"nyc311_{args.name}_raw.parquet"
        STATUS_FILE = RAW_DIR / f"status_counts_{args.name}.csv"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    save_status_counts()
    download_pieces()
    print_summary(combine_pieces())
