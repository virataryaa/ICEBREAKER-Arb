"""
Arb — Front-Month Price Ingest
================================
Fetches 1st and 2nd nearby back-adjusted continuation prices from ICE Connect.
Symbols use the % prefix (%KC 1!, %RC 1!-ICE, etc.) which gives the full
historical series stitched across roll dates (back-adjusted via Panama method).

Note: KC 1! without % only returns today's price — history requires %.
The back-adjustment means absolute spread levels are distorted over long
horizons (different cumulative adjustments per leg), but z-scores within
rolling windows are still valid for relative positioning.

Each parquet has two columns:
    px1  — 1st nearby (front month, back-adjusted)
    px2  — 2nd nearby (back-adjusted)

Units match the raw ICE quote:
    KC  → ¢/lbs    (multiply by 22.0462 for $/MT)
    RC  → $/MT
    CC  → $/MT
    LCC → GBP/MT   (multiply by GBP/USD for $/MT)

Usage:
    python ingest_front.py            # incremental
    python ingest_front.py --full     # full pull from 2014-01-01
    python ingest_front.py --check    # print tail of each file
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import icepython as ice
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

OUT_DIR    = Path(__file__).parent.parent / "Database"
FULL_START = "2014-01-01"

SYMBOLS = {
    "KC":  ("%KC 1!",      "%KC 2!"),
    "RC":  ("%RC 1!-ICE",  "%RC 2!-ICE"),
    "CC":  ("%CC 1!",      "%CC 2!"),
    "LCC": ("%C 1!-ICE",   "%C 2!-ICE"),   # ICE EU London Cocoa root is C, not LCC
}

# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch_series(sym: str, start: str, end: str) -> pd.Series:
    try:
        data = ice.get_timeseries(sym, ["Close"], granularity="D",
                                  start_date=start, end_date=end)
        df = pd.DataFrame(list(data))
        if df.empty or "Error" in str(df.iloc[0, 0]):
            log.warning("No data for %s", sym)
            return pd.Series(dtype=float)
        df.columns = ["Date", "Val"]
        df = df[1:].reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Val"]  = pd.to_numeric(df["Val"], errors="coerce")
        return df.set_index("Date")["Val"]
    except Exception as e:
        log.warning("Fetch failed for %s: %s", sym, e)
        return pd.Series(dtype=float)


def fetch_pair(name: str, sym1: str, sym2: str, start: str, end: str) -> pd.DataFrame:
    log.info("%-4s  fetching %s and %s", name, sym1, sym2)
    s1 = _fetch_series(sym1, start, end).rename("px1")
    s2 = _fetch_series(sym2, start, end).rename("px2")
    df = pd.concat([s1, s2], axis=1)
    df.index.name = "Date"
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",  action="store_true", help="Full pull from 2014-01-01")
    parser.add_argument("--check", action="store_true", help="Print tail of each file and exit")
    args = parser.parse_args()

    if args.check:
        for name in SYMBOLS:
            path = OUT_DIR / f"front_{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n=== {name} ===")
                print(df.tail(5).to_string())
            else:
                print(f"\n=== {name} ===  [not found]")
        return

    log.info("=" * 60)
    log.info("Front-Month Ingest | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    end = datetime.date.today().isoformat()

    for name, (sym1, sym2) in SYMBOLS.items():
        out = OUT_DIR / f"front_{name}.parquet"

        if args.full or not out.exists():
            start = FULL_START
            log.info("%-4s  mode: FULL from %s", name, start)
        else:
            existing = pd.read_parquet(out)
            existing.index = pd.to_datetime(existing.index)
            latest = existing.index.max()
            start  = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            log.info("%-4s  mode: INCREMENTAL from %s", name, start)

        new_df = fetch_pair(name, sym1, sym2, start, end)

        if new_df.dropna(how="all").empty:
            log.error("%-4s  no data returned — check symbol", name)
            continue

        if out.exists() and not args.full:
            old_df = pd.read_parquet(out)
            old_df.index = pd.to_datetime(old_df.index)
            combined = pd.concat([old_df, new_df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = new_df.sort_index()

        # Drop today (intraday partial)
        combined = combined[combined.index < pd.Timestamp.today().normalize()]
        combined = combined.dropna(how="all")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out, engine="pyarrow")
        log.info("%-4s  saved %d rows to %s  (last: %s  px1: %.2f  px2: %.2f)",
                 name, len(combined), out.name,
                 combined.index.max().date(),
                 combined["px1"].iloc[-1] if not combined["px1"].isna().all() else float("nan"),
                 combined["px2"].iloc[-1] if not combined["px2"].isna().all() else float("nan"))

    log.info("Done.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
