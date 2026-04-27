"""
Arb — GBP/USD Ingest
=====================
Fetches GBP/USD spot rate from ICE Connect and saves to Arb/Database/fx_gbp.parquet.

ICE symbol follows the same FXP convention as the Cocoa Currency ingest:
    GBP@FXP A0-FX  →  rate stored as GBP_USD (1 GBP = X USD, e.g. 1.3566)

If ICE returns the inverse (1 USD = X GBP) the stored values will be ~0.73.
Run with --check to print the last 5 rows and verify direction.
If inverted, pass --invert to store 1/rate.

Usage:
    python ingest_gbp.py            # incremental
    python ingest_gbp.py --full     # full pull from 2014-01-01
    python ingest_gbp.py --check    # print tail to verify rate direction
    python ingest_gbp.py --invert   # store 1/rate if ICE returns USD/GBP
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

OUT_FILE   = Path(__file__).parent.parent / "Database" / "fx_gbp.parquet"
FULL_START = "2014-01-01"

# Primary symbol to try — follows FXP convention from cocoa_ingest.py
GBP_SYM = "GBP@FXP A0-FX"

# ── Fetch ─────────────────────────────────────────────────────────────────────

def _fetch(sym: str, start: str, end: str) -> pd.Series:
    try:
        data = ice.get_timeseries(sym, ["Close"], granularity="D",
                                  start_date=start, end_date=end)
        df = pd.DataFrame(list(data))
        if df.empty or "Error" in str(df.iloc[0, 0]):
            return pd.Series(dtype=float)
        df.columns = ["Date", "Val"]
        df = df[1:].reset_index(drop=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Val"]  = pd.to_numeric(df["Val"], errors="coerce")
        return df.set_index("Date")["Val"].rename("GBP_USD")
    except Exception as e:
        log.warning("Fetch failed for %s: %s", sym, e)
        return pd.Series(dtype=float)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   action="store_true", help="Full pull from 2014-01-01")
    parser.add_argument("--check",  action="store_true", help="Print last rows and exit")
    parser.add_argument("--invert", action="store_true", help="Store 1/rate (if ICE returns USD/GBP)")
    args = parser.parse_args()

    if args.check and OUT_FILE.exists():
        df = pd.read_parquet(OUT_FILE)
        print(df.tail(10).to_string())
        print(f"\nLast date: {df.index.max().date()}   Last GBP_USD: {df['GBP_USD'].iloc[-1]:.4f}")
        print("Expected: ~1.25–1.40 (1 GBP = X USD). If ~0.72, run with --invert.")
        return

    log.info("=" * 60)
    log.info("GBP/USD Ingest | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    if args.full or not OUT_FILE.exists():
        start = FULL_START
        log.info("Mode: FULL from %s", start)
    else:
        existing = pd.read_parquet(OUT_FILE)
        latest   = existing.index.max()
        start    = (latest - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        log.info("Mode: INCREMENTAL from %s (overlap 5 days)", start)

    end = datetime.date.today().isoformat()

    log.info("Fetching %s", GBP_SYM)
    series = _fetch(GBP_SYM, start, end)

    if series.empty:
        log.error("No data returned for %s. Check the symbol in ICE Connect.", GBP_SYM)
        log.error("Try: ice.search_symbols('GBP') to find the correct code, then update GBP_SYM.")
        sys.exit(1)

    if args.invert:
        series = (1 / series).rename("GBP_USD")
        log.info("--invert applied: storing 1/rate")

    new_df = series.to_frame()
    new_df.index.name = "Date"

    if OUT_FILE.exists() and not args.full:
        old_df = pd.read_parquet(OUT_FILE)
        combined = (pd.concat([old_df, new_df])
                    .loc[~pd.concat([old_df, new_df]).index.duplicated(keep="last")]
                    .sort_index())
    else:
        combined = new_df.sort_index()

    # Drop today (intraday partial)
    combined = combined[combined.index < pd.Timestamp.today().normalize()]
    combined = combined.dropna()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_FILE, engine="pyarrow")
    log.info("Saved %d rows to %s  (last: %s  rate: %.4f)",
             len(combined), OUT_FILE.name,
             combined.index.max().date(), combined["GBP_USD"].iloc[-1])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
