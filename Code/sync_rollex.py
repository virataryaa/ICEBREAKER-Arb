"""
Arb — Rollex Sync
=================
Reads the four live Rollex parquets (kept up-to-date by the existing
Rollex Task Scheduler job) and writes slim copies into Arb/Database/.

Keeps only the roll-adjusted price columns needed for spread analysis;
drops all the raw contract, return, and label columns to stay small.

Usage:
    python sync_rollex.py          # normal run
    python sync_rollex.py --full   # force overwrite even if up to date
"""

import argparse
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROLLEX_DB = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\Rollex\Database")
ARB_DB    = Path(__file__).parent.parent / "Database"

SOURCES = {
    "KC":  ROLLEX_DB / "rollex_KC.parquet",
    "RC":  ROLLEX_DB / "rollex_RC.parquet",
    "CC":  ROLLEX_DB / "rollex_CC.parquet",
    "LCC": ROLLEX_DB / "rollex_LCC.parquet",
}

# Columns to keep from each Rollex parquet
KEEP_COLS = ["rollex_px", "rollex_open", "rollex_high", "rollex_low", "rollex_ret"]

# ── Main ──────────────────────────────────────────────────────────────────────

def sync_one(name: str, src: Path, force: bool) -> None:
    out = ARB_DB / f"arb_{name}.parquet"

    if not src.exists():
        log.error("Source not found: %s", src)
        return

    df = pd.read_parquet(src, columns=KEEP_COLS)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if out.exists() and not force:
        existing = pd.read_parquet(out)
        existing.index = pd.to_datetime(existing.index)
        latest = existing.index.max()
        new_rows = df[df.index > latest]
        if new_rows.empty:
            log.info("%-4s  up to date (%s)", name, latest.date())
            return
        combined = pd.concat([existing, new_rows]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = df

    ARB_DB.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, engine="pyarrow")
    log.info("%-4s  saved %d rows to %s  (last: %s)",
             name, len(combined), out.name, combined.index.max().date())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Force full overwrite")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Arb Rollex Sync | %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    for name, src in SOURCES.items():
        sync_one(name, src, args.full)

    log.info("Done.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
