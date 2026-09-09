#!/usr/bin/env python3
"""Refresh the offline snapshot under ``data/snapshot/``.

    python scripts/refresh.py                 # from the live FPL API
    python scripts/refresh.py --from-mirror   # from a public GitHub dataset

The live API is the source of truth and should be preferred. The mirror exists
for machines whose network policy blocks fantasy.premierleague.com; it lags the
live API by however long it has been since its maintainer last ran a scrape, and
``meta.json`` records which was used so no report can silently present stale
prices as current.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplpred import data as D  # noqa: E402

MIRROR_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)


def from_live(season: str) -> dict:
    bundle = D.load_live()
    bundle.players.to_csv(D.SNAPSHOT_DIR / "players.csv", index=False)
    bundle.teams.to_csv(D.SNAPSHOT_DIR / "teams.csv", index=False)
    bundle.fixtures.to_csv(D.SNAPSHOT_DIR / "fixtures.csv", index=False)
    return {
        "source": "fantasy.premierleague.com/api",
        "season": season,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "events": bundle.events.to_dict("records") if not bundle.events.empty else [],
    }


def from_mirror(season: str) -> dict:
    base = f"{MIRROR_BASE}/{season}"
    for remote, local in (
        ("players_raw.csv", "players.csv"),
        ("teams.csv", "teams.csv"),
        ("fixtures.csv", "fixtures.csv"),
    ):
        url = f"{base}/{remote}"
        print(f"  fetching {url}")
        pd.read_csv(url).to_csv(D.SNAPSHOT_DIR / local, index=False)
    return {
        "source": f"github:vaastav/Fantasy-Premier-League/{season}",
        "season": season,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Mirror snapshot. Freshness is bounded by the upstream repository's "
            "last scrape, not by this timestamp -- check the completed-gameweek "
            "count in the report header before trusting prices or injury flags."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", default="2026-27")
    ap.add_argument("--from-mirror", action="store_true",
                    help="use the GitHub mirror instead of the live API")
    args = ap.parse_args()

    D.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    meta = from_mirror(args.season) if args.from_mirror else from_live(args.season)
    (D.SNAPSHOT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    bundle = D.load_snapshot()
    print(
        f"snapshot updated: {len(bundle.players)} players, "
        f"{len(bundle.fixtures)} fixtures, "
        f"{bundle.matches_played()} completed gameweeks"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
