"""Data access layer.

Two interchangeable sources:

* ``live``     - the public Fantasy Premier League JSON API. Always current,
                 needs outbound network access to fantasy.premierleague.com.
* ``snapshot`` - CSV files under ``data/snapshot/``. Always available, only as
                 fresh as the last ``scripts/refresh.py`` run.

Both produce the same three frames (``players``, ``teams``, ``fixtures``) plus a
``PriorSeason`` table, so the rest of the package never needs to know which one
was used. The ``DataBundle.vintage`` field records the provenance and is echoed
into every report -- a projection is only as trustworthy as the data behind it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshot"
HISTORY_DIR = REPO_ROOT / "data" / "history"

FPL_BASE = "https://fantasy.premierleague.com/api"
BOOTSTRAP_URL = f"{FPL_BASE}/bootstrap-static/"
FIXTURES_URL = f"{FPL_BASE}/fixtures/"
ENTRY_PICKS_URL = f"{FPL_BASE}/entry/{{entry_id}}/event/{{event}}/picks/"

# Columns the engine relies on. Anything missing is filled with a safe default
# so a schema change upstream degrades rather than crashes.
PLAYER_NUMERIC_COLS = [
    "now_cost", "total_points", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "saves", "bonus", "bps", "yellow_cards",
    "red_cards", "own_goals", "penalties_missed", "penalties_saved",
    "expected_goals", "expected_assists", "expected_goals_conceded",
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goals_conceded_per_90", "saves_per_90",
    "defensive_contribution", "defensive_contribution_per_90",
    "selected_by_percent", "form", "points_per_game", "ep_next",
]
PLAYER_STR_COLS = ["web_name", "first_name", "second_name", "status", "news"]


@dataclass
class DataBundle:
    """Everything the model needs, plus a record of where it came from."""

    players: pd.DataFrame
    teams: pd.DataFrame
    fixtures: pd.DataFrame
    prior_season: pd.DataFrame
    events: pd.DataFrame
    source: str
    vintage: str
    warnings: list[str] = field(default_factory=list)

    @property
    def next_gw(self) -> int:
        """The gameweek the model should project: the first unfinished one."""
        unfinished = self.fixtures[~self.fixtures["finished"]]
        if unfinished.empty:
            return int(self.fixtures["event"].max())
        return int(unfinished["event"].min())

    def team_name(self, team_id: int) -> str:
        row = self.teams.loc[self.teams["id"] == team_id]
        return str(row["short_name"].iloc[0]) if len(row) else f"T{team_id}"

    def matches_played(self) -> int:
        """Gameweeks completed so far (drives every shrinkage weight)."""
        finished = self.fixtures[self.fixtures["finished"]]
        return 0 if finished.empty else int(finished["event"].max())


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _coerce_players(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in PLAYER_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in PLAYER_STR_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df["team"] = pd.to_numeric(df["team"], errors="coerce").astype("Int64")
    df["element_type"] = pd.to_numeric(df["element_type"], errors="coerce").astype("Int64")

    # `chance_of_playing_next_round` is null for fit players and 0-100 otherwise.
    if "chance_of_playing_next_round" not in df.columns:
        df["chance_of_playing_next_round"] = pd.NA
    df["chance_of_playing_next_round"] = pd.to_numeric(
        df["chance_of_playing_next_round"], errors="coerce"
    )

    # Set-piece / penalty order: 1 = first choice, null = not on them.
    for col in ("penalties_order", "corners_and_indirect_freekicks_order",
                "direct_freekicks_order"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["price"] = df["now_cost"] / 10.0
    return df.dropna(subset=["id", "team", "element_type"])


def _coerce_teams(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    for col in ("strength", "strength_overall_home", "strength_overall_away",
                "strength_attack_home", "strength_attack_away",
                "strength_defence_home", "strength_defence_away"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.dropna(subset=["id"])


def _coerce_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("event", "team_h", "team_a", "team_h_difficulty",
                "team_a_difficulty", "team_h_score", "team_a_score", "minutes"):
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "finished" not in df.columns:
        df["finished"] = False
    df["finished"] = df["finished"].astype(str).str.lower().isin(["true", "1"])
    if "kickoff_time" not in df.columns:
        df["kickoff_time"] = ""
    df["kickoff_time"] = df["kickoff_time"].fillna("").astype(str)
    # Fixtures with no event are unscheduled postponements; they cannot be
    # projected into a specific gameweek.
    return df.dropna(subset=["event", "team_h", "team_a"]).astype(
        {"event": int, "team_h": int, "team_a": int}
    )


# ---------------------------------------------------------------------------
# Prior season
# ---------------------------------------------------------------------------

def load_prior_season() -> pd.DataFrame:
    """Per-90 rates from the previous completed season, keyed by FPL player code.

    ``code`` (not ``id``) is the stable cross-season identifier -- ``id`` is
    reassigned every summer, so joining on it would silently mismatch players.
    """
    path = HISTORY_DIR / "players_2025-26.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["code", "prior_pos", "prior_minutes", "prior_xg90",
                     "prior_xa90", "prior_dc90", "prior_saves90",
                     "prior_bonus90", "prior_yellow90", "prior_start_rate"]
        )
    raw = pd.read_csv(path)
    df = _coerce_players(raw)
    mins = df["minutes"].clip(lower=1.0)
    per90 = 90.0 / mins
    out = pd.DataFrame(
        {
            "code": pd.to_numeric(df["code"], errors="coerce"),
            "prior_pos": df["element_type"].astype("Int64"),
            "prior_minutes": df["minutes"],
            "prior_xg90": df["expected_goals"] * per90,
            "prior_xa90": df["expected_assists"] * per90,
            "prior_dc90": df["defensive_contribution"] * per90,
            "prior_saves90": df["saves"] * per90,
            "prior_bonus90": df["bonus"] * per90,
            "prior_yellow90": df["yellow_cards"] * per90,
            # 38 league matches in a Premier League season.
            "prior_start_rate": (df["starts"] / 38.0).clip(0.0, 1.0),
        }
    )
    return out.dropna(subset=["code"])


# ---------------------------------------------------------------------------
# Snapshot loader
# ---------------------------------------------------------------------------

def load_snapshot() -> DataBundle:
    players_path = SNAPSHOT_DIR / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(
            f"No snapshot at {SNAPSHOT_DIR}. Run scripts/refresh.py first."
        )
    players = _coerce_players(pd.read_csv(players_path))
    teams = _coerce_teams(pd.read_csv(SNAPSHOT_DIR / "teams.csv"))
    fixtures = _coerce_fixtures(pd.read_csv(SNAPSHOT_DIR / "fixtures.csv"))

    meta_path = SNAPSHOT_DIR / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    vintage = meta.get(
        "captured_at",
        datetime.fromtimestamp(players_path.stat().st_mtime, tz=timezone.utc)
        .isoformat(timespec="seconds"),
    )

    warnings: list[str] = []
    played = 0 if fixtures[fixtures["finished"]].empty else int(
        fixtures[fixtures["finished"]]["event"].max()
    )
    warnings.append(
        f"Offline snapshot: player form/prices reflect the state after GW{played}. "
        "Prices and injury news drift daily -- re-run with --source live before "
        "committing transfers."
    )
    return DataBundle(
        players=players,
        teams=teams,
        fixtures=fixtures,
        prior_season=load_prior_season(),
        events=pd.DataFrame(meta.get("events", [])),
        source="snapshot",
        vintage=vintage,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Live loader
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: int = 30) -> Any:
    import requests  # imported lazily so offline use needs no dependency

    resp = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "fplpred/0.1 (+https://github.com/)"},
    )
    resp.raise_for_status()
    return resp.json()


def load_live() -> DataBundle:
    """Pull the current state straight from the FPL API.

    Endpoints used (public, unauthenticated):
      GET /api/bootstrap-static/  -> elements, teams, events
      GET /api/fixtures/          -> every fixture with difficulty ratings
    """
    boot = _get_json(BOOTSTRAP_URL)
    players = _coerce_players(pd.DataFrame(boot["elements"]))
    teams = _coerce_teams(pd.DataFrame(boot["teams"]))
    events = pd.DataFrame(boot.get("events", []))
    fixtures = _coerce_fixtures(pd.DataFrame(_get_json(FIXTURES_URL)))

    return DataBundle(
        players=players,
        teams=teams,
        fixtures=fixtures,
        prior_season=load_prior_season(),
        events=events,
        source="live",
        vintage=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        warnings=[],
    )


def load(source: str = "auto") -> DataBundle:
    """Load a bundle. ``auto`` tries the live API and falls back to the snapshot."""
    if source == "live":
        return load_live()
    if source == "snapshot":
        return load_snapshot()
    if source != "auto":
        raise ValueError(f"unknown source {source!r}; use live|snapshot|auto")
    try:
        return load_live()
    except Exception as exc:  # network blocked, API down, schema change
        bundle = load_snapshot()
        bundle.warnings.insert(
            0, f"Live FPL API unreachable ({type(exc).__name__}: {exc}). "
               "Fell back to the offline snapshot."
        )
        return bundle


# ---------------------------------------------------------------------------
# Squad loading
# ---------------------------------------------------------------------------

def load_my_team(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read the manager's squad from ``data/my_team.json``."""
    p = Path(path) if path else REPO_ROOT / "data" / "my_team.json"
    return json.loads(Path(p).read_text())


def fetch_entry_picks(entry_id: int, event: int) -> dict[str, Any]:
    """Fetch a real FPL squad by team id, so my_team.json can be regenerated.

    ``entry_id`` is the number in your team's URL:
    fantasy.premierleague.com/entry/<entry_id>/event/<gw>
    """
    return _get_json(ENTRY_PICKS_URL.format(entry_id=entry_id, event=event))


def resolve_squad(bundle: DataBundle, squad: dict[str, Any]) -> pd.DataFrame:
    """Turn the ids or names in ``my_team.json`` into rows of the player table.

    Names are matched case-insensitively against ``web_name`` and are rejected
    if ambiguous -- guessing which "Fernandes" was meant is exactly the kind of
    silent error that ruins a gameweek.
    """
    players = bundle.players
    rows, problems = [], []
    for entry in squad["squad"]:
        if isinstance(entry, dict):
            pid, name = entry.get("id"), entry.get("name")
        else:
            pid, name = None, str(entry)

        if pid is not None:
            match = players[players["id"] == int(pid)]
        else:
            key = str(name).strip().casefold()
            match = players[players["web_name"].str.casefold() == key]
            if len(match) != 1:
                match = players[
                    players["web_name"].str.casefold().str.contains(key, regex=False)
                ]
        if len(match) == 1:
            rows.append(match.iloc[0])
        elif len(match) == 0:
            problems.append(f"no player matches {name or pid!r}")
        else:
            options = ", ".join(
                f"{r.web_name} ({bundle.team_name(int(r.team))}, id={int(r.id)})"
                for r in match.itertuples()
            )
            problems.append(f"{name!r} is ambiguous -- could be: {options}")

    if problems:
        raise ValueError("Could not resolve squad:\n  - " + "\n  - ".join(problems))
    return pd.DataFrame(rows).reset_index(drop=True)
