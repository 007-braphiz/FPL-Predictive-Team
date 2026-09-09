"""Per-player rate estimation: minutes, availability, and per-90 scoring rates.

Everything here answers one question -- *what will this player do per 90 minutes
of football, and how many of those minutes will he actually get?* -- while
resisting the single biggest early-season trap: treating two or three matches of
data as if it were a season.

The tool for that is empirical-Bayes shrinkage. An observed rate is blended with
a prior in proportion to how much evidence supports it::

    estimate = (observed * minutes + prior * k) / (minutes + k)

``k`` is expressed in minutes and lives in ``config.RATE_SHRINK_MINUTES``. With
270 minutes played and ``k = 700``, the observed rate carries roughly 28% of the
weight -- which is about right for three gameweeks.

The prior itself is a two-stage construction: a player's own previous season
(regressed toward his positional mean, because last season also regresses), and
for anyone without one -- promoted players, new signings, teenagers -- the
positional mean alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import DataBundle

# (output column, current-season total column, shrinkage key)
RATE_SPECS = [
    ("xg90", "expected_goals", "xg90"),
    ("xa90", "expected_assists", "xa90"),
    ("dc90", "defensive_contribution", "dc90"),
    ("saves90", "saves", "saves90"),
    ("bonus90", "bonus", "bonus90"),
    ("yellow90", "yellow_cards", "yellow90"),
]
PRIOR_COLS = {
    "xg90": "prior_xg90",
    "xa90": "prior_xa90",
    "dc90": "prior_dc90",
    "saves90": "prior_saves90",
    "bonus90": "prior_bonus90",
    "yellow90": "prior_yellow90",
}


def _positional_means(prior: pd.DataFrame) -> dict[str, dict[int, float]]:
    """Minutes-weighted league averages per position, from the previous season.

    Minutes weighting matters: an unweighted mean is dragged down by the long
    tail of players who made three substitute appearances all year.
    """
    means: dict[str, dict[int, float]] = {}
    if prior.empty:
        return {key: {} for key in PRIOR_COLS}

    regulars = prior[prior["prior_minutes"] >= 450]
    source = regulars if len(regulars) > 40 else prior

    for key, col in PRIOR_COLS.items():
        by_pos: dict[int, float] = {}
        for pos, grp in source.groupby("prior_pos"):
            w = grp["prior_minutes"].to_numpy(dtype=float)
            v = grp[col].to_numpy(dtype=float)
            by_pos[int(pos)] = float(np.average(v, weights=w)) if w.sum() else 0.0
        means[key] = by_pos
    return means


def _fallback_mean(key: str, pos: int) -> float:
    """Last-resort positional priors if no history file is available at all."""
    table = {
        "xg90": {C.GK: 0.00, C.DEF: 0.06, C.MID: 0.17, C.FWD: 0.38},
        "xa90": {C.GK: 0.00, C.DEF: 0.07, C.MID: 0.15, C.FWD: 0.14},
        "dc90": {C.GK: 0.0, C.DEF: 7.4, C.MID: 7.0, C.FWD: 3.6},
        "saves90": {C.GK: 3.0, C.DEF: 0.0, C.MID: 0.0, C.FWD: 0.0},
        "bonus90": {C.GK: 0.16, C.DEF: 0.16, C.MID: 0.18, C.FWD: 0.24},
        "yellow90": {C.GK: 0.06, C.DEF: 0.19, C.MID: 0.19, C.FWD: 0.15},
    }
    return table[key].get(pos, 0.0)


def availability(row: pd.Series) -> float:
    """Probability the player is fit and eligible, in [0, 1].

    ``chance_of_playing_next_round`` is the club's own reported figure and is
    trusted when present; otherwise the coarser ``status`` flag is used.
    """
    chance = row.get("chance_of_playing_next_round")
    if chance is not None and not pd.isna(chance):
        return float(np.clip(float(chance) / 100.0, 0.0, 1.0))
    return C.STATUS_AVAILABILITY.get(str(row.get("status", "a")).lower(), 1.0)


def build_rates(bundle: DataBundle, team_matches: dict[int, int]) -> pd.DataFrame:
    """Return one row per player with shrunk per-90 rates and a minutes model."""
    players = bundle.players.copy()
    prior = bundle.prior_season
    means = _positional_means(prior)

    if not prior.empty and "code" in players.columns:
        players["code"] = pd.to_numeric(players["code"], errors="coerce")
        players = players.merge(prior, on="code", how="left")
    for col in list(PRIOR_COLS.values()) + ["prior_minutes", "prior_start_rate"]:
        if col not in players.columns:
            players[col] = np.nan

    players["prior_minutes"] = players["prior_minutes"].fillna(0.0)
    pos = players["element_type"].astype(int)
    mins = players["minutes"].astype(float)
    players["team_matches"] = players["team"].astype(int).map(team_matches).fillna(0)

    # ---------------- per-90 rates ----------------
    for out_col, total_col, key in RATE_SPECS:
        pos_mean = pos.map(
            lambda p, k=key: means.get(k, {}).get(p, _fallback_mean(k, p))
        ).astype(float)

        # A player's own previous season, regressed toward his positional mean.
        own_prior = players[PRIOR_COLS[key]].astype(float)
        has_history = (players["prior_minutes"] >= C.PRIOR_SEASON_MIN_MINUTES) & own_prior.notna()
        blended_prior = np.where(
            has_history,
            (1 - C.PRIOR_SEASON_REGRESSION) * own_prior.fillna(0.0)
            + C.PRIOR_SEASON_REGRESSION * pos_mean,
            pos_mean,
        )

        observed = np.where(mins > 0, players[total_col].astype(float) * 90.0 / mins.clip(lower=1.0), 0.0)
        k = C.RATE_SHRINK_MINUTES[key]
        players[out_col] = (observed * mins + blended_prior * k) / (mins + k)
        players[f"{out_col}_obs"] = observed
        players[f"{out_col}_prior"] = blended_prior

    # Goalkeepers are the only players who record saves; zero the rest so a
    # stray data point cannot leak save points to an outfielder.
    players.loc[pos != C.GK, "saves90"] = 0.0
    players.loc[pos == C.GK, "xg90"] = players.loc[pos == C.GK, "xg90"] * 0.0

    # ---------------- minutes model ----------------
    tm = players["team_matches"].astype(float)
    starts = players["starts"].astype(float)

    obs_start = np.where(tm > 0, (starts / tm.clip(lower=1.0)), 0.0).clip(0.0, 1.0)

    # Prior on starting: last season's start rate, else a price-based guess
    # (FPL prices are set from expected involvement, so they carry real signal).
    price_prior = np.clip(0.22 + (players["price"].astype(float) - 4.0) * 0.10, 0.18, 0.85)
    start_prior = np.where(
        players["prior_minutes"] >= C.PRIOR_SEASON_MIN_MINUTES,
        players["prior_start_rate"].fillna(0.4).astype(float),
        price_prior,
    )

    w = tm / (tm + C.START_SHRINK_MATCHES)
    p_start = (1 - w) * start_prior + w * obs_start

    # Substitute appearances: minutes that cannot be explained by starts.
    cameo_minutes = (mins - starts * C.MINUTES_IF_START).clip(lower=0.0)
    obs_cameo = np.where(
        tm > 0, (cameo_minutes / C.MINUTES_IF_CAMEO) / tm.clip(lower=1.0), 0.0
    )
    p_cameo = np.clip((1 - w) * 0.12 + w * obs_cameo, 0.0, 1.0)
    p_cameo = np.minimum(p_cameo, np.clip(0.97 - p_start, 0.0, 1.0))

    avail = players.apply(availability, axis=1).to_numpy(dtype=float)
    players["availability"] = avail
    players["p_start"] = p_start * avail
    players["p_cameo"] = p_cameo * avail
    players["p_play"] = players["p_start"] + players["p_cameo"]
    players["p_60"] = (
        players["p_start"] * C.P60_GIVEN_START + players["p_cameo"] * C.P60_GIVEN_CAMEO
    )
    players["exp_minutes"] = (
        players["p_start"] * C.MINUTES_IF_START + players["p_cameo"] * C.MINUTES_IF_CAMEO
    )

    players["is_pen_taker"] = players["penalties_order"].fillna(99) <= 1
    players["pos_name"] = pos.map(C.POS_NAME)
    players["team_name"] = players["team"].astype(int).map(
        {int(r.id): r.short_name for r in bundle.teams.itertuples()}
    )
    return players
