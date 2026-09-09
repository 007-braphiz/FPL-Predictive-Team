"""Team attack / defence ratings and per-fixture expected goals.

The chain is:

    ratings  ->  expected goals in a specific fixture  ->  multipliers that
    scale a player's own per-90 rates for that fixture.

Ratings are multiplicative and normalised to a league mean of 1.0, so
``attack = 1.30`` means "scores 30% more than an average side against average
opposition". Two sources are blended:

* a **prior** built from FPL's own fixture difficulty ratings, which encode
  bookmaker-like opinion and are available before a ball is kicked; and
* **observed** expected goals for and against so far this season.

Early in a season the prior dominates -- deliberately. Three matches of xG is
not enough to conclude a promoted side has become good, and a model that
overreacts to it will hand you a captain pick built on noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import DataBundle


@dataclass
class TeamRatings:
    attack: dict[int, float]        # >1 = scores more than average
    defence: dict[int, float]       # >1 = concedes fewer than average
    matches: dict[int, int]         # league matches completed, per team
    weight: float                   # how much observed data was trusted (0..1)
    detail: pd.DataFrame

    def expected_goals(self, home_id: int, away_id: int) -> tuple[float, float]:
        """Expected goals for (home, away) in a single fixture."""
        base = C.LEAGUE_GOALS_PER_TEAM
        lam_h = base * self.attack[home_id] / self.defence[away_id] * C.HOME_ADVANTAGE
        lam_a = base * self.attack[away_id] / self.defence[home_id] / C.HOME_ADVANTAGE
        return max(lam_h, 0.15), max(lam_a, 0.15)

    def baseline_goals(self, team_id: int) -> float:
        """Goals the team would expect against an average side at a neutral venue.

        This is the denominator that turns an absolute fixture projection into a
        multiplier on a player's season-average per-90 rate.
        """
        return max(C.LEAGUE_GOALS_PER_TEAM * self.attack[team_id], 0.15)

    def baseline_conceded(self, team_id: int) -> float:
        return max(C.LEAGUE_GOALS_PER_TEAM / self.defence[team_id], 0.15)


def _prior_from_fdr(bundle: DataBundle) -> pd.DataFrame:
    """Infer each team's quality from the difficulty its opponents are assigned.

    ``team_h_difficulty`` is how hard the fixture is *for the home side*, so it
    is a statement about the away side's quality. Averaging every such rating
    aimed at a team over the whole season gives a stable quality index that is
    free of the current season's small-sample noise.
    """
    fx = bundle.fixtures
    ratings: dict[int, list[float]] = {int(t): [] for t in bundle.teams["id"]}

    for row in fx.itertuples():
        # Difficulty faced by the home team describes the away team's quality.
        if not math.isnan(row.team_h_difficulty):
            ratings.setdefault(int(row.team_a), []).append(float(row.team_h_difficulty))
        if not math.isnan(row.team_a_difficulty):
            ratings.setdefault(int(row.team_h), []).append(float(row.team_a_difficulty))

    rows = []
    for team_id, vals in ratings.items():
        mean_fdr = float(np.mean(vals)) if vals else 3.0
        # Interpolate the FDR -> strength map at the (fractional) mean rating.
        keys = sorted(C.FDR_TO_STRENGTH)
        quality = float(
            np.interp(mean_fdr, keys, [C.FDR_TO_STRENGTH[k] for k in keys])
        )
        rows.append({"team": team_id, "mean_fdr": mean_fdr, "quality": quality})

    df = pd.DataFrame(rows)
    df["quality"] /= df["quality"].mean()
    return df


def _observed(bundle: DataBundle) -> pd.DataFrame:
    """Season-to-date expected goals for and against, per team per match."""
    players, fx = bundle.players, bundle.fixtures
    finished = fx[fx["finished"]]

    played: dict[int, int] = {int(t): 0 for t in bundle.teams["id"]}
    for row in finished.itertuples():
        played[int(row.team_h)] = played.get(int(row.team_h), 0) + 1
        played[int(row.team_a)] = played.get(int(row.team_a), 0) + 1

    rows = []
    for team_id in bundle.teams["id"]:
        team_id = int(team_id)
        squad = players[players["team"] == team_id]
        n = played.get(team_id, 0)

        # A team's total player xG is, by construction, the team's xG.
        xg_for = float(squad["expected_goals"].sum()) / n if n else np.nan

        # Goalkeepers between them cover essentially every minute, so their
        # combined xGC divided by their combined 90s is the team's xGC/match.
        keepers = squad[squad["element_type"] == C.GK]
        gk_90s = float(keepers["minutes"].sum()) / 90.0
        xg_against = (
            float(keepers["expected_goals_conceded"].sum()) / gk_90s
            if gk_90s > 0.5
            else np.nan
        )
        rows.append(
            {"team": team_id, "played": n, "xg_for": xg_for, "xg_against": xg_against}
        )
    return pd.DataFrame(rows)


def compute_ratings(bundle: DataBundle) -> TeamRatings:
    prior = _prior_from_fdr(bundle)
    obs = _observed(bundle)
    df = prior.merge(obs, on="team", how="outer")

    matches = int(df["played"].fillna(0).max())
    weight = matches / (matches + C.TEAM_SHRINK_MATCHES) if matches else 0.0

    league_xg_for = df["xg_for"].mean(skipna=True)
    league_xg_against = df["xg_against"].mean(skipna=True)

    # Observed ratings, normalised so the league averages 1.0. Defence is
    # inverted: conceding less than average must score above 1.0.
    if matches and league_xg_for and not math.isnan(league_xg_for):
        df["attack_obs"] = (df["xg_for"] / league_xg_for).fillna(1.0)
    else:
        df["attack_obs"] = 1.0
    if matches and league_xg_against and not math.isnan(league_xg_against):
        df["defence_obs"] = (league_xg_against / df["xg_against"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)
    else:
        df["defence_obs"] = 1.0

    # Winsorise before blending. A side that conceded 0.2 xG in its opening
    # match produces a raw defence rating near 8.0, and even at a small blend
    # weight that would drag the final rating to the clamp ceiling -- turning
    # one match into a season-long verdict. Capping the *observation* keeps the
    # blend weight meaningful instead of letting the clamp do the work.
    lo_obs, hi_obs = C.OBSERVED_RATING_CLAMP
    df["attack_obs"] = df["attack_obs"].clip(lo_obs, hi_obs)
    df["defence_obs"] = df["defence_obs"].clip(lo_obs, hi_obs)

    # The FDR prior is a single quality index; a strong team is assumed both to
    # score more and concede less, in equal measure.
    df["attack_prior"] = df["quality"]
    df["defence_prior"] = df["quality"]

    df["attack"] = (1 - weight) * df["attack_prior"] + weight * df["attack_obs"]
    df["defence"] = (1 - weight) * df["defence_prior"] + weight * df["defence_obs"]

    # Re-normalise and clamp so no single team distorts the whole fixture grid.
    lo, hi = C.FIXTURE_MULT_CLAMP
    for col in ("attack", "defence"):
        df[col] = (df[col] / df[col].mean()).clip(lo, hi)

    df = df.merge(
        bundle.teams[["id", "short_name", "name"]].rename(columns={"id": "team"}),
        on="team",
        how="left",
    )

    return TeamRatings(
        attack={int(r.team): float(r.attack) for r in df.itertuples()},
        defence={int(r.team): float(r.defence) for r in df.itertuples()},
        matches={int(r.team): int(r.played or 0) for r in df.itertuples()},
        weight=weight,
        detail=df.sort_values("attack", ascending=False).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Per-fixture multipliers
# ---------------------------------------------------------------------------

@dataclass
class FixtureContext:
    """Everything one player-fixture pairing needs from the team model."""

    gw: int
    team: int
    opponent: int
    opponent_short: str
    is_home: bool
    kickoff: str
    team_goals: float          # expected goals for this player's team
    team_conceded: float       # expected goals against
    attack_mult: float         # scale factor on the player's xG90 / xA90
    dc_mult: float             # scale factor on defensive-action volume
    saves_mult: float          # scale factor on goalkeeper save volume
    fdr: int


def fixture_contexts(
    bundle: DataBundle, ratings: TeamRatings, gameweeks: list[int]
) -> dict[int, list[FixtureContext]]:
    """Map each team id to its fixtures across ``gameweeks``.

    A team may appear twice in a gameweek (a double) or not at all (a blank);
    the returned list length reflects that, and the rest of the engine simply
    sums over whatever is there.
    """
    lo, hi = C.FIXTURE_MULT_CLAMP
    out: dict[int, list[FixtureContext]] = {int(t): [] for t in bundle.teams["id"]}
    fx = bundle.fixtures
    short = {int(r.id): str(r.short_name) for r in bundle.teams.itertuples()}
    slate = fx[fx["event"].isin(gameweeks) & ~fx["finished"]]

    for row in slate.sort_values("kickoff_time").itertuples():
        h, a = int(row.team_h), int(row.team_a)
        lam_h, lam_a = ratings.expected_goals(h, a)

        for team, opp, is_home, gf, ga, fdr in (
            (h, a, True, lam_h, lam_a, row.team_h_difficulty),
            (a, h, False, lam_a, lam_h, row.team_a_difficulty),
        ):
            attack_mult = float(np.clip(gf / ratings.baseline_goals(team), lo, hi))
            # Out-of-possession volume rises with the opponent's quality.
            opp_pressure = ratings.attack[opp] / ratings.attack[team]
            dc_mult = float(
                np.clip(opp_pressure ** C.DC_FIXTURE_ELASTICITY, lo, hi)
            )
            saves_mult = float(
                np.clip(
                    (ga / ratings.baseline_conceded(team)) ** C.SAVES_FIXTURE_ELASTICITY,
                    lo,
                    hi,
                )
            )
            out[team].append(
                FixtureContext(
                    gw=int(row.event),
                    team=team,
                    opponent=opp,
                    opponent_short=short.get(opp, f"T{opp}"),
                    is_home=is_home,
                    kickoff=str(row.kickoff_time)[:16].replace("T", " "),
                    team_goals=gf,
                    team_conceded=ga,
                    attack_mult=attack_mult,
                    dc_mult=dc_mult,
                    saves_mult=saves_mult,
                    fdr=int(fdr) if not math.isnan(fdr) else 3,
                )
            )
    return out
