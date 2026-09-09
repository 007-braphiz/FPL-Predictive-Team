"""The expected-points engine.

For one player in one fixture, expected points are built up component by
component and summed. The structure that makes this accurate is *conditioning on
minutes*: rather than plugging an average minutes figure into every term, the
model evaluates two scenarios -- the player starts, or the player comes off the
bench -- and weights them by their probabilities::

    xP = P(start) * points_given(82 mins) + P(cameo) * points_given(22 mins)

That matters because most of FPL scoring is non-linear in minutes. A clean sheet
needs 60 minutes. Defensive contribution needs a threshold count of actions. Save
points come in blocks of three. Averaging the minutes first and applying the
thresholds afterwards systematically over-rates fringe players -- it credits a
20-minute substitute with a fraction of a clean sheet he can never be awarded.

Components modelled
-------------------
appearance, goals, assists, penalty top-up, clean sheet, goals conceded,
saves, penalty saves, defensive contribution, bonus, cards.

Known simplifications, stated plainly so they are not mistaken for precision:

* Bonus is a shrunk per-90 rate, not a simulation of the BPS table. It is right
  on average and wrong on any individual match.
* Goals and assists use the team's expected goals for the fixture as a scaling
  factor; they do not model the correlation between a team's goals and which of
  its players scores them.
* Own goals and penalty misses are omitted (worth roughly -0.02 points a match).
* Rotation risk beyond observed start rate and the official injury flag is not
  modelled -- no engine can read a manager's mind, and press-conference news is
  the manager's job to apply on top of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .strength import FixtureContext

_MAX_COUNT = 25  # truncation point for count-distribution sums


# ---------------------------------------------------------------------------
# Small probability helpers
# ---------------------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def expected_floor_div(lam: float, divisor: int) -> float:
    """E[floor(K / divisor)] for K ~ Poisson(lam).

    Used for the two FPL rules that pay out in blocks: one point per three
    saves, and minus one per two goals conceded. Taking ``lam / divisor``
    instead would over-pay, because floor() rounds down.
    """
    if lam <= 0:
        return 0.0
    total = 0.0
    for k in range(1, _MAX_COUNT + 1):
        total += (k // divisor) * poisson_pmf(k, lam)
    return total


def negbin_sf(threshold: int, mean: float, overdispersion: float) -> float:
    """P(X >= threshold) for X negative-binomial with the given mean.

    Variance is ``mean * overdispersion``. Defensive-action counts are more
    variable than a Poisson process allows -- a midfielder averaging 11 actions
    does not hit exactly 11 every week -- and understating that variance
    understates the chance of clearing the threshold.
    """
    if threshold <= 0:
        return 1.0
    if mean <= 0:
        return 0.0
    phi = max(overdispersion, 1.0000001)
    r = mean / (phi - 1.0)
    p = 1.0 / phi

    cdf = 0.0
    log_p_r = r * math.log(p)
    for k in range(0, threshold):
        log_pmf = (
            math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + log_p_r + k * math.log1p(-p)
        )
        cdf += math.exp(log_pmf)
        if cdf >= 1.0:
            return 0.0
    return max(0.0, 1.0 - cdf)


# ---------------------------------------------------------------------------
# Per-fixture projection
# ---------------------------------------------------------------------------

@dataclass
class Projection:
    player_id: int
    gw: int
    xp: float
    components: dict[str, float] = field(default_factory=dict)


def points_given_minutes(
    player: pd.Series, ctx: FixtureContext, minutes: float
) -> dict[str, float]:
    """Expected points broken down by component, conditional on playing ``minutes``."""
    pos = int(player["element_type"])
    share = minutes / 90.0
    comp: dict[str, float] = {}

    # --- appearance -------------------------------------------------------
    comp["appearance"] = float(C.SCORING["appearance_any"])
    if minutes >= 60:
        comp["appearance"] += float(C.SCORING["appearance_60"])

    # --- goals and assists ------------------------------------------------
    lam_goal = float(player["xg90"]) * share * ctx.attack_mult
    lam_assist = float(player["xa90"]) * share * ctx.attack_mult

    if bool(player.get("is_pen_taker", False)):
        # Expected penalties for this side in this fixture, scaled by how
        # attacking the fixture looks, then only partially credited so the
        # spot-kicks already inside the player's xG are not counted twice.
        pens = C.PENS_PER_TEAM_MATCH * ctx.attack_mult * C.PEN_TAKER_SHARE * share
        lam_goal += pens * C.PEN_CONVERSION * C.PEN_TOPUP_SHARE

    comp["goals"] = lam_goal * float(C.SCORING["goal"][pos])
    comp["assists"] = lam_assist * float(C.SCORING["assist"])

    # --- clean sheet ------------------------------------------------------
    # FPL awards the clean sheet if the player reaches 60 minutes and no goal
    # is conceded while he is on the pitch, hence scaling the rate by `share`.
    cs_points = float(C.SCORING["clean_sheet"][pos])
    if minutes >= 60 and cs_points:
        comp["clean_sheet"] = math.exp(-ctx.team_conceded * share) * cs_points
    else:
        comp["clean_sheet"] = 0.0

    # --- goals conceded ---------------------------------------------------
    if pos in (C.GK, C.DEF):
        conceded = expected_floor_div(
            ctx.team_conceded * share, int(C.SCORING["conceded_per_penalty"])
        )
        comp["conceded"] = conceded * float(C.SCORING["conceded_penalty"])
    else:
        comp["conceded"] = 0.0

    # --- saves ------------------------------------------------------------
    if pos == C.GK:
        lam_saves = float(player["saves90"]) * share * ctx.saves_mult
        comp["saves"] = expected_floor_div(
            lam_saves, int(C.SCORING["saves_per_point"])
        )
        pens_faced = C.PENS_PER_TEAM_MATCH * (1.0 / max(ctx.attack_mult, 0.2)) * share
        comp["pen_saves"] = (
            pens_faced * C.PEN_SAVE_RATE * float(C.SCORING["penalty_save"])
        )
    else:
        comp["saves"] = 0.0
        comp["pen_saves"] = 0.0

    # --- defensive contribution ------------------------------------------
    threshold = int(C.SCORING["dc_threshold"][pos])
    if threshold < 900:
        lam_dc = float(player["dc90"]) * share * ctx.dc_mult
        p_hit = negbin_sf(threshold, lam_dc, C.DC_OVERDISPERSION)
        comp["def_contrib"] = p_hit * float(C.SCORING["defensive_contribution"])
    else:
        comp["def_contrib"] = 0.0

    # --- bonus and cards --------------------------------------------------
    comp["bonus"] = float(player["bonus90"]) * share
    comp["cards"] = float(player["yellow90"]) * share * C.CARD_POINTS_PER_YELLOW

    return comp


def project_player(
    player: pd.Series, contexts: list[FixtureContext]
) -> list[Projection]:
    """Project one player across every fixture his club plays in the horizon."""
    p_start = float(player["p_start"])
    p_cameo = float(player["p_cameo"])
    out: list[Projection] = []

    for ctx in contexts:
        if p_start + p_cameo <= 1e-6:
            out.append(Projection(int(player["id"]), ctx.gw, 0.0, {}))
            continue

        start_comp = points_given_minutes(player, ctx, C.MINUTES_IF_START)
        cameo_comp = points_given_minutes(player, ctx, C.MINUTES_IF_CAMEO)
        merged = {
            key: p_start * start_comp[key] + p_cameo * cameo_comp[key]
            for key in start_comp
        }
        out.append(
            Projection(int(player["id"]), ctx.gw, float(sum(merged.values())), merged)
        )
    return out


COMPONENT_ORDER = [
    "appearance", "goals", "assists", "clean_sheet", "conceded",
    "saves", "pen_saves", "def_contrib", "bonus", "cards",
]


def project_all(
    rates: pd.DataFrame,
    contexts_by_team: dict[int, list[FixtureContext]],
    gameweeks: list[int],
) -> pd.DataFrame:
    """Project every player over ``gameweeks``.

    Returns one row per player with ``xp_gw<N>`` columns, a discounted
    ``xp_horizon`` total, and the component breakdown for the first gameweek so
    that any projection can be interrogated rather than merely trusted.
    """
    rows = []
    for player in rates.itertuples(index=False):
        series = pd.Series(player._asdict())
        team_ctxs = contexts_by_team.get(int(series["team"]), [])
        projections = project_player(series, team_ctxs)

        by_gw: dict[int, float] = {gw: 0.0 for gw in gameweeks}
        first_components: dict[str, float] = {}
        for proj in projections:
            if proj.gw in by_gw:
                by_gw[proj.gw] += proj.xp
                if proj.gw == gameweeks[0]:
                    for key, val in proj.components.items():
                        first_components[key] = first_components.get(key, 0.0) + val

        horizon = sum(
            by_gw[gw] * (C.HORIZON_DISCOUNT ** i) for i, gw in enumerate(gameweeks)
        )

        row = {
            "id": int(series["id"]),
            "name": series["web_name"],
            "pos": series["pos_name"],
            "element_type": int(series["element_type"]),
            "team": int(series["team"]),
            "team_name": series["team_name"],
            "price": float(series["price"]),
            "status": series["status"],
            "news": series["news"],
            "availability": float(series["availability"]),
            "p_start": float(series["p_start"]),
            "exp_minutes": float(series["exp_minutes"]),
            "selected_by": float(series["selected_by_percent"]),
            "fixtures": len(team_ctxs),
            "opponents": " + ".join(
                f"{c.opponent_short}({'H' if c.is_home else 'A'},{c.fdr})"
                for c in team_ctxs
            ),
            "next_opponent": " + ".join(
                f"{c.opponent_short}({'H' if c.is_home else 'A'},{c.fdr})"
                for c in team_ctxs
                if c.gw == gameweeks[0]
            ) or "BLANK",
            "xp_next": by_gw[gameweeks[0]],
            "xp_horizon": horizon,
            "xp_total_raw": sum(by_gw.values()),
        }
        for gw in gameweeks:
            row[f"xp_gw{gw}"] = by_gw[gw]
        for key in COMPONENT_ORDER:
            row[f"c_{key}"] = first_components.get(key, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["value"] = df["xp_horizon"] / df["price"].clip(lower=0.1)
    return df.sort_values("xp_horizon", ascending=False).reset_index(drop=True)
