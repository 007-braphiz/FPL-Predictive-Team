"""Tests for the scoring maths and the optimiser's legality guarantees.

These check the properties that would be expensive to notice by eye: that the
block-scoring rules round down rather than pro-rate, that the threshold
probability behaves like a real distribution, that a substitute cannot be
credited with a clean sheet, and that every squad the optimiser returns is one
FPL would actually accept.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fplpred import config as C  # noqa: E402
from fplpred import data as D  # noqa: E402
from fplpred.cli import project  # noqa: E402
from fplpred.optimizer import build_squad, optimise_transfers, pick_xi  # noqa: E402
from fplpred.strength import FixtureContext  # noqa: E402
from fplpred.xpoints import (  # noqa: E402
    expected_floor_div,
    negbin_sf,
    points_given_minutes,
    poisson_pmf,
)


# ---------------------------------------------------------------------------
# Distribution helpers
# ---------------------------------------------------------------------------

def test_poisson_pmf_sums_to_one():
    for lam in (0.3, 1.4, 3.0):
        total = sum(poisson_pmf(k, lam) for k in range(0, 40))
        assert total == pytest.approx(1.0, abs=1e-9)


def test_expected_floor_div_rounds_down():
    """E[floor(K/2)] must be strictly below lam/2 -- flooring loses value."""
    lam = 1.4
    brute = sum((k // 2) * poisson_pmf(k, lam) for k in range(0, 60))
    assert expected_floor_div(lam, 2) == pytest.approx(brute, abs=1e-9)
    assert expected_floor_div(lam, 2) < lam / 2


def test_expected_floor_div_zero_rate():
    assert expected_floor_div(0.0, 3) == 0.0


def test_negbin_sf_is_a_probability_and_monotonic():
    mean = 9.5
    probs = [negbin_sf(t, mean, C.DC_OVERDISPERSION) for t in range(1, 20)]
    assert all(0.0 <= p <= 1.0 for p in probs)
    # Clearing a higher threshold can never be more likely.
    assert all(a >= b - 1e-12 for a, b in zip(probs, probs[1:]))


def test_negbin_sf_rises_with_the_mean():
    low = negbin_sf(10, 7.0, C.DC_OVERDISPERSION)
    high = negbin_sf(10, 12.0, C.DC_OVERDISPERSION)
    assert high > low


def test_overdispersion_moves_probability_toward_the_threshold():
    """Extra variance helps players below the bar and hurts those above it.

    A defender averaging 7 defensive actions only ever reaches 10 on a busy
    afternoon, so widening the distribution raises his chance. One averaging 13
    clears it most weeks, and the same widening can only cost him. Modelling the
    counts as Poisson would understate the first player and overstate the
    second -- which is precisely the population where the +2 is decided.
    """
    assert negbin_sf(10, 7.0, 1.8) > negbin_sf(10, 7.0, 1.05)
    assert negbin_sf(10, 13.0, 1.8) < negbin_sf(10, 13.0, 1.05)


# ---------------------------------------------------------------------------
# Points components
# ---------------------------------------------------------------------------

def _ctx(**kw) -> FixtureContext:
    defaults = dict(
        gw=4, team=1, opponent=2, opponent_short="OPP", is_home=True,
        kickoff="2026-09-12 14:00", team_goals=1.5, team_conceded=1.1,
        attack_mult=1.0, dc_mult=1.0, saves_mult=1.0, fdr=3,
    )
    defaults.update(kw)
    return FixtureContext(**defaults)


def _player(**kw) -> pd.Series:
    base = dict(
        element_type=C.DEF, xg90=0.10, xa90=0.08, dc90=9.0, saves90=0.0,
        bonus90=0.20, yellow90=0.15, is_pen_taker=False,
    )
    base.update(kw)
    return pd.Series(base)


def test_appearance_points_step_at_60_minutes():
    p, ctx = _player(), _ctx()
    assert points_given_minutes(p, ctx, 45)["appearance"] == 1
    assert points_given_minutes(p, ctx, 60)["appearance"] == 2
    assert points_given_minutes(p, ctx, 82)["appearance"] == 2


def test_substitute_cannot_earn_a_clean_sheet():
    p, ctx = _player(), _ctx()
    assert points_given_minutes(p, ctx, 22)["clean_sheet"] == 0.0
    assert points_given_minutes(p, ctx, 82)["clean_sheet"] > 0.0


def test_clean_sheet_falls_as_the_opponent_gets_better():
    p = _player()
    easy = points_given_minutes(p, _ctx(team_conceded=0.6), 82)["clean_sheet"]
    hard = points_given_minutes(p, _ctx(team_conceded=2.2), 82)["clean_sheet"]
    assert easy > hard


def test_midfielders_are_not_charged_for_goals_conceded():
    ctx = _ctx(team_conceded=2.5)
    assert points_given_minutes(_player(element_type=C.MID), ctx, 82)["conceded"] == 0.0
    assert points_given_minutes(_player(element_type=C.DEF), ctx, 82)["conceded"] < 0.0


def test_only_goalkeepers_earn_save_points():
    ctx = _ctx()
    gk = _player(element_type=C.GK, saves90=3.4, dc90=0.0)
    assert points_given_minutes(gk, ctx, 82)["saves"] > 0.0
    assert points_given_minutes(_player(saves90=3.4), ctx, 82)["saves"] == 0.0


def test_goal_points_follow_the_position_table():
    ctx = _ctx()
    kw = dict(xg90=0.5, dc90=0.0, xa90=0.0, bonus90=0.0, yellow90=0.0)
    defender = points_given_minutes(_player(element_type=C.DEF, **kw), ctx, 90)["goals"]
    forward = points_given_minutes(_player(element_type=C.FWD, **kw), ctx, 90)["goals"]
    assert defender / forward == pytest.approx(
        C.SCORING["goal"][C.DEF] / C.SCORING["goal"][C.FWD]
    )


def test_penalty_taker_is_credited_more_than_an_identical_non_taker():
    ctx = _ctx()
    taker = points_given_minutes(_player(element_type=C.FWD, is_pen_taker=True), ctx, 82)
    other = points_given_minutes(_player(element_type=C.FWD, is_pen_taker=False), ctx, 82)
    assert taker["goals"] > other["goals"]


def test_defensive_contribution_scales_with_minutes():
    p, ctx = _player(dc90=11.0), _ctx()
    assert (
        points_given_minutes(p, ctx, 22)["def_contrib"]
        < points_given_minutes(p, ctx, 82)["def_contrib"]
    )


def test_attack_multiplier_lifts_goal_expectation():
    p = _player(element_type=C.FWD, xg90=0.6)
    weak = points_given_minutes(p, _ctx(attack_mult=0.7), 82)["goals"]
    strong = points_given_minutes(p, _ctx(attack_mult=1.4), 82)["goals"]
    assert strong / weak == pytest.approx(1.4 / 0.7, rel=1e-6)


# ---------------------------------------------------------------------------
# Optimiser legality
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def projection():
    bundle = D.load_snapshot()
    proj, ratings, gameweeks = project(bundle, horizon=3, start_gw=4)
    return bundle, proj, gameweeks


def test_pick_xi_returns_a_legal_formation(projection):
    bundle, proj, _ = projection
    squad_ids = [
        int(r["id"]) for _, r in D.resolve_squad(bundle, D.load_my_team()).iterrows()
    ]
    sel = pick_xi(proj[proj["id"].isin(squad_ids)])

    assert len(sel.xi) == C.XI_SIZE
    assert len(sel.bench) == C.SQUAD_SIZE - C.XI_SIZE
    assert set(sel.xi) & set(sel.bench) == set()
    assert sel.captain in sel.xi and sel.vice in sel.xi
    assert sel.captain != sel.vice

    pos = dict(zip(proj["id"].astype(int), proj["element_type"].astype(int)))
    for p, lo in C.XI_MIN.items():
        n = sum(1 for i in sel.xi if pos[i] == p)
        assert lo <= n <= C.XI_MAX[p]

    # The reserve goalkeeper always occupies bench slot 1.
    assert pos[sel.bench[0]] == C.GK


def test_pick_xi_captains_the_highest_projected_starter(projection):
    _, proj, _ = projection
    squad = proj.head(15)
    sel = pick_xi(squad)
    best = squad.loc[squad["id"].isin(sel.xi), "xp_next"].max()
    captain_xp = float(squad.loc[squad["id"] == sel.captain, "xp_next"].iloc[0])
    assert captain_xp == pytest.approx(best)


def test_build_squad_respects_every_fpl_rule(projection):
    _, proj, _ = projection
    plan = build_squad(proj, budget=100.0)
    chosen = proj[proj["id"].isin(plan.squad_ids)]

    assert plan.status == "Optimal"
    assert len(chosen) == C.SQUAD_SIZE
    assert chosen["price"].sum() <= 100.0 + 1e-6
    for p, n in C.SQUAD_BY_POS.items():
        assert (chosen["element_type"] == p).sum() == n
    assert chosen["team"].value_counts().max() <= C.MAX_PER_CLUB


def test_transfer_plans_never_exceed_the_requested_count(projection):
    bundle, proj, _ = projection
    squad_ids = [
        int(r["id"]) for _, r in D.resolve_squad(bundle, D.load_my_team()).iterrows()
    ]
    plans = optimise_transfers(
        proj, current_ids=squad_ids, bank=0.0, free_transfers=1, max_transfers=2
    )
    assert [p.n_transfers for p in plans] == [0, 1, 2]
    for plan in plans:
        assert plan.status == "Optimal"
        assert len(plan.squad_ids) == C.SQUAD_SIZE
        assert len(plan.in_ids) == len(plan.out_ids) == plan.n_transfers
        assert plan.bank_after >= -1e-6


def test_more_transfers_never_lower_the_gross_objective(projection):
    """Relaxing a constraint cannot make the optimum worse."""
    bundle, proj, _ = projection
    squad_ids = [
        int(r["id"]) for _, r in D.resolve_squad(bundle, D.load_my_team()).iterrows()
    ]
    plans = optimise_transfers(
        proj, current_ids=squad_ids, bank=0.0, free_transfers=1, max_transfers=2
    )
    objectives = [p.objective for p in plans]
    assert all(a <= b + 1e-6 for a, b in zip(objectives, objectives[1:]))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def test_projection_covers_every_player_and_is_finite(projection):
    bundle, proj, gameweeks = projection
    assert len(proj) == len(bundle.players)
    assert proj["xp_next"].notna().all()
    assert math.isfinite(float(proj["xp_horizon"].max()))
    assert (proj["xp_next"] >= -1.0).all()


def test_components_sum_to_the_projected_total(projection):
    _, proj, _ = projection
    comp_cols = [c for c in proj.columns if c.startswith("c_")]
    assert (proj[comp_cols].sum(axis=1) - proj["xp_next"]).abs().max() < 1e-6


def test_unavailable_players_project_zero(projection):
    _, proj, _ = projection
    out = proj[proj["availability"] == 0.0]
    if len(out):
        assert out["xp_next"].abs().max() < 1e-9


def test_squad_resolution_rejects_an_ambiguous_name():
    """A partial surname that matches two players must fail loudly.

    "Sangar" matches both M.Sangaré and I.Sangaré. Silently picking one would
    put the wrong player in the projection and the error would surface only
    after the deadline.
    """
    bundle = D.load_snapshot()
    with pytest.raises(ValueError, match="ambiguous"):
        D.resolve_squad(bundle, {"squad": [{"name": "Sangar"}]})


def test_squad_resolution_prefers_an_exact_name_match():
    """An exact web_name is unambiguous even when it is a substring of others."""
    bundle = D.load_snapshot()
    resolved = D.resolve_squad(bundle, {"squad": [{"name": "Fernandes"}]})
    assert len(resolved) == 1
    assert resolved.iloc[0]["web_name"] == "Fernandes"
