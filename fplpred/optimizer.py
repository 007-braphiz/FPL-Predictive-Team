"""Squad, starting-XI and transfer optimisation as integer linear programmes.

Greedily swapping the lowest-projected player for the highest-projected one you
can afford is not optimisation -- it ignores the budget, the three-per-club cap
and the positional quotas all interacting at once. Those constraints are exactly
what an ILP solves properly, so every decision here is posed as one and handed
to CBC.

Three problems, one formulation:

* ``pick_xi``          - given 15 players, choose the best legal XI, captain and
                         bench order for a single gameweek.
* ``optimise_transfers`` - given a squad, a bank balance and a number of free
                         transfers, choose which players to sell and buy, paying
                         the 4-point hit for each transfer beyond the free ones.
* ``build_squad``      - a wildcard: the best legal 15 from scratch under budget.

The objective in the transfer and wildcard problems is the discounted horizon
projection of the starting XI, plus a fraction of each bench player's projection
(bench points only count if someone in the XI does not play), plus the captain's
next-gameweek projection again for the armband.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pulp

from . import config as C


@dataclass
class XISelection:
    xi: list[int]
    bench: list[int]          # ordered: first substitute first
    captain: int
    vice: int
    formation: str
    xp: float                 # projected points including the captain's double


@dataclass
class TransferPlan:
    out_ids: list[int] = field(default_factory=list)
    in_ids: list[int] = field(default_factory=list)
    squad_ids: list[int] = field(default_factory=list)
    n_transfers: int = 0
    hit: int = 0
    objective: float = 0.0
    net_objective: float = 0.0
    bank_after: float = 0.0
    status: str = "unsolved"


def _solver(verbose: bool = False) -> pulp.LpSolver:
    return pulp.PULP_CBC_CMD(msg=1 if verbose else 0)


# ---------------------------------------------------------------------------
# Starting XI
# ---------------------------------------------------------------------------

def pick_xi(squad: pd.DataFrame, points_col: str = "xp_next") -> XISelection:
    """Choose the legal XI, captain and bench order that maximise ``points_col``.

    ``squad`` must be the 15 rows of a projection frame. The captain's points are
    counted twice, which is what makes this a genuine optimisation rather than a
    sort: the best XI and the best captain are chosen jointly.
    """
    ids = squad["id"].astype(int).tolist()
    xp = dict(zip(ids, squad[points_col].astype(float)))
    pos = dict(zip(ids, squad["element_type"].astype(int)))

    prob = pulp.LpProblem("xi", pulp.LpMaximize)
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in ids}
    c = {i: pulp.LpVariable(f"c_{i}", cat="Binary") for i in ids}

    prob += pulp.lpSum(y[i] * xp[i] + c[i] * xp[i] for i in ids)
    prob += pulp.lpSum(y.values()) == C.XI_SIZE
    prob += pulp.lpSum(c.values()) == 1
    for i in ids:
        prob += c[i] <= y[i]
    for p, lo in C.XI_MIN.items():
        members = [y[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(members) >= lo
        prob += pulp.lpSum(members) <= C.XI_MAX[p]

    prob.solve(_solver())

    xi = [i for i in ids if y[i].value() and y[i].value() > 0.5]
    captain = next(i for i in ids if c[i].value() and c[i].value() > 0.5)

    # Bench order: the goalkeeper always occupies the reserve-keeper slot; the
    # outfielders are ordered by projection so the likeliest replacement is
    # first in line for an automatic substitution.
    bench_all = [i for i in ids if i not in xi]
    bench_gk = [i for i in bench_all if pos[i] == C.GK]
    bench_out = sorted(
        (i for i in bench_all if pos[i] != C.GK), key=lambda i: -xp[i]
    )
    bench = bench_gk + bench_out

    xi_sorted = sorted(xi, key=lambda i: -xp[i])
    vice = next((i for i in xi_sorted if i != captain), captain)

    counts = {p: sum(1 for i in xi if pos[i] == p) for p in (C.DEF, C.MID, C.FWD)}
    formation = f"{counts[C.DEF]}-{counts[C.MID]}-{counts[C.FWD]}"

    return XISelection(
        xi=xi_sorted,
        bench=bench,
        captain=captain,
        vice=vice,
        formation=formation,
        xp=sum(xp[i] for i in xi) + xp[captain],
    )


# ---------------------------------------------------------------------------
# Shared squad-building programme
# ---------------------------------------------------------------------------

def _build_problem(
    pool: pd.DataFrame,
    horizon_col: str,
    next_col: str,
    banned: set[int],
    locked: set[int],
):
    """Construct the squad/XI/captain variables and the rules common to all modes."""
    ids = [int(i) for i in pool["id"]]
    xp_h = dict(zip(ids, pool[horizon_col].astype(float)))
    xp_n = dict(zip(ids, pool[next_col].astype(float)))
    pos = dict(zip(ids, pool["element_type"].astype(int)))
    team = dict(zip(ids, pool["team"].astype(int)))

    prob = pulp.LpProblem("squad", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}
    y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in ids}
    c = {i: pulp.LpVariable(f"c_{i}", cat="Binary") for i in ids}

    bench_out_w = sum(C.BENCH_WEIGHTS) / len(C.BENCH_WEIGHTS)
    bench_w = {i: (C.BENCH_GK_WEIGHT if pos[i] == C.GK else bench_out_w) for i in ids}

    objective = pulp.lpSum(
        y[i] * xp_h[i] + (x[i] - y[i]) * xp_h[i] * bench_w[i] + c[i] * xp_n[i]
        for i in ids
    )

    prob += pulp.lpSum(x.values()) == C.SQUAD_SIZE
    for p, n in C.SQUAD_BY_POS.items():
        prob += pulp.lpSum(x[i] for i in ids if pos[i] == p) == n
    for t in set(team.values()):
        prob += pulp.lpSum(x[i] for i in ids if team[i] == t) <= C.MAX_PER_CLUB

    prob += pulp.lpSum(y.values()) == C.XI_SIZE
    for i in ids:
        prob += y[i] <= x[i]
        prob += c[i] <= y[i]
    prob += pulp.lpSum(c.values()) == 1
    for p, lo in C.XI_MIN.items():
        members = [y[i] for i in ids if pos[i] == p]
        prob += pulp.lpSum(members) >= lo
        prob += pulp.lpSum(members) <= C.XI_MAX[p]

    for i in banned:
        if i in x:
            prob += x[i] == 0
    for i in locked:
        if i in x:
            prob += x[i] == 1

    return prob, x, y, c, objective, ids, pos, team, xp_h, xp_n


def build_squad(
    pool: pd.DataFrame,
    budget: float = 100.0,
    horizon_col: str = "xp_horizon",
    next_col: str = "xp_next",
    banned: set[int] | None = None,
    locked: set[int] | None = None,
) -> TransferPlan:
    """Best legal 15 under ``budget`` -- the wildcard / free-hit problem."""
    prob, x, y, c, objective, ids, pos, team, xp_h, xp_n = _build_problem(
        pool, horizon_col, next_col, banned or set(), locked or set()
    )
    price = dict(zip(ids, pool["price"].astype(float)))
    prob += objective
    prob += pulp.lpSum(x[i] * price[i] for i in ids) <= budget
    prob.solve(_solver())

    chosen = [i for i in ids if x[i].value() and x[i].value() > 0.5]
    spend = sum(price[i] for i in chosen)
    return TransferPlan(
        squad_ids=chosen,
        objective=float(pulp.value(prob.objective) or 0.0),
        net_objective=float(pulp.value(prob.objective) or 0.0),
        bank_after=round(budget - spend, 1),
        status=pulp.LpStatus[prob.status],
    )


def optimise_transfers(
    pool: pd.DataFrame,
    current_ids: list[int],
    bank: float,
    free_transfers: int = 1,
    max_transfers: int = 3,
    sell_prices: dict[int, float] | None = None,
    horizon_col: str = "xp_horizon",
    next_col: str = "xp_next",
    banned: set[int] | None = None,
    locked: set[int] | None = None,
) -> list[TransferPlan]:
    """Solve the transfer problem once per allowed transfer count.

    Returning a plan for 0, 1, 2 ... ``max_transfers`` transfers rather than a
    single answer is deliberate: the point-hit trade-off is a judgement call that
    depends on how much you trust the projections, and seeing "two transfers gain
    3.1 points before a 4-point hit" is far more useful than being told to take
    the hit.
    """
    banned = set(banned or set())
    locked = set(locked or set())
    current = set(int(i) for i in current_ids)
    sell_prices = sell_prices or {}

    plans: list[TransferPlan] = []
    for n_transfers in range(0, max_transfers + 1):
        prob, x, y, c, objective, ids, pos, team, xp_h, xp_n = _build_problem(
            pool, horizon_col, next_col, banned, locked
        )
        price = dict(zip(ids, pool["price"].astype(float)))

        # Players outside the pool cannot be evaluated; keeping them would make
        # the squad-size constraint unsatisfiable, so they count as forced sales.
        missing = current - set(ids)
        in_pool = current & set(ids)

        sold = pulp.lpSum(1 - x[i] for i in in_pool) + len(missing)
        prob += sold <= n_transfers

        # Cash: what is spent on incoming players cannot exceed the bank plus
        # what the outgoing players are sold for.
        incoming = pulp.lpSum(x[i] * price[i] for i in ids if i not in current)
        released = pulp.lpSum(
            (1 - x[i]) * sell_prices.get(i, price[i]) for i in in_pool
        ) + sum(sell_prices.get(i, 0.0) for i in missing)
        prob += incoming <= bank + released

        hit = C.TRANSFER_HIT * max(0, n_transfers - free_transfers)
        prob += objective
        prob.solve(_solver())

        if pulp.LpStatus[prob.status] != "Optimal":
            plans.append(TransferPlan(n_transfers=n_transfers, hit=hit,
                                      status=pulp.LpStatus[prob.status]))
            continue

        chosen = [i for i in ids if x[i].value() and x[i].value() > 0.5]
        out_ids = sorted(current - set(chosen))
        in_ids = sorted(set(chosen) - current)
        gross = float(pulp.value(prob.objective) or 0.0)
        spent = sum(price[i] for i in in_ids)
        recovered = sum(sell_prices.get(i, price.get(i, 0.0)) for i in out_ids)

        plans.append(
            TransferPlan(
                out_ids=out_ids,
                in_ids=in_ids,
                squad_ids=chosen,
                n_transfers=len(out_ids),
                hit=C.TRANSFER_HIT * max(0, len(out_ids) - free_transfers),
                objective=gross,
                net_objective=gross - C.TRANSFER_HIT * max(0, len(out_ids) - free_transfers),
                bank_after=round(bank + recovered - spent, 1),
                status="Optimal",
            )
        )
    return plans
