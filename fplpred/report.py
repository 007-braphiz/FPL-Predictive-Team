"""Markdown rendering of projections, line-ups and transfer plans.

Every report opens with a provenance block. A projection is a claim about the
future built on data of a particular age, and the age of that data is part of
the claim -- a squad sheet that does not say when its prices and injury flags
were captured invites the reader to trust it more than they should.
"""

from __future__ import annotations

import pandas as pd

from . import config as C
from .data import DataBundle
from .optimizer import TransferPlan, XISelection
from .strength import TeamRatings
from .xpoints import COMPONENT_ORDER

COMPONENT_LABEL = {
    "appearance": "Mins",
    "goals": "Goals",
    "assists": "Assist",
    "clean_sheet": "CS",
    "conceded": "Conc",
    "saves": "Saves",
    "pen_saves": "PenSv",
    "def_contrib": "DefCon",
    "bonus": "Bonus",
    "cards": "Cards",
}


def _table(rows: list[list[str]], headers: list[str], align: str = "l") -> str:
    """Render a markdown table with columns padded to a consistent width."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "|" + "|".join(
        ("-" * (w + 2)) if align == "l" else ("-" * (w + 1) + ":") for w in widths
    ) + "|"
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |", sep]
    for row in rows:
        out.append(
            "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        )
    return "\n".join(out)


def provenance(bundle: DataBundle, ratings: TeamRatings, gameweeks: list[int]) -> str:
    lines = [
        "## Data provenance",
        "",
        f"- **Source:** `{bundle.source}`",
        f"- **Captured:** {bundle.vintage}",
        f"- **Completed gameweeks in this data:** {bundle.matches_played()}",
        f"- **Projecting:** GW{gameweeks[0]}"
        + (f"-GW{gameweeks[-1]}" if len(gameweeks) > 1 else ""),
        f"- **Weight on this season's team form:** {ratings.weight:.0%} "
        f"(the remainder comes from the pre-season difficulty prior)",
    ]
    if bundle.warnings:
        lines += ["", "> **Read before acting on this:**"]
        lines += [f"> - {w}" for w in bundle.warnings]
    return "\n".join(lines)


def squad_table(proj: pd.DataFrame, ids: list[int], title: str) -> str:
    sub = proj[proj["id"].isin(ids)].copy()
    order = {pid: i for i, pid in enumerate(ids)}
    sub = sub.sort_values("id", key=lambda s: s.map(order))
    rows = []
    for r in sub.itertuples():
        flag = "" if r.availability >= 0.99 else f" [{r.availability:.0%}]"
        rows.append([
            r.pos, f"{r.name}{flag}", r.team_name, f"{r.price:.1f}",
            r.opponents or "-", f"{r.p_start:.0%}",
            f"{r.xp_next:.2f}", f"{r.xp_horizon:.2f}",
        ])
    headers = ["Pos", "Player", "Club", "£", "Fixture(s)", "Start%", "xP next", "xP horizon"]
    return f"### {title}\n\n" + _table(rows, headers)


def component_breakdown(proj: pd.DataFrame, ids: list[int], title: str) -> str:
    """Show where each player's projected points come from.

    The breakdown is the point of the exercise: two players on 4.5 expected
    points are not equivalent if one gets them from goal threat and the other
    from a clean sheet against a side that has scored in every match.
    """
    sub = proj[proj["id"].isin(ids)].copy().sort_values("xp_next", ascending=False)
    cols = [c for c in COMPONENT_ORDER if abs(sub[f"c_{c}"]).max() > 0.005]
    rows = []
    for r in sub.itertuples():
        row = [r.name, r.pos]
        for c in cols:
            row.append(f"{getattr(r, f'c_{c}'):+.2f}")
        row.append(f"{r.xp_next:.2f}")
        rows.append(row)
    headers = ["Player", "Pos"] + [COMPONENT_LABEL[c] for c in cols] + ["Total"]
    return f"### {title}\n\n" + _table(rows, headers)


def lineup(proj: pd.DataFrame, sel: XISelection, gw: int) -> str:
    by_id = {int(r.id): r for r in proj.itertuples()}
    lines = [
        f"### Recommended GW{gw} line-up - {sel.formation}",
        "",
        f"**Projected total: {sel.xp:.1f} points** (starting XI plus the captain's double)",
        "",
    ]
    rows = []
    for pid in sel.xi:
        r = by_id[pid]
        badge = " (C)" if pid == sel.captain else " (V)" if pid == sel.vice else ""
        rows.append([r.pos, f"{r.name}{badge}", r.team_name, r.next_opponent,
                     f"{r.xp_next:.2f}"])
    lines.append(_table(rows, ["Pos", "Player", "Club", "Fixture", "xP"]))

    lines += ["", "**Bench** (in order)", ""]
    brows = []
    for i, pid in enumerate(sel.bench, start=1):
        r = by_id[pid]
        brows.append([str(i), r.pos, r.name, r.team_name, r.next_opponent,
                      f"{r.xp_next:.2f}"])
    lines.append(_table(brows, ["#", "Pos", "Player", "Club", "Fixture", "xP"]))
    return "\n".join(lines)


def captain_ranking(proj: pd.DataFrame, ids: list[int], top: int = 6) -> str:
    """Rank captaincy options.

    Expected points alone is not the whole captaincy question. A captain also
    needs to be certain to play -- the armband on a doubtful starter risks two
    points rather than merely under-delivering -- so start probability is shown
    beside the projection.
    """
    sub = proj[proj["id"].isin(ids)].nlargest(top, "xp_next")
    rows = []
    for r in sub.itertuples():
        rows.append([
            r.name, r.team_name, r.next_opponent, f"{r.p_start:.0%}",
            f"{r.xp_next:.2f}", f"{2 * r.xp_next:.2f}",
        ])
    return "### Captaincy ranking\n\n" + _table(
        rows, ["Player", "Club", "Fixture", "Start%", "xP", "xP as captain"]
    )


def transfer_plans(
    proj: pd.DataFrame, plans: list[TransferPlan], baseline: float
) -> str:
    """Compare each transfer count against doing nothing.

    ``baseline`` is the objective value of the zero-transfer plan. Gains are
    reported net of the point hit, and a plan is only worth taking if that net
    figure is clearly positive -- a projected gain of half a point is inside the
    model's own error bars.
    """
    by_id = {int(r.id): r for r in proj.itertuples()}
    lines = ["### Transfer options", ""]
    rows = []
    for plan in plans:
        if plan.status != "Optimal":
            rows.append([str(plan.n_transfers), "-", "-", "-", "-", plan.status])
            continue
        outs = ", ".join(by_id[i].name for i in plan.out_ids) or "-"
        ins = ", ".join(by_id[i].name for i in plan.in_ids) or "-"
        gain = plan.net_objective - baseline
        rows.append([
            str(plan.n_transfers), outs, ins, f"-{plan.hit}" if plan.hit else "0",
            f"{gain:+.2f}", f"£{plan.bank_after:.1f}m",
        ])
    lines.append(_table(
        rows, ["#", "Out", "In", "Hit", "Net gain vs no move", "Bank after"]
    ))
    lines += [
        "",
        "_Net gain is measured over the whole projection horizon and already "
        "includes the point hit. Treat anything under about +1.0 as noise._",
    ]
    return "\n".join(lines)


def team_ratings_table(ratings: TeamRatings, top: int = 20) -> str:
    df = ratings.detail.head(top)
    rows = []
    for r in df.itertuples():
        rows.append([
            r.short_name, f"{r.attack:.2f}", f"{r.defence:.2f}",
            f"{r.mean_fdr:.2f}",
            "-" if pd.isna(r.xg_for) else f"{r.xg_for:.2f}",
            "-" if pd.isna(r.xg_against) else f"{r.xg_against:.2f}",
        ])
    return "### Team ratings\n\n" + _table(
        rows,
        ["Club", "Attack", "Defence", "Mean FDR faced", "xG/match", "xGC/match"],
    ) + (
        "\n\n_Attack and defence are multiplicative and centred on 1.00. "
        "Defence above 1.00 means the side concedes less than average._"
    )


def best_available(
    proj: pd.DataFrame, exclude: list[int], max_price: float | None = None, top: int = 12
) -> str:
    pool = proj[~proj["id"].isin(exclude)]
    if max_price is not None:
        pool = pool[pool["price"] <= max_price]
    rows = []
    for r in pool.nlargest(top, "xp_horizon").itertuples():
        rows.append([
            r.pos, r.name, r.team_name, f"{r.price:.1f}", r.opponents or "-",
            f"{r.selected_by:.1f}%", f"{r.xp_next:.2f}", f"{r.xp_horizon:.2f}",
        ])
    return "### Highest-projected players you do not own\n\n" + _table(
        rows,
        ["Pos", "Player", "Club", "£", "Fixture(s)", "Owned", "xP next", "xP horizon"],
    )


# ---------------------------------------------------------------------------
# Phone-friendly output
# ---------------------------------------------------------------------------

def mobile_summary(
    proj: pd.DataFrame,
    sel: XISelection,
    plans: list[TransferPlan],
    bundle: DataBundle,
    gw: int,
    baseline: float,
) -> str:
    """A narrow summary that reads on a phone without horizontal scrolling.

    The wide markdown tables elsewhere in this module are unusable on a 6-inch
    screen -- they wrap into nonsense. This renders the same decisions as short
    lines under 40 characters, which is the format most likely to be read on the
    walk to the deadline.
    """
    by_id = {int(r.id): r for r in proj.itertuples()}
    out = [
        f"GW{gw} | {sel.formation} | proj {sel.xp:.1f} pts",
        f"data: {bundle.source}, {bundle.matches_played()} GWs complete",
        "",
        "STARTING XI",
    ]
    for pid in sel.xi:
        r = by_id[pid]
        badge = "(C)" if pid == sel.captain else "(V)" if pid == sel.vice else "   "
        venue = r.next_opponent.replace("(", " ").replace(")", "")
        out.append(f" {badge} {r.name[:13]:<13} {venue:<11} {r.xp_next:4.1f}")

    out += ["", "BENCH"]
    for i, pid in enumerate(sel.bench, start=1):
        r = by_id[pid]
        out.append(f" {i}. {r.name[:13]:<13} {r.pos:<3} {r.xp_next:4.1f}")

    out += ["", "CAPTAIN OPTIONS"]
    for r in proj[proj["id"].isin(sel.xi)].nlargest(3, "xp_next").itertuples():
        out.append(f" {r.name[:14]:<14} {r.p_start:3.0%} start  {2 * r.xp_next:4.1f}")

    out += ["", "TRANSFERS"]
    for plan in plans:
        if plan.status != "Optimal" or plan.n_transfers == 0:
            continue
        gain = plan.net_objective - baseline
        outs = "/".join(by_id[i].name[:10] for i in plan.out_ids)
        ins = "/".join(by_id[i].name[:10] for i in plan.in_ids)
        hit = f" (-{plan.hit})" if plan.hit else ""
        out.append(f" {plan.n_transfers}x{hit}: {outs}")
        out.append(f"     -> {ins}  {gain:+.1f}")
    if len(out) and out[-1] == "TRANSFERS":
        out.append(" none improve on holding")

    flagged = proj[(proj["id"].isin(sel.xi + sel.bench)) & (proj["availability"] < 1.0)]
    if len(flagged):
        out += ["", "FLAGGED"]
        for r in flagged.itertuples():
            note = (r.news or "no detail")[:34]
            out.append(f" {r.name[:13]:<13} {r.availability:3.0%}  {note}")

    if bundle.warnings:
        out += ["", "WARNINGS"]
        for w in bundle.warnings:
            out.append(f" - {w[:120]}")
    return "\n".join(out)
