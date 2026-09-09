"""Command-line entry point.

    python -m fplpred.cli analyse --horizon 3
    python -m fplpred.cli analyse --source live --free-transfers 2 --bank 0.3
    python -m fplpred.cli wildcard --budget 100.0
    python -m fplpred.cli import-team --entry 1234567 --gw 3

``analyse`` is the one to run before a deadline. It projects every player,
picks the best legal XI and captain from the current squad, and prices up the
transfer options against doing nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import config as C
from . import data as D
from . import report as R
from .optimizer import build_squad, optimise_transfers, pick_xi
from .rates import build_rates
from .strength import compute_ratings, fixture_contexts
from .xpoints import project_all

REPORTS_DIR = D.REPO_ROOT / "reports"


def _team_matches(bundle: D.DataBundle) -> dict[int, int]:
    played: dict[int, int] = {int(t): 0 for t in bundle.teams["id"]}
    for row in bundle.fixtures[bundle.fixtures["finished"]].itertuples():
        played[int(row.team_h)] = played.get(int(row.team_h), 0) + 1
        played[int(row.team_a)] = played.get(int(row.team_a), 0) + 1
    return played


def project(bundle: D.DataBundle, horizon: int, start_gw: int | None = None):
    """Run the full pipeline and return (projection frame, ratings, gameweeks)."""
    ratings = compute_ratings(bundle)
    first = start_gw or bundle.next_gw
    gameweeks = [first + i for i in range(horizon)]
    contexts = fixture_contexts(bundle, ratings, gameweeks)
    rates = build_rates(bundle, _team_matches(bundle))
    proj = project_all(rates, contexts, gameweeks)
    return proj, ratings, gameweeks


def cmd_analyse(args: argparse.Namespace) -> int:
    bundle = D.load(args.source)
    proj, ratings, gameweeks = project(bundle, args.horizon, args.gw)
    gw = gameweeks[0]

    squad_cfg = D.load_my_team(args.team)
    squad_rows = D.resolve_squad(bundle, squad_cfg)
    squad_ids = [int(i) for i in squad_rows["id"]]
    squad_proj = proj[proj["id"].isin(squad_ids)]

    if len(squad_proj) != C.SQUAD_SIZE:
        print(
            f"warning: resolved {len(squad_proj)} of {C.SQUAD_SIZE} squad players",
            file=sys.stderr,
        )

    bank = args.bank if args.bank is not None else float(squad_cfg.get("bank", 0.0))
    free = (
        args.free_transfers
        if args.free_transfers is not None
        else int(squad_cfg.get("free_transfers", 1))
    )
    sell_prices = {
        int(e["id"]): float(e["selling_price"])
        for e in squad_cfg.get("squad", [])
        if isinstance(e, dict) and e.get("id") and e.get("selling_price") is not None
    }

    selection = pick_xi(squad_proj)

    # Players who cannot be bought are excluded from the transfer market rather
    # than being ranked at zero -- a suspended player is not a cheap enabler.
    banned = set(
        int(i) for i in proj.loc[proj["availability"] <= 0.0, "id"]
    ) - set(squad_ids)
    plans = optimise_transfers(
        proj,
        current_ids=squad_ids,
        bank=bank,
        free_transfers=free,
        max_transfers=args.max_transfers,
        sell_prices=sell_prices,
        banned=banned,
    )
    baseline = plans[0].net_objective if plans else 0.0

    parts = [
        f"# FPL projection - Gameweek {gw}",
        "",
        R.provenance(bundle, ratings, gameweeks),
        "",
        R.lineup(proj, selection, gw),
        "",
        R.captain_ranking(proj, squad_ids),
        "",
        R.squad_table(proj, squad_ids, "Full squad"),
        "",
        R.component_breakdown(proj, squad_ids, "Where the projected points come from"),
        "",
        R.transfer_plans(proj, plans, baseline),
        "",
        R.best_available(proj, exclude=squad_ids, top=args.top),
        "",
        R.team_ratings_table(ratings),
        "",
    ]
    text = "\n".join(parts)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_md = REPORTS_DIR / f"gw{gw}_report.md"
    out_md.write_text(text)
    proj.to_csv(REPORTS_DIR / f"gw{gw}_projections.csv", index=False)

    print(text)
    print(f"\n[written] {out_md}")
    print(f"[written] {REPORTS_DIR / f'gw{gw}_projections.csv'}")
    return 0


def cmd_wildcard(args: argparse.Namespace) -> int:
    bundle = D.load(args.source)
    proj, ratings, gameweeks = project(bundle, args.horizon, args.gw)
    banned = set(int(i) for i in proj.loc[proj["availability"] <= 0.0, "id"])
    plan = build_squad(proj, budget=args.budget, banned=banned)

    squad_proj = proj[proj["id"].isin(plan.squad_ids)]
    selection = pick_xi(squad_proj)
    text = "\n".join([
        f"# Wildcard squad - Gameweek {gameweeks[0]}",
        "",
        R.provenance(bundle, ratings, gameweeks),
        "",
        f"Budget £{args.budget:.1f}m, £{plan.bank_after:.1f}m unspent.",
        "",
        R.lineup(proj, selection, gameweeks[0]),
        "",
        R.squad_table(proj, plan.squad_ids, "Full 15"),
        "",
    ])
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"gw{gameweeks[0]}_wildcard.md"
    out.write_text(text)
    print(text)
    print(f"\n[written] {out}")
    return 0


def cmd_import_team(args: argparse.Namespace) -> int:
    """Rebuild data/my_team.json from a live FPL entry."""
    picks = D.fetch_entry_picks(args.entry, args.gw)
    bundle = D.load("live")
    names = {int(r.id): r.web_name for r in bundle.players.itertuples()}
    squad = [
        {
            "id": int(p["element"]),
            "name": names.get(int(p["element"]), ""),
            "selling_price": p.get("selling_price", 0) / 10.0,
            "purchase_price": p.get("purchase_price", 0) / 10.0,
        }
        for p in picks["picks"]
    ]
    payload = {
        "entry_id": args.entry,
        "gameweek": args.gw,
        "bank": picks.get("entry_history", {}).get("bank", 0) / 10.0,
        "free_transfers": args.free_transfers or 1,
        "squad": squad,
    }
    out = Path(args.out or D.REPO_ROOT / "data" / "my_team.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"[written] {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fplpred", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--source", default="auto", choices=["auto", "live", "snapshot"],
        help="where to read data from; 'auto' tries the live API first",
    )
    common.add_argument("--horizon", type=int, default=3,
                        help="number of gameweeks to project (default 3)")
    common.add_argument("--gw", type=int, default=None,
                        help="override the first gameweek to project")

    p = sub.add_parser("analyse", parents=[common], help="project and pick a team")
    p.add_argument("--team", default=None, help="path to my_team.json")
    p.add_argument("--bank", type=float, default=None, help="money in the bank (£m)")
    p.add_argument("--free-transfers", type=int, default=None)
    p.add_argument("--max-transfers", type=int, default=3)
    p.add_argument("--top", type=int, default=15,
                   help="how many unowned players to list")
    p.set_defaults(func=cmd_analyse)

    p = sub.add_parser("wildcard", parents=[common], help="best 15 from scratch")
    p.add_argument("--budget", type=float, default=100.0)
    p.set_defaults(func=cmd_wildcard)

    p = sub.add_parser("import-team", help="pull a squad from the FPL API")
    p.add_argument("--entry", type=int, required=True, help="your FPL team id")
    p.add_argument("--gw", type=int, required=True)
    p.add_argument("--free-transfers", type=int, default=1)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_import_team)

    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
