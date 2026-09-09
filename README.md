# FPL Predictive Team

An expected-points engine and squad optimiser for Fantasy Premier League.

It answers three questions before a deadline:

1. **How many points is each player likely to score?** A per-fixture expected
   points (xP) model, built component by component from the actual FPL scoring
   rules rather than from last season's total.
2. **Which eleven of my fifteen should start, and who captains?** An integer
   programme, not a sort — the best XI and the best captain are chosen jointly
   under the real formation constraints.
3. **Is a transfer worth it?** Every transfer count from 0 to *n* is priced
   against doing nothing, net of the 4-point hit, so the trade-off is visible
   instead of asserted.

## Running this from a phone

You do not need a computer. Two options, both free.

### Option A — GitHub Actions (no install at all)

The projection runs on GitHub's servers and commits the result back here, where
you read it as a normal page in your mobile browser.

1. Open the repository → **Actions** tab → **FPL projection**.
2. Tap **Run workflow**. Fill in free transfers, bank and (optionally) your FPL
   team id, then confirm.
3. Wait about two minutes. The summary appears on the run's own page, and the
   full report is committed to `reports/gw<N>_report.md`.

It also runs itself at 07:00 UTC on Thursdays and Fridays, so a fresh report is
usually waiting before the deadline. Your FPL team id is the number in your
team's web address: `fantasy.premierleague.com/entry/`**`1234567`**`/event/4`.

Note that GitHub's scheduled runs can be delayed by several minutes under load,
so trigger it manually if you are deciding close to the deadline.

### Option B — Google Colab (interactive)

[`notebooks/FPL_Predictive_Team.ipynb`](notebooks/FPL_Predictive_Team.ipynb) —
open it on colab.research.google.com and tap play on each cell. It clones this
repository, installs everything on Google's machine and runs the projection with
live FPL data. Use this when you want to change the horizon or try different
transfer counts and see the answer move.

### Option C — Termux (a real terminal on Android)

Possible, but the least reliable of the three: `pip install pulp` bundles the CBC
solver as a pre-built binary, and I have not verified that a working build ships
for Android's ARM64. If it does not, the optimiser will fail to solve while the
projection itself still works. Try A or B first.

### Reading the output on a small screen

Add `--mobile` and the report is printed as short lines instead of wide tables,
which wrap into nonsense on a phone. Both formats are always written to
`reports/` regardless.

## Quick start (on a computer)

```bash
pip install -r requirements.txt

# project the next gameweek and the two after it
python -m fplpred.cli analyse --horizon 3

# pull your real squad from the FPL API first (entry id is in your team URL)
python -m fplpred.cli import-team --entry 1234567 --gw 4
python -m fplpred.cli analyse --source live --horizon 3 --free-transfers 1

# best possible 15 under budget, ignoring your current squad
python -m fplpred.cli wildcard --budget 100.0

# narrow output for a small screen
python -m fplpred.cli analyse --source live --mobile
```

Reports are printed and written to `reports/gw<N>_report.md`, with the full
player-by-player projection in `reports/gw<N>_projections.csv`.

## How the model works

### 1. Team strength → expected goals in a fixture

Each club gets a multiplicative attack and defence rating centred on 1.00,
blended from two sources:

* a **prior** derived from FPL's own fixture difficulty ratings — specifically,
  the average difficulty assigned to a club's *opponents*, which is a statement
  about that club's quality and is available from the first day of the season;
* **observed** expected goals for and against, season to date.

The blend weight is `matches / (matches + 5)`, so after three gameweeks the
observed data carries about 37% of the weight. Raw observations are winsorised
before blending: a side that conceded 0.2 xG in its opening match implies a
defence rating near 8.0, and one match should not deliver a season-long verdict.

Fixture expected goals then follow a Poisson model:

```
λ_home = 1.45 · attack_home / defence_away · home_advantage
λ_away = 1.45 · attack_away / defence_home / home_advantage
```

### 2. Player rates → shrunk per-90 numbers

Early-season rates are mostly noise. Every rate — xG, xA, defensive actions,
saves, bonus, cards — is shrunk toward a prior in proportion to the evidence
behind it:

```
estimate = (observed · minutes + prior · k) / (minutes + k)
```

The prior is the player's **own previous season**, itself regressed toward the
positional mean; players without one (promoted clubs, new signings, teenagers)
fall back to the positional mean alone. `data/history/` holds the previous
season for exactly this purpose.

### 3. Minutes, conditioned rather than averaged

The model estimates `P(start)` and `P(cameo)` from start rate, price and the
official availability flag, then evaluates the scoring components **twice** — once
for a player who starts, once for a substitute — and weights the results:

```
xP = P(start) · points_given(82 mins) + P(cameo) · points_given(22 mins)
```

This matters because most of FPL scoring is non-linear in minutes. Plugging an
average minutes figure into a clean-sheet term credits a 20-minute substitute
with a fraction of a clean sheet he can never be awarded.

### 4. Components

| Component | How it is modelled |
|---|---|
| Appearance | Step at 60 minutes, per branch |
| Goals / assists | Shrunk xG90 / xA90 × fixture attack multiplier |
| Penalties | Partial top-up for first-choice takers (xG already contains some) |
| Clean sheet | `exp(−λ_conceded · minutes/90)`, only on the 60+ branch |
| Goals conceded | `E[floor(K/2)]` under Poisson — the rule rounds down, so the model must too |
| Saves | `E[floor(K/3)]`, with save volume scaled by fixture difficulty |
| Defensive contribution | `P(X ≥ threshold)` under a **negative binomial**, not a Poisson |
| Bonus | Shrunk per-90 rate |
| Cards | Expected yellows, priced slightly above −1 to absorb rare reds |

The negative binomial is the one that repays attention. Defensive-action counts
are overdispersed, and the players whose +2 is genuinely in doubt are exactly
those averaging a little under the threshold. A Poisson understates them and
overstates the players comfortably clear.

### 5. Optimisation

All three problems are integer linear programmes solved with CBC via PuLP:

* `pick_xi` — best legal XI, captain, vice and bench order for one gameweek.
* `optimise_transfers` — which players to sell and buy, honouring the bank, the
  three-per-club cap, the positional quotas and the point hit.
* `build_squad` — the wildcard problem: the best legal 15 under a budget.

The objective is the discounted horizon projection of the starting XI, plus a
weighted fraction of each bench player's projection, plus the captain's next-
gameweek projection again for the armband.

## Data sources

| Source | Used for | Freshness |
|---|---|---|
| `fantasy.premierleague.com/api` (live) | everything | current |
| `data/snapshot/` | offline fallback | whenever `scripts/refresh.py` last ran |
| `data/history/` | previous-season priors | fixed |

```bash
python scripts/refresh.py                 # from the live API (preferred)
python scripts/refresh.py --from-mirror   # from a public GitHub dataset
```

Every report opens with a provenance block naming the source, the capture time
and the number of completed gameweeks in the data. **Read it.** A projection
built on week-old prices and injury flags is a different claim from one built on
this morning's, and the header is the only place that distinction is recorded.

## What this does not do

Stated plainly, so the output is not mistaken for more than it is:

* **It cannot read a press conference.** Rotation beyond observed start rate and
  the official injury flag is not modelled. Manager comments, European fixtures
  in midweek and cup rotation are yours to apply on top.
* **Bonus is an average, not a simulation.** The BPS table is not reproduced;
  bonus is a shrunk per-90 rate. Right on average, wrong in any single match.
* **Goals are modelled independently of who else scores.** The engine does not
  capture the correlation between a team's goals and which of its players gets
  them, so two attackers from the same club are treated as more independent than
  they are.
* **Expected points are not points.** A 6.0 xP captain still returns 2 more often
  than the number suggests. The model raises the average outcome over a season;
  it makes no promise about any single gameweek.
* **Differentials are not modelled.** The engine maximises raw points, not rank.
  If you are chasing a mini-league from behind, deliberately picking
  lower-owned players with more variance can be correct, and this will not
  suggest it.

## Configuration

`fplpred/config.py` holds every constant: the scoring table, shrinkage weights,
home advantage, bench weights and the horizon discount. The scoring block encodes
the rules as of the 2025/26 season, including the defensive-contribution bonus.
**Verify it against the current official rules before trusting the output** — if
FPL has changed a value, that one edit corrects the whole engine.

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the scoring maths (block rounding, the 60-minute step, position
tables), the threshold distribution's behaviour, and the optimiser's legality
guarantees — every squad it returns satisfies the budget, the quotas and the
three-per-club cap.

## Layout

```
fplpred/
  config.py      scoring rules and every tunable constant
  data.py        live API + snapshot loaders, squad resolution
  strength.py    team ratings and per-fixture multipliers
  rates.py       shrunk per-90 rates and the minutes model
  xpoints.py     the expected-points engine
  optimizer.py   XI, transfer and wildcard ILPs
  report.py      markdown rendering
  cli.py         command-line entry point
.github/workflows/
  fpl-projection.yml   runs the pipeline on GitHub's servers
notebooks/
  FPL_Predictive_Team.ipynb   Colab notebook, for running from a phone
data/
  my_team.json   your squad
  snapshot/      offline copy of the current season
  history/       previous season, used for priors
scripts/refresh.py
tests/test_model.py
```
