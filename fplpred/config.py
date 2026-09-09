"""Scoring rules and model hyper-parameters.

Everything the model treats as a constant lives here so that a rule change or a
tuning experiment is a one-line edit rather than a hunt through the codebase.

IMPORTANT: the SCORING block encodes the FPL rules as of the 2025/26 season
(including the "defensive contribution" bonus introduced that year). If FPL has
changed any values, edit them here -- the whole engine reads from this block.
Verify against the official rules page before trusting the numbers:
https://fantasy.premierleague.com/help/rules
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Position codes as used by the FPL API (`element_type`)
# --------------------------------------------------------------------------
GK, DEF, MID, FWD = 1, 2, 3, 4
POS_NAME = {GK: "GK", DEF: "DEF", MID: "MID", FWD: "FWD"}
POS_CODE = {v: k for k, v in POS_NAME.items()}

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
SCORING = {
    "appearance_any": 1,          # played at all
    "appearance_60": 1,           # additional point for 60+ minutes
    "goal": {GK: 10, DEF: 6, MID: 5, FWD: 4},
    "assist": 3,
    "clean_sheet": {GK: 4, DEF: 4, MID: 1, FWD: 0},
    "saves_per_point": 3,         # 1 point per N saves
    "penalty_save": 5,
    "penalty_miss": -2,
    "conceded_per_penalty": 2,    # -1 for every N conceded (GK/DEF)
    "conceded_penalty": -1,
    "yellow_card": -1,
    "red_card": -3,
    "own_goal": -2,
    # Defensive contribution: +2 when a player reaches the threshold count of
    # qualifying defensive actions in a match.
    "defensive_contribution": 2,
    "dc_threshold": {GK: 999, DEF: 10, MID: 12, FWD: 12},
}

# Bonus points are modelled from a per-90 rate rather than by simulating BPS.
MAX_BONUS = 3

# --------------------------------------------------------------------------
# Team-strength model
# --------------------------------------------------------------------------
# League-average goals scored by one team in one match. Long-run Premier League
# value is close to 1.45; used as the scale for the Poisson goal model.
LEAGUE_GOALS_PER_TEAM = 1.45

# Multiplicative home advantage applied to the home side's expected goals and
# divided out of the away side's.
HOME_ADVANTAGE = 1.10

# Matches of current-season evidence needed before observed team form carries
# equal weight to the prior. Low K = trust this season quickly.
TEAM_SHRINK_MATCHES = 5.0

# Maps FPL's fixture difficulty rating (1..5, from the perspective of the team
# facing it) onto a rough opponent-quality multiplier. Used to build the
# pre-season prior when little or no current-season data exists.
FDR_TO_STRENGTH = {1: 0.72, 2: 0.85, 3: 1.00, 4: 1.16, 5: 1.32}

# Clamp on any single fixture multiplier, so one extreme rating cannot dominate.
FIXTURE_MULT_CLAMP = (0.55, 1.75)

# Clamp applied to a *raw observation* of team attack/defence before it is
# blended with the prior. Necessary early in a season, when one match of xG can
# imply a rating of 8.0 and would otherwise swamp the blend.
OBSERVED_RATING_CLAMP = (0.45, 2.20)

# --------------------------------------------------------------------------
# Player rate model (empirical-Bayes shrinkage)
# --------------------------------------------------------------------------
# Minutes of current-season evidence at which an observed per-90 rate is
# weighted equally against its prior. Larger = more conservative.
RATE_SHRINK_MINUTES = {
    "xg90": 700.0,
    "xa90": 800.0,
    "dc90": 400.0,
    "saves90": 400.0,
    "bonus90": 900.0,
    "yellow90": 900.0,
}

# Minutes of *previous-season* evidence required before that player's own prior
# season is used as the prior (below this, fall back to the positional mean).
PRIOR_SEASON_MIN_MINUTES = 500.0

# How much a player's previous season is regressed toward the positional mean
# before being used as this season's prior. 0 = use it raw, 1 = ignore it.
PRIOR_SEASON_REGRESSION = 0.25

# Overdispersion factor for defensive-contribution counts. Real counts are
# more variable than Poisson; >1 widens the distribution (negative binomial).
DC_OVERDISPERSION = 1.45

# --------------------------------------------------------------------------
# Minutes model
# --------------------------------------------------------------------------
MINUTES_IF_START = 82.0       # expected minutes for a player who starts
MINUTES_IF_CAMEO = 22.0       # expected minutes for a substitute appearance
P60_GIVEN_START = 0.88        # P(reaches 60 mins | started)
P60_GIVEN_CAMEO = 0.04        # P(reaches 60 mins | came off the bench)

# Team matches of evidence at which observed start-rate outweighs the prior.
START_SHRINK_MATCHES = 3.0

# Availability multipliers by FPL `status` when `chance_of_playing_next_round`
# is not supplied by the API.
STATUS_AVAILABILITY = {
    "a": 1.00,   # available
    "d": 0.55,   # doubtful
    "i": 0.00,   # injured
    "s": 0.00,   # suspended
    "u": 0.00,   # unavailable
    "n": 0.00,   # not in squad / ineligible
}

# --------------------------------------------------------------------------
# Squad rules
# --------------------------------------------------------------------------
SQUAD_SIZE = 15
SQUAD_BY_POS = {GK: 2, DEF: 5, MID: 5, FWD: 3}
MAX_PER_CLUB = 3
XI_SIZE = 11
XI_MIN = {GK: 1, DEF: 3, MID: 2, FWD: 1}
XI_MAX = {GK: 1, DEF: 5, MID: 5, FWD: 3}
TRANSFER_HIT = 4              # points deducted per transfer beyond the free ones

# --------------------------------------------------------------------------
# Planning horizon
# --------------------------------------------------------------------------
# Weight applied to gameweek k of the horizon (k=0 is the upcoming one).
# Future weeks are discounted because transfers can still be made before them.
HORIZON_DISCOUNT = 0.82

# Probability that a bench player in slot i actually contributes points.
# Slot 0 is the first outfield substitute.
BENCH_WEIGHTS = [0.20, 0.11, 0.05]
BENCH_GK_WEIGHT = 0.03

# How strongly defensive-action volume responds to fixture difficulty. A player
# facing a stronger side spends more time out of possession and racks up more
# tackles/interceptions/recoveries. 0 disables the adjustment.
DC_FIXTURE_ELASTICITY = 0.45

# Same idea for goalkeeper saves: more shots faced against stronger opposition.
SAVES_FIXTURE_ELASTICITY = 0.85

# Expected penalties awarded to an average team per match, and the conversion
# rate. Used to top up the designated taker, whose season xG may not yet
# include a spot-kick.
PENS_PER_TEAM_MATCH = 0.115
PEN_CONVERSION = 0.79

# Fraction of a first-choice penalty taker's expected spot-kicks treated as
# *not already* reflected in his shrunk xG90. A full top-up would double-count
# (FPL's expected_goals includes penalties taken); zero would under-rate a taker
# whose sample happens to contain none. Half is the deliberate compromise.
PEN_TOPUP_SHARE = 0.5

# Share of a team's penalties taken by its designated first-choice taker.
PEN_TAKER_SHARE = 0.85

# Probability a goalkeeper saves a penalty he faces.
PEN_SAVE_RATE = 0.21

# Points charged per expected yellow card. Slightly worse than -1 so that the
# rarer red cards are accounted for without modelling them separately.
CARD_POINTS_PER_YELLOW = -1.15
