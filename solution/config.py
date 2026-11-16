"""Every constant the pipeline uses, in one place, with the reason for its value.

None of these were tuned on the public ground truth (RULES.md forbids it, and EVALUATION.md shows
it picks the wrong answer anyway). They come from football and physics -- how close a player has
to be to control a ball, how long a deliberate touch lasts -- or are derived from the whole match
without labels (the speed threshold, see features.otsu_threshold).
"""

FPS = 25

# -- ball repair -------------------------------------------------------------------------------
# Gaps in the ball track up to this long are bridged by linear interpolation. 2 s: a pass or a
# short occlusion. Longer gaps are left missing -- straight-line guesses over longer spans are
# usually wrong (aerial balls, deflections).
MAX_INTERP_GAP = 2 * FPS
# A gap longer than this counts as weak evidence of a stoppage (ball out of shot, restart).
LONG_GAP = 3 * FPS
# Ball "jumps" faster than this between observed frames are tracker glitches and are dropped.
# The hardest shots in football are ~35-40 m/s.
MAX_BALL_SPEED = 45.0  # m/s

# -- team control evidence ---------------------------------------------------------------------
# A player within R_NEAR metres of the ball is treated as in contact with it; beyond R_FAR there is
# no control evidence at all. Between, evidence falls off linearly.
R_NEAR = 1.0
R_FAR = 3.0
# How much closer the nearest player must be than the nearest opponent for the evidence to be
# clean. Smaller margins are a contested ball and count proportionally less.
MARGIN = 2.0
# Per-frame log-evidence for a clean, close, observed touch. Interpolated ball counts half.
K_TEAM = 1.0
INTERP_WEIGHT = 0.5

# -- dead ball evidence ------------------------------------------------------------------------
# Ball beyond a line by more than this many metres: out of play.
OUT_MARGIN = 0.5
K_OUT = 2.0
# Players slow down at stoppages. Mean tracked-player speed is smoothed over +/- SPEED_WINDOW
# frames and compared with a threshold found by Otsu's method over the whole match (no labels).
SPEED_WINDOW = FPS
K_SLOW = 0.6
# Width of the ramp around that threshold, in log units: the evidence reaches full weight when the
# mean speed is ~35% below (standing) or above (moving) the threshold.
SPEED_SPREAD = 0.3
# After the ball has been beyond a line for this many frames in a row (0.4 s: not a one-frame
# tracking excursion) it is out, and stays "awaiting restart" until it is seen inside the lines
# moving at RESTART_SPEED or more. A pass, throw-in or kick is well above 8 m/s; a ball carried or
# rolled back to the spot is below it. Speed is measured over +/- RESTART_WINDOW frames.
LATCH_MIN_OUT = 10
RESTART_SPEED = 8.0  # m/s
RESTART_WINDOW = 3
K_LONG_GAP = 0.3
# Fewer tracked players than this: camera not on the pitch (halftime, replays) -> not play.
MIN_PLAYERS = 10
K_FEW_PLAYERS = 0.5

# -- Smart Tagging priors (weak on purpose: the tags are automatic and imprecise) ----------------
K_TAG_SETPIECE = 0.5   # D evidence over the first half of a SET PIECES window
# Team evidence per frame inside a phase interval. Overlapping tags for the same team count once
# (features.tagging_scores), and the evidence is applied only on frames where the tracking has no
# ball at all (features.only_without_ball), so a tag can decide an unseen stretch but never
# argues with what the tracking shows.
K_TAG_TEAM = 0.15
# Codes whose Team tag is the team IN possession, and codes whose Team tag is the team WITHOUT the
# ball. Codes not listed here contribute no team evidence.
TAG_TEAM_IN_POSSESSION = (
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box",
)
TAG_TEAM_DEFENDING = (
    "defending in defensive third", "defending in middle third", "defending in attacking third",
    "defensive transition",
)

# -- YOLO ball detections (optional, run.py --yolo) ---------------------------------------------
# A COCO "sports ball" box larger than this, in pixels, is not the ball in a wide tactical shot.
YOLO_MAX_BOX_PX = 60
# Candidates below this confidence are ignored when filling the ball track.
YOLO_MIN_CONF = 0.25
# Accept detections up to this many metres outside the lines (the ball crossing a line is real;
# a spare ball held by a ball boy further out is not).
YOLO_PITCH_MARGIN = 2.0
# How far back the last known ball position may be and still be used to choose between candidates.
YOLO_ANCHOR_FRAMES = 2 * FPS
# Control evidence from a YOLO ball counts this much relative to an ATD observation, and it never
# counts as "ball out of play" on its own.
YOLO_WEIGHT = 0.5

# -- HMM ---------------------------------------------------------------------------------------
# Log-penalty for changing state. With K_TEAM = 1 per frame, a team switch needs roughly
# SWITCH_PENALTY frames of clean control by the other side (~0.4 s), so a deflection or a
# 2-3 frame touch does not flip possession -- the CHALLENGE.md definition of "controlled".
SWITCH_PENALTY = 10.0
# Entering or leaving dead ball. Stoppages last seconds, so this is set higher than a switch.
# Decoding starts on the kickoff of each half, where the ball is in play, so starting in D costs
# the same (hmm.kickoff_init).
DEAD_PENALTY = 12.0
# A stoppage shorter than this does not happen: the ball has to be fetched and put back into play,
# and even a quick throw-in takes about three seconds. Shorter D runs between two stretches of
# play are given back to the team that had the ball (hmm.drop_short_stoppages).
MIN_STOPPAGE = 3 * FPS
