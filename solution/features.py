"""Per-frame evidence for the three states H, A, D, as log-scores.

Every function here is a plain function of the tracking (and optionally the Smart Tagging). Nothing
reads a ground truth.
"""

import numpy as np

from . import config as C

H, A, D = 0, 1, 2


# -- ball ------------------------------------------------------------------------------------

def _runs(mask):
    """[(start, stop)] half-open runs where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def repair_ball(ball, pitch):
    """(ball_repaired, kind, gap_length).

    kind: 0 observed, 1 interpolated, 2 missing. gap_length: for missing/interpolated frames, the
    length of the gap they sit in (0 for observed frames).
    """
    length, width = pitch
    ball = ball.copy()
    seen = np.isfinite(ball[:, 0])

    # Drop physically impossible jumps between consecutive observations.
    idx = np.flatnonzero(seen)
    if len(idx) > 1:
        dt = np.diff(idx) / float(C.FPS)
        dx = np.diff(ball[idx, 0]) * length
        dy = np.diff(ball[idx, 1]) * width
        speed = np.hypot(dx, dy) / dt
        # A glitch is a point that is too fast both to reach and to leave.
        bad = np.zeros(len(idx), dtype=bool)
        fast = speed > C.MAX_BALL_SPEED
        bad[1:-1] = fast[:-1] & fast[1:]
        ball[idx[bad]] = np.nan
        seen = np.isfinite(ball[:, 0])

    kind = np.where(seen, 0, 2).astype(np.int8)
    gap_len = np.zeros(len(ball), dtype=np.int32)
    for start, stop in _runs(~seen):
        gap_len[start:stop] = stop - start
        if start == 0 or stop >= len(ball) or stop - start > C.MAX_INTERP_GAP:
            continue
        t = (np.arange(start, stop) - (start - 1)) / float(stop - (start - 1))
        ball[start:stop] = ball[start - 1] + t[:, None] * (ball[stop] - ball[start - 1])
        kind[start:stop] = 1
    return ball, kind, gap_len


# -- team control ----------------------------------------------------------------------------

def nearest_by_team(atd, ball, pitch):
    """(d_home, d_away) distance in metres from the ball to the nearest player of each team."""
    length, width = pitch
    dx = (atd["px"] - ball[:, :1]) * length
    dy = (atd["py"] - ball[:, 1:2]) * width
    dist = np.hypot(dx, dy)
    out = []
    for team in (0, 1):
        sub = dist[:, atd["team"] == team]
        with np.errstate(all="ignore"):
            d = np.where(np.isfinite(sub).any(axis=1), np.nanmin(np.where(np.isfinite(sub), sub, np.inf), axis=1), np.nan)
        out.append(d)
    return out[0], out[1]


def team_scores(d_home, d_away, kind, extra_weight=None):
    """Per-frame log-evidence for H and for A from who is touching the ball.

    extra_weight, if given, multiplies the evidence per frame (used to discount YOLO balls).
    """
    n = len(d_home)
    score = np.zeros((n, 2))
    ok = np.isfinite(d_home) & np.isfinite(d_away) & (kind < 2)
    dh, da = d_home[ok], d_away[ok]
    near = np.minimum(dh, da)
    far = np.maximum(dh, da)
    strength = np.clip((C.R_FAR - near) / (C.R_FAR - C.R_NEAR), 0.0, 1.0)
    clean = np.clip((far - near) / C.MARGIN, 0.0, 1.0)
    weight = np.where(kind[ok] == 1, C.INTERP_WEIGHT, 1.0)
    if extra_weight is not None:
        weight = weight * extra_weight[ok]
    e = C.K_TEAM * strength * clean * weight
    rows = np.flatnonzero(ok)
    home_closer = dh < da
    score[rows[home_closer], 0] = e[home_closer]
    score[rows[~home_closer], 1] = e[~home_closer]
    return score


# -- dead ball -------------------------------------------------------------------------------

def mean_player_speed(atd, pitch):
    """Mean speed (m/s) of tracked players per frame, smoothed over +/- SPEED_WINDOW frames."""
    ps = atd["ps"].astype(np.float64)
    if not np.isfinite(ps).any():
        # No speed column in this export: derive it from positions.
        length, width = pitch
        vx = np.diff(atd["px"], axis=0, prepend=np.nan) * length * C.FPS
        vy = np.diff(atd["py"], axis=0, prepend=np.nan) * width * C.FPS
        ps = np.hypot(vx, vy)
    with np.errstate(all="ignore"):
        v = np.nanmean(ps, axis=1)
    finite = np.isfinite(v)
    vz = np.where(finite, v, 0.0)
    w = C.SPEED_WINDOW
    cs = np.concatenate(([0.0], np.cumsum(vz)))
    cn = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(len(v))
    lo, hi = np.maximum(0, i - w), np.minimum(len(v), i + w + 1)
    count = cn[hi] - cn[lo]
    with np.errstate(all="ignore"):
        return np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)


def otsu_threshold(values, bins=256):
    """Threshold splitting values into two classes with maximal between-class variance.

    Unsupervised: it only looks at the distribution of the values, never at a label. Used on log
    player speed, which is bimodal -- standing at a stoppage versus moving in play.
    """
    values = values[np.isfinite(values)]
    hist, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist).astype(np.float64)
    total = weight[-1]
    mean = np.cumsum(hist * centers)
    w0, w1 = weight[:-1], total - weight[:-1]
    with np.errstate(all="ignore"):
        m0 = mean[:-1] / w0
        m1 = (mean[-1] - mean[:-1]) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[np.nanargmax(between)])


def dead_scores(atd, ball, kind, gap_len, pitch, graded, untrusted=None):
    """(per-frame log-evidence for D, speed threshold used).

    untrusted, if given, marks frames whose ball position must not count as "out of play" on its
    own (YOLO detections).
    """
    length, width = pitch
    n = len(ball)
    score = np.zeros(n)

    # Ball beyond a line.
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    trusted = (kind < 2) if untrusted is None else (kind < 2) & ~untrusted
    out = trusted & ((bx < -C.OUT_MARGIN) | (bx > length + C.OUT_MARGIN)
                        | (by < -C.OUT_MARGIN) | (by > width + C.OUT_MARGIN))
    score[out] += C.K_OUT

    # Players standing still.
    v = mean_player_speed(atd, pitch)
    log_v = np.log(np.maximum(v, 0.05))
    threshold = otsu_threshold(log_v[graded])
    spread = 0.3  # in log units: ~35% either side of the threshold reaches full weight
    score += np.where(np.isfinite(log_v),
                      C.K_SLOW * np.clip((threshold - log_v) / spread, -1.0, 1.0), 0.0)

    # Ball lost for a long time.
    score[(kind == 2) & (gap_len > C.LONG_GAP)] += C.K_LONG_GAP

    # Camera not on the players.
    n_players = np.isfinite(atd["px"]).sum(axis=1)
    score[n_players < C.MIN_PLAYERS] += C.K_FEW_PLAYERS
    return score, float(np.exp(threshold))


# -- Smart Tagging ---------------------------------------------------------------------------

def tagging_scores(events, n_frames, home_name, away_name):
    """(team score n x 2, dead score n) from the Smart Tagging intervals."""
    team = np.zeros((n_frames, 2))
    dead = np.zeros(n_frames)
    home, away = home_name.strip().lower(), away_name.strip().lower()
    for event in events:
        code = (event["code"] or "").strip().lower()
        start = max(0, event["start_frame"])
        stop = min(n_frames, event["end_frame"])
        if stop <= start:
            continue
        if code == "set pieces":
            dead[start:start + (stop - start) // 2] += C.K_TAG_SETPIECE
            continue
        tag = (event["tags"].get("Team") or "").strip().lower()
        side = 0 if tag == home else 1 if tag == away else None
        if side is None:
            continue
        if code in C.TAG_TEAM_DEFENDING:
            side = 1 - side
        elif code not in C.TAG_TEAM_IN_POSSESSION:
            continue
        team[start:stop, side] += C.K_TAG_TEAM
    return team, dead
