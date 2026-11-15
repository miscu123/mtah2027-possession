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


def standstill(atd, pitch, graded):
    """(s, speed threshold in m/s). s is per frame in [-1, 1]: +1 when the players are clearly
    standing (a stoppage), -1 when they are clearly moving (play), 0 at the threshold or when no
    player is tracked.

    The threshold is found by Otsu's method on log mean player speed over the graded frames.
    """
    v = mean_player_speed(atd, pitch)
    log_v = np.log(np.maximum(v, 0.05))
    threshold = otsu_threshold(log_v[graded])
    s = np.where(np.isfinite(log_v),
                 np.clip((threshold - log_v) / C.SPEED_SPREAD, -1.0, 1.0), 0.0)
    return s, float(np.exp(threshold))


def damp_when_dead_looking(team, still, waiting):
    """Team evidence scaled down while the players are standing or a restart is awaited.

    A player next to the ball while everyone stands still, or after the ball has gone out and
    before it is put back into play, is about to take a free kick, a corner, a goal kick or a
    throw-in -- not controlling a live ball. So "who is nearest" counts in full only while the
    game is moving, fades to nothing at a clear standstill, and is ignored while a restart is
    awaited.
    """
    quiet = np.maximum(np.clip(still, 0.0, 1.0), waiting.astype(np.float64))
    return team * (1.0 - quiet)[:, None]


def ball_out(ball, kind, pitch, untrusted=None):
    """Bool mask: the ball is seen beyond a line by more than OUT_MARGIN.

    untrusted, if given, marks frames whose ball position must not count as "out of play" on its
    own (YOLO detections).
    """
    length, width = pitch
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    trusted = (kind < 2) if untrusted is None else (kind < 2) & ~untrusted
    with np.errstate(invalid="ignore"):
        return trusted & ((bx < -C.OUT_MARGIN) | (bx > length + C.OUT_MARGIN)
                          | (by < -C.OUT_MARGIN) | (by > width + C.OUT_MARGIN))


def awaiting_restart(ball, kind, out, pitch):
    """Bool mask over frames with a ball position: the ball has been out of play and no restart
    has been seen since.

    By the laws of the game a ball that has left the pitch stays dead until it is thrown or kicked
    back into play, even though it is usually carried or rolled back inside the lines first. The
    state starts when the ball has been beyond a line for LATCH_MIN_OUT frames in a row (shorter
    excursions are tracking noise or a ball running along the line) and ends at the first frame
    where the ball is seen inside the lines moving at RESTART_SPEED or more.
    """
    length, width = pitch
    n = len(ball)
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    w = C.RESTART_WINDOW
    speed = np.full(n, np.nan)
    speed[w:-w] = np.hypot(bx[2 * w:] - bx[:-2 * w], by[2 * w:] - by[:-2 * w]) * C.FPS / (2.0 * w)
    inside = (kind < 2) & (bx >= 0) & (bx <= length) & (by >= 0) & (by <= width)
    restart = inside & (np.nan_to_num(speed, nan=0.0) >= C.RESTART_SPEED)

    event = np.zeros(n, dtype=np.int8)       # +1 ball confirmed out, -1 restart seen
    event[restart] = -1
    for start, stop in _runs(out):
        if stop - start >= C.LATCH_MIN_OUT:
            event[start + C.LATCH_MIN_OUT - 1:stop] = 1
    # State at each frame = the most recent event.
    last = np.maximum.accumulate(np.where(event != 0, np.arange(n), -1))
    return (last >= 0) & (event[np.maximum(last, 0)] == 1) & (kind < 2)


def dead_scores(kind, gap_len, n_players, still, out, waiting):
    """Per-frame log-evidence for D.

    still is the output of standstill(), out of ball_out(), waiting of awaiting_restart().
    """
    score = np.zeros(len(kind))

    # Ball beyond a line.
    score[out] += C.K_OUT

    # Players standing still (positive) or clearly moving (negative). While a restart is awaited,
    # players moving is not evidence of live play -- they are fetching the ball and walking back
    # into position -- so the negative side is switched off there.
    score += C.K_SLOW * np.where(waiting, np.maximum(still, 0.0), still)

    # Ball lost for a long time.
    score[(kind == 2) & (gap_len > C.LONG_GAP)] += C.K_LONG_GAP

    # Camera not on the players.
    score[n_players < C.MIN_PLAYERS] += C.K_FEW_PLAYERS
    return score


# -- Smart Tagging ---------------------------------------------------------------------------

def tagging_scores(events, n_frames, home_name, away_name):
    """(team score n x 2, dead score n) from the Smart Tagging intervals.

    Phase tags overlap heavily (a single attack is often tagged PROGRESSION, BALL IN FINAL THIRD
    and PLAYERS IN THE BOX at once), and they all come from the same automatic model, so several
    tags for one team are one opinion, not several. Per frame the tags vote; the side with more
    tags gets K_TAG_TEAM once, and a tie gets nothing.
    """
    votes = np.zeros((n_frames, 2), dtype=np.int32)
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
        votes[start:stop, side] += 1
    team = np.zeros((n_frames, 2))
    team[votes[:, 0] > votes[:, 1], 0] = C.K_TAG_TEAM
    team[votes[:, 1] > votes[:, 0], 1] = C.K_TAG_TEAM
    return team, dead


def fill_silence(team, tag_team):
    """Tag evidence scaled so that it only speaks where the tracking is silent.

    The tracking sees who is on the ball frame by frame; a tag is one automatic guess about a
    whole phase. So the tag counts in full where the tracking has no evidence at all (ball
    missing, nobody near it, ball contested) and fades to nothing as the tracking evidence
    approaches a clean touch.
    """
    silence = 1.0 - np.clip(team.max(axis=1) / C.K_TEAM, 0.0, 1.0)
    return tag_team * silence[:, None]
