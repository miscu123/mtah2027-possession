"""Fill frames where the tracking has no ball with YOLO detections from detect_ball.py.

Only frames where the ATD has no ball are touched; an ATD observation always wins. A candidate is
accepted only if it is confident enough and lands on (or just beside) the pitch. Among accepted
candidates the one closest to the last known ball position is taken, because a second ball near
the touchline or a white boot is more often the highest-confidence box than you would like.
"""

import os

import numpy as np

from . import config as C


def fill_ball(ball, det_path, pitch):
    """(ball with gaps filled where YOLO saw it, bool mask of the frames that came from YOLO)."""
    if not os.path.isfile(det_path):
        raise IOError("no detections at {}; run python detect_ball.py first".format(det_path))
    data = np.load(det_path)
    frames, cand = data["frame"], data["cand"]
    length, width = pitch
    mx, my = C.YOLO_PITCH_MARGIN / length, C.YOLO_PITCH_MARGIN / width

    out = ball.copy()
    from_yolo = np.zeros(len(ball), dtype=bool)
    observed = np.flatnonzero(np.isfinite(ball[:, 0]))

    for f, rows in zip(frames, cand):
        f = int(f)
        if np.isfinite(ball[f, 0]):
            continue
        ok = (np.isfinite(rows[:, 2]) & (rows[:, 2] >= C.YOLO_MIN_CONF)
              & (rows[:, 0] >= -mx) & (rows[:, 0] <= 1 + mx)
              & (rows[:, 1] >= -my) & (rows[:, 1] <= 1 + my))
        if not ok.any():
            continue
        rows = rows[ok]
        # last known position: the latest ATD ball before f, or the previous YOLO fill
        k = np.searchsorted(observed, f) - 1
        anchor, age = None, None
        if k >= 0:
            anchor, age = ball[observed[k]], f - observed[k]
        if from_yolo[max(0, f - 1)] and (age is None or age > 1):
            anchor, age = out[f - 1], 1
        if anchor is not None and age <= C.YOLO_ANCHOR_FRAMES:
            dist = np.hypot((rows[:, 0] - anchor[0]) * length, (rows[:, 1] - anchor[1]) * width)
            reach = C.MAX_BALL_SPEED * age / C.FPS
            if (dist <= reach).any():
                pick = int(np.argmin(np.where(dist <= reach, dist, np.inf)))
            else:
                continue  # every candidate is physically out of reach: probably not our ball
        else:
            pick = 0  # rows are confidence-sorted
        # Clip to just inside the "out of play" margin, so a YOLO point (or a line interpolated
        # through it) can never by itself make the ball look out of play.
        lim_x, lim_y = 0.9 * C.OUT_MARGIN / length, 0.9 * C.OUT_MARGIN / width
        out[f] = (np.clip(rows[pick, 0], -lim_x, 1 + lim_x),
                  np.clip(rows[pick, 1], -lim_y, 1 + lim_y))
        from_yolo[f] = True
    return out, from_yolo
