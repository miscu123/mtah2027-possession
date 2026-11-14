"""Read the raw ATD into numpy arrays, once, and cache them.

Raw line format (see starter/baseline.py):  <frame_id>:<x,y,s>;<x,y,s>;...:<ball x,y,s>
Positions are normalised 0-1, speeds in m/s, missing values are NaN. frame_id counts from 1.

The parsed arrays are cached as data/cache/atd_<source>.npz. data/ is gitignored, so the cache
never reaches the public repository.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def read_meta(meta_path):
    """(team_of_slot array, home_name, away_name). First team in the file is home."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    names = [team.findtext("Name") or "" for team in teams]
    slots = [player.get("teamId") for player in root.iter("Player")]
    team_of_slot = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return team_of_slot, names[0], names[1]


def _parse_raw(raw_path, n_frames, n_slots):
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # video_frame = frame_id - 1
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";")
            values = np.array(",".join(groups).split(","), dtype=np.float64)
            width = len(values) // max(len(groups), 1)
            if width < 2 or len(groups) != n_slots:
                raise ValueError("unexpected ATD layout at frame_id {}: {} groups, {} values"
                                 .format(parts[0], len(groups), len(values)))
            values = values.reshape(n_slots, width)
            px[row], py[row] = values[:, 0], values[:, 1]
            if width >= 3:
                ps[row] = values[:, 2]
            ball_values = parts[2].split(",")
            if len(ball_values) >= 2 and ball_values[0][:1] != "N":
                ball[row] = float(ball_values[0]), float(ball_values[1])
    return px, py, ps, ball


def load(meta_path, raw_path, n_frames, cache_dir=None):
    """dict with px, py, ps (frames x slots), ball (frames x 2), team (slots), home, away."""
    team, home, away = read_meta(meta_path)
    cache = None
    if cache_dir:
        stat = os.stat(raw_path)
        cache = os.path.join(cache_dir, "atd_{}_{}.npz".format(stat.st_size, int(stat.st_mtime)))
        if os.path.isfile(cache):
            data = np.load(cache)
            return dict(px=data["px"], py=data["py"], ps=data["ps"], ball=data["ball"],
                        team=team, home=home, away=away)

    print("parsing {} (one pass, ~1-2 min; cached afterwards) ...".format(
        os.path.basename(raw_path)))
    px, py, ps, ball = _parse_raw(raw_path, n_frames, len(team))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, px=px, py=py, ps=ps, ball=ball)
    return dict(px=px, py=py, ps=ps, ball=ball, team=team, home=home, away=away)
