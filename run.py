"""Regenerate the submission from the provided data. One command:

    python run.py --name <team_name>

writes evaluation/results/<team_name>.csv, validates it with the official checks, and prints what
it claims (label counts, number and length of possessions). Deterministic: the same data gives the
same file, byte for byte.

Options:
    --yolo          also use the ball detections from experiments/detect_ball.py (the video)
    --no-tagging    ignore the Smart Tagging (tracking only)
    --out DIR       write somewhere other than evaluation/results/
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402
from solution import atd as atd_io  # noqa: E402
from solution import config as C  # noqa: E402
from solution import features as F  # noqa: E402
from solution import hmm  # noqa: E402

LABELS = np.array(["H", "A", "D"])


def build_labels(use_tagging=True, yolo_path=None):
    game = GAMES[GAME]
    n_frames, pitch = game["n_video_frames"], game["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])

    atd = atd_io.load(meta_path, raw_path, n_frames, cache_dir=os.path.join(DATA_DIR, "cache"))
    print("teams: H = {}, A = {}".format(atd["home"], atd["away"]))

    graded = np.zeros(n_frames, dtype=bool)
    halves = half_video_blocks(GAME)
    for _, first, last in halves:
        graded[first:last + 1] = True

    raw_ball = atd["ball"]
    from_yolo = np.zeros(n_frames, dtype=bool)
    if yolo_path:
        from solution import yolo
        raw_ball, from_yolo = yolo.fill_ball(raw_ball, yolo_path, pitch)
        print("yolo: ball added on {} graded frames where the tracking had none".format(
            int((from_yolo & graded).sum())))

    ball, kind, gap_len = F.repair_ball(raw_ball, pitch)
    from_yolo &= kind == 0  # a YOLO point the glitch filter rejected is no longer a YOLO point
    print("ball: observed {:.1f}%  interpolated {:.1f}%  missing {:.1f}%  (graded frames)".format(
        *(100.0 * np.mean(kind[graded] == k) for k in (0, 1, 2))))

    d_home, d_away = F.nearest_by_team(atd, ball, pitch)
    weight = np.where(from_yolo, C.YOLO_WEIGHT, 1.0)
    team = F.team_scores(d_home, d_away, kind, extra_weight=weight)
    still, speed_threshold = F.standstill(atd, pitch, graded)
    n_players = np.isfinite(atd["px"]).sum(axis=1)
    out = F.ball_out(ball, kind, pitch, untrusted=from_yolo)
    waiting = F.awaiting_restart(ball, kind, out, pitch)
    dead = F.dead_scores(kind, gap_len, n_players, still, out, waiting)
    print("dead-ball speed threshold (Otsu, whole match): {:.2f} m/s".format(speed_threshold))

    if use_tagging:
        import load_smart_tagging
        events = load_smart_tagging.load()
        tag_team, tag_dead = F.tagging_scores(events, n_frames, atd["home"], atd["away"])
        team += F.only_without_ball(tag_team, kind)
        dead += tag_dead
        print("smart tagging: {} intervals used as weak priors".format(len(events)))
    team = F.damp_when_dead_looking(team, still, waiting)

    scores = np.column_stack([team[:, 0], team[:, 1], dead])
    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in halves:
        # A half is graded from its first tracked frame, but the ball is not in play before the
        # kickoff. Those frames are dead ball, and decoding starts on the kickoff itself.
        kickoff = kickoff_frame(half, first, last)
        labels[first:kickoff] = "D"
        path = hmm.drop_short_stoppages(hmm.viterbi(scores[kickoff:last + 1]))
        labels[kickoff:last + 1] = LABELS[path]
        print("half {}: kickoff at frame {} ({} graded frames before it are dead ball)".format(
            half, kickoff, kickoff - first))
    return labels


def kickoff_frame(half, first, last):
    """Video frame of a half's kickoff, from the table the organisers publish in the starter
    (starter/load_smart_tagging.py, the same numbers as DATA.md and FAQ.md), kept inside the
    half. A half with no published kickoff starts in play on its first graded frame."""
    from load_smart_tagging import KICKOFF_VIDEO_FRAMES
    name = {1: "1st Half", 2: "2nd Half"}.get(half)
    return int(min(max(KICKOFF_VIDEO_FRAMES.get(name, first), first), last))


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def describe(path):
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, GAMES[GAME]["n_video_frames"])
    if reason:
        print("\nINVALID: {}".format(reason))
        return False
    codes = pm.label_codes(table)
    runs = pm.valid_runs(codes)
    spells = pm.possession_spells(codes, runs)
    lengths = [b - a for a, b, _ in spells]
    counts = {name: int(np.sum(codes == i)) for i, name in enumerate(pm.LABELS)}
    print("\nvalid -> {}".format(os.path.relpath(path, ROOT)))
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("  {} possessions, mean {:.1f} s, median {:.1f} s".format(
        len(spells), np.mean(lengths) / 25.0 if lengths else 0.0,
        np.median(lengths) / 25.0 if lengths else 0.0))
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="submission", help="team name; the CSV is <name>.csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    parser.add_argument("--no-tagging", action="store_true")
    parser.add_argument("--yolo", nargs="?", const=os.path.join(DATA_DIR, "cache", "yolo_ball.npz"),
                        default=None,
                        help="fill missing ball positions with experiments/detect_ball.py output")
    args = parser.parse_args()

    labels = build_labels(use_tagging=not args.no_tagging, yolo_path=args.yolo)
    path = os.path.join(args.out, "{}.csv".format(args.name))
    write_submission(path, labels)
    ok = describe(path)
    print("\nscore on the public 5 minutes (a correctness check, not a leaderboard):")
    print("  python evaluation/run_evaluation.py")
    sys.exit(0 if ok else 1)
