"""Build a YOLO training set for the ball from the match itself, with no human labels.

    python experiments/make_pseudo_labels.py            # -> data/yolo_ball_ds/ (gitignored)
    yolo detect train data=data/yolo_ball_ds/ball.yaml model=yolov8s.pt imgsz=640 epochs=30 ^
         batch=8 device=0 seed=0 deterministic=True workers=2 scale=0.25 project=runs name=ball
    python experiments/detect_ball.py --weights runs/detect/ball/weights/best.pt --cls 0 ^
         --out data/cache/yolo_ball_ft.npz
    python run.py --name <team> --yolo data/cache/yolo_ball_ft.npz

Every label here is produced by an algorithm (RULES.md allows algorithmic pseudo-labels on the
provided game; it forbids human ones). The boxes come from the provided tracking: on frames where
the ATD sees the ball and that observation looks trustworthy, the ball position is projected into
the image with the provided homography. A frame is trusted only if

  - it is graded play and the homography is valid with a low reprojection error,
  - the ball is observed on the frames around it and moves at a plausible speed (no glitch),
  - a player is within a couple of metres of it -- so it is most likely on the ground, where the
    homography (which maps the ground plane) puts it on the right pixel.

The box size is not hand-measured either: a 22 cm ball is projected through the homography at that
spot, so balls far from the camera get smaller boxes.

Training images are crops at the video's native scale (a GTX 1650 cannot train on 1920x1080), so a
model trained on them sees the ball at the same pixel size as detect_ball.py does at --imgsz 1920.
Validation is a held-out set of whole time blocks, never adjacent frames of training images.
The output lives in data/, so it is never committed: it is derived from the match video.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402
from solution import atd as atd_io  # noqa: E402
from solution import features as F  # noqa: E402

BALL_DIAMETER = 0.22   # metres
MAX_SPEED = 30.0       # m/s, between consecutive observations, for a trusted frame
NEAR_PLAYER = 2.5      # metres
MAX_REPROJ = 1.0       # homography reprojection error
VAL_BLOCK = 3000       # frames per time block; every 5th block is validation


def project(m, x, y):
    w = m[2, 0] * x + m[2, 1] * y + m[2, 2]
    return (m[0, 0] * x + m[0, 1] * y + m[0, 2]) / w, (m[1, 0] * x + m[1, 1] * y + m[1, 2]) / w


def trusted_frames(atd, pitch, valid, reproj, graded):
    length, width = pitch
    ball = atd["ball"]
    seen = np.isfinite(ball[:, 0])
    ok = graded & seen & valid & (reproj < MAX_REPROJ)
    # observed on both neighbours, and slow enough between them
    for shift in (-2, -1, 1, 2):
        ok &= np.roll(seen, shift)
    for shift in (-1, 1):
        nb = np.roll(ball, shift, axis=0)
        speed = np.hypot((ball[:, 0] - nb[:, 0]) * length, (ball[:, 1] - nb[:, 1]) * width) * 25
        ok &= np.nan_to_num(speed, nan=np.inf) < MAX_SPEED
    d_home, d_away = F.nearest_by_team(atd, ball, pitch)
    near = np.fmin(d_home, d_away)
    ok &= np.nan_to_num(near, nan=np.inf) < NEAR_PLAYER
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-pos", type=int, default=4000, help="images with a ball")
    parser.add_argument("--n-neg", type=int, default=1000, help="images without a ball")
    parser.add_argument("--crop", type=int, default=640)
    parser.add_argument("--min-box", type=float, default=6.0,
                        help="smallest box side in pixels. The tracked ball position can be off by "
                             "more than the ball's own size, so a larger box tolerates that offset")
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "yolo_ball_ds"))
    args = parser.parse_args()

    import cv2
    import load_homography

    game = GAMES[GAME]
    n_frames, pitch = game["n_video_frames"], game["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    atd = atd_io.load(meta_path, raw_path, n_frames, cache_dir=os.path.join(DATA_DIR, "cache"))
    matrices, reproj, valid = load_homography.load()
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    ok = trusted_frames(atd, pitch, valid, np.nan_to_num(reproj, nan=np.inf), graded)
    candidates = np.flatnonzero(ok)
    print("trusted ball frames: {} of {} graded".format(len(candidates), int(graded.sum())))
    if not len(candidates):
        sys.exit("no trusted frames -- nothing to train on")

    # spread evenly over the match, deterministically
    pos = candidates[np.unique(np.linspace(0, len(candidates) - 1, args.n_pos).astype(int))]
    rng = np.random.RandomState(0)
    neg = np.setdiff1d(candidates, pos)
    neg = np.sort(rng.choice(neg, min(args.n_neg, len(neg)), replace=False)) if len(neg) else neg
    jobs = {int(f): "pos" for f in pos}
    jobs.update({int(f): "neg" for f in neg if int(f) not in jobs})

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    cap = cv2.VideoCapture(require(["video"])[0])
    half = args.crop // 2
    written = {"train": 0, "val": 0}
    sizes = []
    for f in range(n_frames):
        if not cap.grab():
            sys.exit("video ended at frame {}, expected {}".format(f, n_frames))
        if f not in jobs:
            continue
        ok_read, img = cap.retrieve()
        if not ok_read:
            continue
        h_img, w_img = img.shape[:2]
        crop = min(args.crop, h_img, w_img)
        m = matrices[f]
        bx, by = atd["ball"][f]
        u, v = project(m, bx, by)
        # ball size in pixels: project a 22 cm offset along both pitch axes
        du = abs(project(m, bx + BALL_DIAMETER / pitch[0], by)[0] - u)
        dv = abs(project(m, bx, by + BALL_DIAMETER / pitch[1])[1] - v)
        size = float(np.clip(max(du, dv) * 1.3, args.min_box, max(40.0, args.min_box)))
        if jobs[f] == "pos":
            cx = int(np.clip(u + rng.randint(-crop // 3, crop // 3 + 1), crop // 2, w_img - crop // 2))
            cy = int(np.clip(v + rng.randint(-crop // 3, crop // 3 + 1), crop // 2, h_img - crop // 2))
        else:
            # a crop that does not contain the ball
            for _ in range(20):
                cx = rng.randint(crop // 2, w_img - crop // 2 + 1)
                cy = rng.randint(crop // 2, h_img - crop // 2 + 1)
                if abs(cx - u) > crop // 2 + size and abs(cy - v) > crop // 2 + size:
                    break
            else:
                continue
        x0, y0 = cx - crop // 2, cy - crop // 2
        patch = img[y0:y0 + crop, x0:x0 + crop]
        split = "val" if (f // VAL_BLOCK) % 5 == 4 else "train"
        name = "f{:06d}".format(f)
        cv2.imwrite(os.path.join(args.out, "images", split, name + ".jpg"), patch,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as handle:
            if jobs[f] == "pos" and 0 <= u - x0 < crop and 0 <= v - y0 < crop:
                handle.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    (u - x0) / crop, (v - y0) / crop, size / crop, size / crop))
                sizes.append(size)
        written[split] += 1
    cap.release()

    with open(os.path.join(args.out, "ball.yaml"), "w") as handle:
        handle.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(
            os.path.abspath(args.out).replace("\\", "/")))
    print("wrote {} train / {} val images to {}".format(written["train"], written["val"], args.out))
    if sizes:
        print("ball box size in pixels: median {:.1f}, 5-95% {:.1f}-{:.1f}".format(
            np.median(sizes), *np.percentile(sizes, [5, 95])))