"""Find the ball in the tactical video on the frames where the tracking has lost it.

    pip install -r experiments/requirements-yolo.txt
    python experiments/detect_ball.py                      # ~1-2 h on a GTX 1650; resumable
    python run.py --name <team> --yolo         # uses the detections

Only the graded frames where the ball is still missing after the pipeline's own gap repair are
sent to YOLO -- everything else is already covered by the tracking. Every frame of the video is
decoded in order (never seeked), so frame i of the video is guaranteed to be row i of the
submission; frames that do not need a detection are skipped with grab(), which is cheap.

A pretrained COCO model is used as-is (class 32, "sports ball"). Nothing is trained, and nothing
here touches a label. Each detection's box centre is mapped from pixels to pitch coordinates with
the inverse of the provided homography, so it lands in the same 0-1 coordinates as the ATD.

Output: data/cache/yolo_ball.npz (gitignored with the rest of data/):
    frame (M,)        video frames with at least one candidate
    cand  (M, K, 3)   up to K candidates per frame: pitch x, pitch y (0-1), confidence; NaN-padded
The script checkpoints every few thousand frames and resumes from the checkpoint if interrupted.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402
from solution import atd as atd_io  # noqa: E402
from solution import config as C  # noqa: E402
from solution import features as F  # noqa: E402

SPORTS_BALL = 32  # COCO class id
TOP_K = 3


def frames_to_detect():
    """Boolean mask over video frames: graded, and no ball even after gap repair."""
    game = GAMES[GAME]
    n_frames, pitch = game["n_video_frames"], game["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    atd = atd_io.load(meta_path, raw_path, n_frames, cache_dir=os.path.join(DATA_DIR, "cache"))
    _, kind, _ = F.repair_ball(atd["ball"], pitch)
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    return graded & (kind == 2)


def to_pitch(matrix, u, v):
    """Pixels -> normalised pitch coordinates with the inverse homography."""
    inv = np.linalg.inv(matrix)
    w = inv[2, 0] * u + inv[2, 1] * v + inv[2, 2]
    return (inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]) / w, \
           (inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]) / w


def save(path, results, meta):
    frames = np.array(sorted(results), dtype=np.int64)
    cand = np.full((len(frames), TOP_K, 3), np.nan)
    for i, f in enumerate(frames):
        rows = results[f]
        if rows:
            cand[i, :len(rows)] = rows
    tmp = path + ".tmp.npz"
    np.savez(tmp, frame=frames, cand=cand, **meta)
    os.replace(tmp, path)


def load_partial(path):
    if not os.path.isfile(path):
        return {}
    data = np.load(path)
    out = {}
    for f, rows in zip(data["frame"], data["cand"]):
        out[int(f)] = [list(r) for r in rows if np.isfinite(r[2])]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default="yolov8s.pt",
                        help="any Ultralytics detection weights; downloaded on first use")
    parser.add_argument("--imgsz", type=int, default=1920,
                        help="inference size; the ball is tiny, so keep it at full width")
    parser.add_argument("--cls", type=int, default=SPORTS_BALL,
                        help="class id of the ball: 32 for COCO weights, 0 for fine-tuned ones")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="keep candidates above this; run.py applies its own stricter cut")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="'0' for the first GPU, 'cpu' for CPU")
    parser.add_argument("--limit", type=int, default=0, help="stop after N video frames (debug)")
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "cache", "yolo_ball.npz"))
    args = parser.parse_args()

    import cv2
    import ultralytics
    from ultralytics import YOLO
    import load_homography

    n_frames = GAMES[GAME]["n_video_frames"]
    need = frames_to_detect()
    matrices, _, valid = load_homography.load()
    video_path = require(["video"])[0]

    partial_path = args.out.replace(".npz", ".partial.npz")
    results = load_partial(partial_path)
    done = np.zeros(n_frames, dtype=bool)
    done[list(results)] = True
    todo = need & ~done
    print("frames needing a detection: {} (already done: {})".format(
        int(need.sum()), int((need & done).sum())))

    meta = dict(weights=args.weights, imgsz=args.imgsz, conf=args.conf,
                ultralytics=ultralytics.__version__, cv2=cv2.__version__)
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit("cannot open {}".format(video_path))

    batch_frames, batch_images = [], []
    processed, decoded, started = 0, 0, time.time()
    total = int(todo.sum())

    def flush():
        global processed
        if not batch_images:
            return
        preds = model.predict(batch_images, imgsz=args.imgsz, conf=args.conf,
                              classes=[args.cls], device=args.device, verbose=False)
        for f, pred in zip(batch_frames, preds):
            rows = []
            boxes = pred.boxes
            if boxes is not None and len(boxes) and valid[f]:
                xywh = boxes.xywh.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                keep = (xywh[:, 2] <= C.YOLO_MAX_BOX_PX) & (xywh[:, 3] <= C.YOLO_MAX_BOX_PX)
                order = np.argsort(-conf[keep], kind="stable")[:TOP_K]
                for (u, v, _, _), c in zip(xywh[keep][order], conf[keep][order]):
                    x, y = to_pitch(matrices[f], float(u), float(v))
                    rows.append([x, y, float(c)])
            results[f] = rows
        processed += len(batch_frames)
        batch_frames.clear()
        batch_images.clear()

    last_report = 0
    for f in range(n_frames):
        if args.limit and f >= args.limit:
            break
        if not cap.grab():
            break
        decoded += 1
        if not todo[f]:
            continue
        ok, image = cap.retrieve()
        if not ok:
            print("could not decode frame {}; stopping".format(f))
            break
        batch_frames.append(f)
        batch_images.append(image)
        if len(batch_images) >= args.batch:
            flush()
        if processed - last_report >= 2000:
            last_report = processed
            rate = processed / max(time.time() - started, 1e-9)
            eta = (total - processed) / max(rate, 1e-9) / 60.0
            print("  video frame {:>6}  detected {:>6}/{}  {:.1f} fps  ~{:.0f} min left".format(
                f, processed, total, rate, eta))
            save(partial_path, results, meta)
    flush()
    cap.release()

    if not args.limit:
        if decoded != n_frames:
            print("WARNING: decoded {} frames, expected {}. The video and the tracking would not "
                  "line up -- not writing the final file.".format(decoded, n_frames))
            save(partial_path, results, meta)
            sys.exit(1)
        save(args.out, results, meta)
        if os.path.isfile(partial_path):
            os.remove(partial_path)
        found = sum(1 for rows in results.values() if rows)
        print("done: candidates on {} of {} frames -> {}".format(found, len(results), args.out))
    else:
        save(partial_path, results, meta)
        print("debug run stopped at frame {}; partial results in {}".format(args.limit,
                                                                           partial_path))
