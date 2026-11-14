# Possession pipeline

Built on top of the official starter repository. Everything added for the submission is in
`solution/` and `run.py`.

## Reproduce the submission

```bash
pip install -r requirements-lock.txt
# unpack the data bundle into ./data (see DATA.md), or set MTAH_DATA_DIR
python run.py --name <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

The run is deterministic: the same data produces the same file, byte for byte. The first run parses
the tracking once and caches it under `data/cache/` (gitignored, never committed).

## How it works

1. **Ball repair** (`features.repair_ball`): drop physically impossible jumps (>45 m/s both in and
   out), linearly bridge gaps up to 2 s, leave longer gaps missing.
2. **Team control evidence** (`features.team_scores`): distance from the ball to the nearest home
   and nearest away player, in metres. A player within 1 m, clearly closer than any opponent, is
   strong evidence for their team; it fades to nothing at 3 m or when the ball is contested.
3. **Dead-ball evidence** (`features.dead_scores`): ball beyond a line; players standing still
   (mean tracked-player speed, threshold found by Otsu's method over the whole match, no labels);
   ball lost for over 3 s; fewer than 10 players tracked.
4. **Smart Tagging** (`features.tagging_scores`, weak priors): set pieces add dead-ball evidence at
   their start; phases add a little evidence for the team in possession (the tagged team for
   attacking phases, the other team for defending ones).
5. **Decoding** (`hmm.viterbi`): a three-state H/A/D Viterbi per half with fixed penalties for every
   change of state, so possession changes only when another team's control outweighs the switch
   cost (~0.4 s of clean control). Brief touches and deflections do not flip it.

## Optional: the video (YOLO ball detection)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-yolo.txt
python detect_ball.py                 # once; ~1-2 h on a GTX 1650, resumable
python run.py --name <team_name> --yolo
```

`detect_ball.py` decodes the tactical video in order and runs a pretrained COCO YOLO model
(class "sports ball", no training) only on graded frames where the ball is still missing after
gap repair. Box centres are mapped to pitch coordinates with the inverse of the provided
homography. `run.py --yolo` fills missing ball positions with those detections: only where the
tracking has no ball, only candidates on or just beside the pitch, choosing the candidate closest
to the last known ball position and rejecting any that are physically out of reach. YOLO balls
count half as much as tracked ones and never signal "out of play" on their own.

### Optional: fine-tuning the detector on pseudo-labels

`make_pseudo_labels.py` builds a training set from the match without any human label: on frames
where the tracking sees the ball and the observation is trustworthy (smooth trajectory, a player
within 2.5 m so the ball is on the ground, valid homography), the ball is projected into the image
and a box sized by projecting a 22 cm ball is written. Crops are at native scale; validation is
held-out time blocks. Then:

```bash
python make_pseudo_labels.py
yolo detect train data=data/yolo_ball_ds/ball.yaml model=yolov8s.pt imgsz=640 epochs=30 batch=8 device=0 seed=0 deterministic=True workers=2 scale=0.25 project=runs name=ball
python detect_ball.py --weights runs/detect/ball/weights/best.pt --cls 0 --out data/cache/yolo_ball_ft.npz
python run.py --name <team_name> --yolo data/cache/yolo_ball_ft.npz
```

GPU training is not bit-for-bit reproducible across machines; the training command and seed are
fixed so it can be re-run, and the resulting weights are what detect_ball.py uses.

All constants are in `solution/config.py`, each with the physical reason for its value. None were
tuned on the public ground truth.
