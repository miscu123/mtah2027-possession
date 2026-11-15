# Possession pipeline (team FCPS)

The submitted file is the tracking-only run: `python run.py --name FCPS`. A video step (YOLO ball
detection) was tried and is documented at the end; it relabels under 1% of graded frames and is
not used.

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
3. **Standstill** (`features.standstill`): mean tracked-player speed against a threshold found by
   Otsu's method over the whole match (no labels). Players standing is evidence for a stoppage,
   players moving is evidence for live play.
4. **Awaiting restart** (`features.awaiting_restart`): once the ball has been beyond a line for
   0.4 s it is out, and it stays "awaiting restart" until it is seen inside the lines moving at
   8 m/s or more (a throw, a pass, a kick). The ball is usually carried or rolled back onto the
   pitch well before the restart, so "inside the lines" alone does not mean "in play".
5. **Dead-ball evidence** (`features.dead_scores`): ball beyond a line; standstill; ball lost for
   over 3 s; fewer than 10 players tracked. While a restart is awaited, players moving does not
   count as evidence of live play.
6. **Smart Tagging** (`features.tagging_scores`, `features.fill_silence`, weak priors): set pieces
   add dead-ball evidence at their start. Phases vote for the team in possession (the tagged team
   for attacking phases, the other team for defending ones); overlapping tags for the same team
   count once, and the vote is applied only where the tracking is silent (ball missing, nobody
   near it, ball contested). A tag can decide a stretch the tracking cannot see, but never
   outvotes a clean touch. Measured without labels: on frames with a clean touch the tags name
   the same team 76% of the time, which is why they are not allowed to argue with the tracking.
7. **Who is nearest only counts in live-looking play** (`features.damp_when_dead_looking`): all
   team evidence fades out as the players come to a standstill and is ignored while a restart is
   awaited. A player standing over the ball at a free kick, corner or goal kick is not in
   possession yet.
8. **Decoding** (`hmm.viterbi`): a three-state H/A/D Viterbi per half with fixed penalties for every
   change of state, so possession changes only when another team's control outweighs the switch
   cost (~0.4 s of clean control). Brief touches and deflections do not flip it.

## Tried and not used: the video (YOLO ball detection)

The tracking has no ball on 41% of graded frames, so we tried to recover it from the tactical
video. It is not part of the submitted file. The code is kept in `experiments/` and
`solution/yolo.py` so the result can be checked:

- `experiments/detect_ball.py` runs a YOLO model on the graded frames where the ball is still
  missing and maps box centres to pitch coordinates with the inverse homography.
  `run.py --yolo <file>` fills missing ball positions with those detections (never overriding the
  tracking, half weight, never "out of play" on their own).
- `experiments/make_pseudo_labels.py` builds a training set with no human labels by projecting the
  tracked ball into the image on trustworthy frames; a `yolov8n` was fine-tuned on it for 20
  epochs.

What we measured:

- The fine-tuned detector reached precision 0.30 / recall 0.20 on held-out time blocks and was
  less confident than the stock COCO model. The pseudo-label boxes had to be 24 px wide to cover
  the tracking's position error on a ball under 10 px, and they exist only where the tracker
  already sees the ball next to a player -- the opposite of the frames where it is needed.
- Using its detections changes 0.6% of graded labels (973 frames, 268 possessions instead of 267).
- The missing ball is mostly a stoppage signal, not lost information: the match has 134 gaps,
  almost all longer than 5 s, and the pipeline labels 87% of those frames dead ball without
  seeing the ball.
- A ball in the air is mapped to the wrong pitch position by a ground-plane homography, and GPU
  training is not bit-for-bit reproducible, which the rules require of the submitted file.

Commands, for the record:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r experiments/requirements-yolo.txt
python experiments/make_pseudo_labels.py --min-box 24
yolo detect train data=data/yolo_ball_ds/ball.yaml model=yolov8n.pt imgsz=640 epochs=20 batch=8 device=0 seed=0 deterministic=True workers=2 scale=0.25 project=<absolute path of this repo>/runs name=ball exist_ok=True
python experiments/detect_ball.py --weights runs/ball/weights/best.pt --cls 0 --out data/cache/yolo_ball_ft.npz
python run.py --name <team_name> --yolo data/cache/yolo_ball_ft.npz
```

All constants are in `solution/config.py`, each with the physical reason for its value. None were
tuned on the public ground truth: the public 5 minutes were used to check frame alignment and to
see how stoppages and restarts are labelled, never to choose a value or to pick between versions.
