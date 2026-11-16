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
the tracking once and caches it under `data/cache/` (gitignored, never committed). Generated with
Python 3.12 on Windows 11 and the versions pinned in `requirements-lock.txt`; the pipeline needs
only numpy and pandas.

## The approach, and why

Possession is an interval, not a per-frame decision, and the ranking metric counts each
possession once. So the pipeline is built in two layers:

- **Per-frame evidence** for three states -- home has the ball, away has the ball, dead ball --
  computed from the geometry of the players and the ball. Each piece of evidence is a small,
  explainable rule with a physical reason for its constant.
- **A three-state decoder** (Viterbi) that turns that evidence into whole possessions by charging
  a fixed cost for every change of state.

We chose this over a learned model because there is nothing legal to train one on: the provided
game may not be hand-labelled, the public 5 minutes are not a training or tuning set, and we had
no labelled tracking from other games. A rule-based model with constants taken from football and
physics needs no labels and can be read and checked line by line.

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
6. **Smart Tagging** (`features.tagging_scores`, `features.only_without_ball`, weak priors): set
   pieces add dead-ball evidence at their start. Phases vote for the team in possession (the
   tagged team for attacking phases, the other team for defending ones); overlapping tags for the
   same team count once. The vote is used only on frames where the tracking has no ball at all.
   While the ball is visible the tracking decides alone. Measured without labels: on frames with a
   clean touch the tags name the same team 76% of the time, which is why they are not allowed to
   argue with the tracking.
7. **Who is nearest only counts in live-looking play** (`features.damp_when_dead_looking`): all
   team evidence fades out as the players come to a standstill and is ignored while a restart is
   awaited. A player standing over the ball at a free kick, corner or goal kick is not in
   possession yet.
8. **Kickoff** (`run.kickoff_frame`): a half is graded from its first tracked frame, but the ball
   is not in play before the kickoff. Frames of a half before its kickoff are labelled dead ball
   and decoding starts on the kickoff, in play. The kickoff frames are the ones the organisers
   publish (DATA.md, FAQ.md, `starter/load_smart_tagging.py`); for the second half that is 2,934
   graded frames.
9. **Decoding** (`hmm.viterbi`): a three-state H/A/D Viterbi per half with a fixed cost for every
   change of state, so possession changes only when another team's control outweighs the switch
   cost (~0.4 s of clean control). Brief touches and deflections do not flip it. Where the
   evidence is silent, a state persists and the change is placed as late as possible, at the
   first frame that speaks against it.
10. **Minimum stoppage** (`hmm.drop_short_stoppages`): a dead-ball run shorter than 3 s between two
    stretches of play is given back to the team that had the ball. The ball cannot go out, be
    fetched and be put back into play in less.

All constants are in `solution/config.py`, each with the physical reason for its value. None were
fitted to the public ground truth, and no value was swept or selected by a score on it.

## The possession rules we implemented

- **Control.** A team controls the ball when one of its players is within about a metre of it and
  clearly closer than any opponent. Between 1 and 3 m, or with an opponent almost as close, the
  evidence is proportionally weaker.
- **Loose and contested balls.** They are evidence for nobody, so possession stays with the last
  team that had it. This follows CHALLENGE.md directly. It also fixes where a change of possession
  is placed: at the new team's first touch, not somewhere in the pass or clearance before it.
- **Touches without control.** Changing team costs the equivalent of about 0.4 s of clean control.
  A deflection, a two-frame touch or an aerial duel with no control afterwards does not reach
  that, and is absorbed.
- **Going dead.** The ball is dead when it is seen beyond a line, when the players come to a
  standstill, or when the tracking has lost the ball for more than 3 s. A whistle with the ball
  still on the pitch is only visible through the players stopping.
- **Restarts.** After the ball has been out, play resumes when the ball is seen inside the lines
  moving at pass speed. In other stoppages, play resumes when the players are moving again and a
  team shows control. Until then the taker standing over the ball is not in possession.
- **Kickoffs.** Before the kickoff of a half the ball is dead; from the kickoff frame it is in
  play.
- **Dead-ball ownership** is not labelled; every dead frame is `D`.
- **Abstention.** `X` is written only outside the two graded halves. Inside them the pipeline
  always answers, because on the ranking metric an abstention only repeats the previous label.

## Assets used and dropped

| asset | status | how |
|---|---|---|
| tactical tracking (ATD), players and ball | **used** | all the evidence in steps 1-5 and 7 |
| tactical tracking metadata | **used** | which track slot belongs to which team; team names |
| Smart Tagging | **used, weakly** | team prior where the ball is missing; set pieces as dead-ball evidence |
| published half boundaries and kickoff frames | **used** | graded range and step 8 |
| tactical video | tried, **dropped** | YOLO ball detection, see below |
| tactical homography | tried, **dropped** | only needed by the video experiment |
| broadcast video, tracking and homography | **not used** | not synchronised to the tactical timeline; mapping it back was out of reach in the time available |
| public 5-minute ground truth | **checking only** | see the section on how it was used |

## Training data

None. The submitted pipeline contains no trained model and uses no labelled dataset, from this
game or any other. The only value estimated from data is the player-speed threshold of step 3,
computed by Otsu's method on the match itself with no labels.

The dropped video experiment fine-tuned a COCO-pretrained `yolov8n` (public Ultralytics weights)
on pseudo-labels generated by an algorithm from the provided tracking and homography. No person
labelled or corrected any of them, and the resulting detector is not used in the submission.

## Limitations

- **The ball is missing on 41% of graded frames.** A change of possession inside a gap cannot be
  seen. There the label follows the players' movement for dead ball and the Smart Tagging for the
  team, both of which are weak.
- **Nearest player is not the same as control.** A ball in the air, a ball position that is off
  by a metre, or a defender pressing the carrier closely can all make the wrong team look
  nearest. The pipeline has no touch detection (a change in the ball's velocity) to tell them
  apart.
- **Very short possessions are missed.** The reference contains possessions of about a second.
  With the switch cost we use, a possession under roughly 1.5 s inside another team's spell is
  never detected. Lowering the cost would find some of them and add false ones; we had no legal
  data to choose the balance on.
- **Stoppage boundaries are about a second off.** Players take time to stop after a whistle and
  time to get moving after a restart, and the speed signal is smoothed over 2 s, so dead ball
  tends to start late and end late.
- **Slow restarts.** After the ball has been out, a short restart under 8 m/s (a goal kick rolled
  to a centre-back) is not recognised until the next faster pass.
- **Slow build-up can look like a stoppage.** The standstill signal is the mean speed of all
  tracked players. A team walking the ball around at the back for more than 3 s can be labelled
  dead ball.
- **Dead ball is probably over-called.** The pipeline labels 49% of graded frames dead ball
  (2,934 of them are the frames before the second-half kickoff). In the public window about one
  dead-ball frame in five is play in the reference, most of it at the edges of real stoppages,
  where our dead ball starts early or ends late.
- **Fouls with the ball on the pitch** are detected only through the players stopping; there is
  no whistle signal.
- **Constants are reasoned, not validated.** Every value has a physical justification, but none
  could be checked against labelled data. With more time we would validate them on open tracking
  data from other matches, add touch detection, and synchronise the broadcast tracking to recover
  the ball where the tactical tracker loses it.

## How the public 5 minutes were used

- To check frame alignment and the file format.
- To see how the reference labels stoppages, restarts and short possessions, which CHALLENGE.md
  points to it for, and to locate where the pipeline went wrong. This was done by listing the
  reference segments next to ours, more than once.
- Design decisions that followed from that inspection:
  - Tags were stopped from overruling the tracking (step 6).
  - Team evidence was gated at standstills and while a restart is awaited (step 7).
  - An idea that a stationary ball means a dead ball was dropped, because the window contradicted
    it.
  - The minimum stoppage of step 10 was added after the window showed a 2.8 s dead-ball run
    inside one possession. The 3 s value comes from how long a restart takes, not from the
    window: under the final decoder that same run measures 3.0 s and is not removed, and the
    value was left where it was.
- The awaiting-restart rule (step 4) was strengthened after a first version left one stoppage in
  the window split into pieces. That decision was backed by a count over the whole match that
  needs no labels: predicted possessions lying entirely between the ball going out and the next
  restart.
- Two decisions did not come from the window. The kickoff rule (step 8) comes from the published
  kickoff frames; our earlier output had labelled part of the second half before its kickoff as
  possession. The rule that a state persists through silent frames (step 9) comes from the
  CHALLENGE.md sentence on loose balls and from reading the decoder, which used to break such
  ties differently for the two teams.
- Each version of the pipeline was scored on the window once, as a sanity check. No constant was
  swept and no version was chosen by that score.

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

What we measured (on the pipeline as it stood before the last decoder changes):

- The fine-tuned detector reached precision 0.30 / recall 0.20 on held-out time blocks and was
  less confident than the stock COCO model. The pseudo-label boxes had to be 24 px wide to cover
  the tracking's position error on a ball under 10 px, and they exist only where the tracker
  already sees the ball next to a player -- the opposite of the frames where it is needed.
- Using its detections changed 0.6% of graded labels and the number of possessions by one.
- The missing ball is mostly a stoppage signal, not lost information: the match has 134 gaps,
  almost all longer than 5 s, and the pipeline labelled close to nine in ten of those frames dead
  ball without seeing the ball.
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
