"""Viterbi decoding over three states, H / A / D.

The per-frame scores are log-evidence; the transition matrix charges a fixed log-penalty for every
change of state. Decoding picks the label sequence with the best total, which is what turns
flickering per-frame evidence into whole possessions: a switch is only worth paying for when the
other side's evidence outweighs it.
"""

import numpy as np

from . import config as C

H, A, D = 0, 1, 2


def transition_matrix():
    """log P(next | prev), as penalties (0 = free)."""
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -C.SWITCH_PENALTY
    t[H, D] = t[A, D] = t[D, H] = t[D, A] = -C.DEAD_PENALTY
    return t


def kickoff_init():
    """Initial scores for a sequence that starts on a kickoff: the ball is in play, so starting
    in D costs what any other stoppage costs to enter."""
    return np.array([0.0, 0.0, -C.DEAD_PENALTY])


def viterbi(scores, trans=None, init=None):
    """Best state path for scores (T x 3).

    Where several paths score the same, the one that changes state as late as possible is taken:
    a state persists until there is evidence against it. That is the CHALLENGE.md rule -- "in
    phases where the ball is contested or loose, possession stays with the last team that had
    it" -- and it is what places a change of possession at the new team's first touch rather than
    somewhere in the pass that preceded it. Remaining ties resolve to the lower state index, so
    the result is deterministic.
    """
    trans = transition_matrix() if trans is None else trans
    init = kickoff_init() if init is None else init
    n, k = scores.shape
    idx = np.arange(k)
    stay_cost = np.diagonal(trans).copy()
    move_cost = trans.copy()
    np.fill_diagonal(move_cost, -np.inf)

    back = np.zeros((n, k), dtype=np.int8)
    best = init + scores[0]
    for t in range(1, n):
        stay = best + stay_cost
        cand = best[:, None] + move_cost          # cand[prev, next], prev != next
        move_from = np.argmax(cand, axis=0)
        move = cand[move_from, idx]
        take_move = move >= stay                  # on a tie, change now rather than earlier
        back[t] = np.where(take_move, move_from, idx)
        best = np.where(take_move, move, stay) + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def drop_short_stoppages(path, min_frames=None):
    """Copy of path with dead-ball runs shorter than min_frames given back to the team that had
    the ball before them.

    A stoppage cannot be over in a second or two: the ball has to be fetched and put back into
    play. A shorter D run between two stretches of play is a lull -- players walking during slow
    build-up -- and by the same rule as above the ball stays with the last team. Runs touching
    either end of the sequence are left alone.
    """
    min_frames = C.MIN_STOPPAGE if min_frames is None else min_frames
    path = path.copy()
    dead = (path == D).astype(np.int8)
    edges = np.diff(np.concatenate(([0], dead, [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start > 0 and stop < len(path) and stop - start < min_frames:
            path[start:stop] = path[start - 1]
    return path
