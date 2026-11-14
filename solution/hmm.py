"""Viterbi decoding over three states, H / A / D.

The per-frame scores are log-evidence; the transition matrix charges a fixed log-penalty for every
change of state. Decoding picks the label sequence with the best total, which is what turns
flickering per-frame evidence into whole possessions: a switch is only worth paying for when the
other side's evidence outweighs it.
"""

import numpy as np

from . import config as C


def transition_matrix():
    """log P(next | prev), as penalties (0 = free)."""
    t = np.zeros((3, 3))
    t[0, 1] = t[1, 0] = -C.SWITCH_PENALTY
    t[0, 2] = t[1, 2] = t[2, 0] = t[2, 1] = -C.DEAD_PENALTY
    return t


def viterbi(scores, trans=None, init=None):
    """Best state path for scores (T x 3). Ties resolve to the lower state index, deterministically."""
    trans = transition_matrix() if trans is None else trans
    if init is None:
        init = np.array([-C.INIT_ALIVE_PENALTY, -C.INIT_ALIVE_PENALTY, 0.0])
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    best = init + scores[0]
    for t in range(1, n):
        cand = best[:, None] + trans          # cand[prev, next]
        back[t] = np.argmax(cand, axis=0)
        best = cand[back[t], np.arange(3)] + scores[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
