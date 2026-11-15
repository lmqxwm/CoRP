"""Trans-dimensional sampler with non-reversible tempering (the alternative to bridge_sampler.py).

Every rung updates each atom, the weights and the count (split/merge); adjacent rungs swap in DEO order; the ladder
(2 B + 1 rungs, at most 48) and kappa_K adapt during burn-in and are then frozen."""
import numpy as np
from helper_functions import run


def trans_sampler(M, n_iter, init=None, burn=0.3, thin=10, seed=0):
    """Pi(K | x) from the frequencies of K on the last rung after burn-in, with its thinned draws (z, w).
    init: (z, w), default one atom at a random data point."""
    if init is None:
        init = (M.x[[np.random.default_rng(seed).integers(M.n)]], np.ones(1))
    out = run(M, n_iter, init, split_merge=True, temper=True, burn=burn, thin=thin, seed=seed)
    out['pK'] = np.bincount(out['K'], minlength=M.K_max + 1)[1:] / len(out['K'])
    return out
