"""
The following code is a reimplementation of the cosine ESPRIT algorithm
from https://doi.org/10.1007/s11075-022-01432-6.
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as sla


def esprit_cosine(p, M, N_s, rng, greater_than_half=False):
    """Estimate amplitude with the cosine-ESPRIT procedure.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Odd-depth grid cap; depths are ``1, 3, ..., M-1``.
    N_s : int
        Number of repeated samples per odd depth.
    rng : numpy.random.Generator
        Random number generator for multinomial sampling.
    greater_than_half : bool, default=False
        Selects inverse mapping branch for amplitudes above ``0.5``.

    Returns
    -------
    tuple[float, int, int]
        ``(estimate, queries, depth)`` where ``estimate`` is amplitude,
        ``queries`` is total query usage, and ``depth`` is max circuit depth.
    """
    queries = np.sum(np.arange(1, M, 2)) * N_s
    grid = np.arange(1, M, 2) * np.arcsin(np.sqrt(p)) * 2
    depth = np.max(np.arange(1, M, 2))
    # Generate ±1 samples from Bernoulli observations at each odd depth.
    signals = (1 + np.cos(grid)) / 2
    signals = np.stack([signals, 1 - signals]).T
    samp = rng.multinomial(np.ones((N_s, 1), dtype=np.int16), signals).argmax(-1)
    samp = np.mean(1 - 2 * samp, 0)

    # Build the structured matrix used by cosine-ESPRIT.
    L = (M + 3) // 4
    c = np.abs(np.arange(-1, M - 2 * L + 3, 2))
    t = np.abs(np.arange(-1, -2 * L, -2))
    h = np.abs(np.arange(c[-1], M + 1, 2))
    c = samp[((c - 1) / 2).astype(int)]
    t = samp[((t - 1) / 2).astype(int)]
    h = samp[((h - 1) / 2).astype(int)]
    m = la.toeplitz(c, t) + la.hankel(c, h)

    if m.shape[-1] == 1:
        U, _, _ = la.svd(m)
        U = U[:, :1]
    else:
        U, _, _ = sla.svds(m, 1)
    # Single-mode rotational invariance step.
    phi = la.pinv(U[1:-1]) @ (U[0:-2] + U[2:])
    eigs, _ = la.eig(phi)
    eigs = np.clip(eigs, -2, 2)[0]
    if greater_than_half:
        res = (1 - np.sqrt((1 - eigs.real / 2) / 2)) / 2
    else:
        res = (1 - np.sqrt((1 + eigs.real / 2) / 2)) / 2
    return res, queries, depth
