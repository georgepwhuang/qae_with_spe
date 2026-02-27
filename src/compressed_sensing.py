import numpy as np
from scipy import linalg as la
from sklearn.linear_model import Lasso


def grid_shifted_dct(N, indices, theta=0):
    """Build a shifted cosine sensing matrix for sampled query depths.

    Parameters
    ----------
    N : int
        Number of cosine basis columns.
    indices : array_like
        Row indices (query depths) at which the basis is evaluated.
    theta : float, default=0
        Fractional shift applied to the sampling grid.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(len(indices), N)`` used for sparse regression.
    """
    n = np.expand_dims(indices, -1)
    k = np.arange(N)
    A = np.cos(np.pi * (2 * (n + theta) + 1) * k / 2 / N)
    return A


def compressed_sensing(p, M, N, rng, alpha=0.1, sample_repeat=100, G=16):
    """Estimate amplitude by sparse cosine recovery with Lasso.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Maximum frequency/grid scale for cosine dictionary.
    N : int
        Target total query budget.
    rng : numpy.random.Generator
        Random number generator for index and sample generation.
    alpha : float, default=0.1
        Lasso regularization weight.
    sample_repeat : int, default=100
        Number of random index-set proposals for budget matching.
    G : int, default=16
        Number of fractional grid shifts evaluated.

    Returns
    -------
    tuple[float, int, int]
        ``(estimate, queries, depth)`` where ``estimate`` is recovered amplitude,
        ``queries`` is realized query cost, and ``depth`` is maximum sampled depth.
    """
    # Ideal noiseless signal model at each query depth.
    grid = np.arange(M) * np.arcsin(np.sqrt(p)) * 2
    signals = (1 + np.cos(grid)) / 2
    signals = np.stack([signals, 1 - signals]).T
    best_indices = None
    best_diff = 1e10
    best_queries = 0
    ev = np.mean(np.arange(M))
    # Randomly search for a sample set whose total query cost is near target N.
    for _ in range(sample_repeat):
        r = np.min((5, N // ev - 1))
        N_s = int(N // ev + rng.uniform(-r, r))
        indices = rng.choice(M, size=N_s)
        queries = np.sum(indices)
        diff = np.abs(queries - N)
        if diff < best_diff:
            best_diff = diff
            best_queries = queries
            best_indices = indices
        if best_diff == 0:
            break
    N_s = len(best_indices)
    depth = np.max(best_indices)
    # Convert multinomial outcomes to ±1-valued observation samples.
    samp = 1 - 2 * rng.multinomial(
        np.ones(N_s, dtype=np.int16), signals[best_indices]
    ).argmax(-1)
    best_l1_norm = 1e10
    best_vx = None
    # Grid over sub-bin shifts; keep the sparsest reconstruction.
    for i in range(G):
        A = grid_shifted_dct(M, best_indices, theta=(i / G - 0.5))
        lasso = Lasso(alpha=alpha)
        lasso.fit(A, samp)
        sig = lasso.coef_.argmax(-1) + i / G - 0.5
        cost = la.norm(lasso.coef_, ord=1)
        if cost < best_l1_norm:
            best_l1_norm = cost
            best_vx = sig
    sig = best_vx / M * np.pi
    res = (1 - np.cos(sig)) / 2
    return res, best_queries, depth
