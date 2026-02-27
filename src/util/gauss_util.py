import numpy as np


def generate_gaussian_samples(
    p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, sine=False, odd_only=False
):
    """Generate depth samples from a Gaussian profile and simulate outcomes.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Maximum supported query depth.
    N : int
        Target total query budget.
    rng : numpy.random.Generator
        Random number generator used for depth and outcome sampling.
    sigma : float, default=4
        Width control for Gaussian depth distribution.
    gaussian_radius : int or None, default=None
        Truncation radius for candidate depths; defaults to ``4 * M``.
    sample_repeat : int, default=100
        Number of attempts to match the target query budget.
    sine : bool, default=False
        Generate sine signal samples.
    odd_only : bool, default=False
        Generate odd degree signals only.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, int, int]
        ``(samples, indices, queries, depth)`` with ±1 outcomes, sampled depths,
        realized query count, and maximum sampled depth.
    """
    if gaussian_radius is None:
        gaussian_radius = 4 * M
    T = M / sigma
    x = np.arange(-gaussian_radius, gaussian_radius)
    probs = np.exp(-0.5 * (x / T) ** 2)
    probs /= probs.sum()
    best_indices = None
    best_diff = 1e10
    best_queries = 0
    ev = np.sum(probs * np.abs(x))
    # Stochastically match the target query budget N.
    for _ in range(sample_repeat):
        r = np.min((5, N // ev - 1))
        N_s = int(N // ev + rng.uniform(-r, r))
        indices = np.abs(rng.choice(x, size=N_s, p=probs))
        indices = np.where(indices <= M, indices, 0)
        queries = np.sum(indices)
        diff = np.abs(queries - N)
        if diff < best_diff:
            best_diff = diff
            best_queries = queries
            best_indices = indices
        if best_diff == 0:
            break
    N = len(best_indices)
    depth = np.max(best_indices)
    grid = np.arange(M + 1) * np.arcsin(np.sqrt(p)) * 2
    signals = (1 + np.cos(grid)) / 2
    signals = np.stack([signals, 1 - signals]).T
    # Convert Bernoulli outcomes into ±1 sample representation.
    cos = rng.multinomial(np.ones(N, dtype=np.int16), signals[best_indices]).argmax(-1)
    cos = 1 - 2 * cos
    if sine:
        signals = (1 + np.sin(grid)) / 2
        signals = np.stack([signals, 1 - signals]).T
        # Convert Bernoulli outcomes into ±1 sample representation.
        sin = rng.multinomial(np.ones(N, dtype=np.int16), signals[best_indices]).argmax(
            -1
        )
        sin = 1 - 2 * sin
        return (cos, sin), best_indices, best_queries, depth
    else:
        return cos, best_indices, best_queries, depth