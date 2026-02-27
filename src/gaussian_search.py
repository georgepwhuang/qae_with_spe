import numpy as np


def generate_gaussian_samples(
    p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100
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
    samp = rng.multinomial(np.ones(N, dtype=np.int16), signals[best_indices]).argmax(-1)
    samp = 1 - 2 * samp
    return samp, best_indices, best_queries, depth


def glsae(p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, G=16):
    """Run GLSAE using Gaussian depth sampling and least-squares matching.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Grid size for phase/amplitude search.
    N : int
        Target total query budget.
    rng : numpy.random.Generator
        Random number generator used internally.
    sigma : float, default=4
        Width parameter for Gaussian depth proposal.
    gaussian_radius : int or None, default=None
        Truncation radius for candidate depth values.
    sample_repeat : int, default=100
        Number of budget-matching proposals.
    G : int, default=16
        Local refinement granularity around the coarse grid estimate.

    Returns
    -------
    tuple[float, int, int]
        ``(estimate, queries, depth)`` from GLSAE.
    """
    samp, indices, queries, depth = generate_gaussian_samples(
        p, M, N, rng, sigma, gaussian_radius, sample_repeat
    )
    # Coarse grid search.
    t_list = np.expand_dims(np.arange(M + 1), -1)
    theta = np.mean((samp - np.cos(indices * t_list / M * np.pi)) ** 2, axis=1).argmin(
        -1
    )
    # Local refinement around the coarse estimate.
    t_list = np.expand_dims(np.arange(8 * G), -1)
    t = np.mean(
        (samp - np.cos(indices * (theta + t_list / G - 4) / M * np.pi)) ** 2, axis=1
    ).argmin(-1)
    out = (1 - np.cos((theta + t / G - 4) / M * np.pi)) / 2
    return out, queries, depth


def gdmae(p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, G=16):
    """Run GDMAE using Gaussian depth sampling and correlation maximization.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Grid size for phase/amplitude search.
    N : int
        Target total query budget.
    rng : numpy.random.Generator
        Random number generator used internally.
    sigma : float, default=4
        Width parameter for Gaussian depth proposal.
    gaussian_radius : int or None, default=None
        Truncation radius for candidate depth values.
    sample_repeat : int, default=100
        Number of budget-matching proposals.
    G : int, default=16
        Local refinement granularity around the coarse grid estimate.

    Returns
    -------
    tuple[float, int, int]
        ``(estimate, queries, depth)`` from GDMAE.
    """
    samp, indices, queries, depth = generate_gaussian_samples(
        p, M, N, rng, sigma, gaussian_radius, sample_repeat
    )
    # Coarse search maximizing absolute cosine correlation.
    t_list = np.expand_dims(np.arange(M + 1), -1)
    theta = np.abs(np.mean(samp * np.cos(indices * t_list / M * np.pi), -1)).argmax(-1)
    # Local refinement around the coarse estimate.
    t_list = np.expand_dims(np.arange(8 * G), -1)
    t = np.abs(
        np.mean(samp * np.cos(indices * (theta + t_list / G - 4) / M * np.pi), -1)
    ).argmax(-1)
    out = (1 - np.cos((theta + t / G - 4) / M * np.pi)) / 2
    return out, queries, depth
