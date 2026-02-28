import numpy as np
from src.util.gauss_util import generate_gaussian_samples


def glsae(p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, G=16):
    """Run GLSAE using Gaussian depth sampling and least-squares matching.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Grid size for phase/amplitude search.
    N : int
        Number of parallel queries.
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
    if gaussian_radius is None:
        gaussian_radius = 4 * M
    T = M / sigma
    x = np.arange(-gaussian_radius, gaussian_radius)
    probs = np.exp(-0.5 * (x / T) ** 2)
    probs /= probs.sum()
    indices = np.abs(rng.choice(x, size=N, p=probs))
    indices = np.where(indices <= M, indices, 0)
    queries = np.sum(indices)
    depth = np.max(indices)
    grid = np.arange(M + 1) * np.arcsin(np.sqrt(p)) * 2
    signals = (1 + np.cos(grid)) / 2
    signals = np.stack([signals, 1 - signals]).T
    # Convert Bernoulli outcomes into ±1 sample representation.
    samp = rng.multinomial(np.ones(N, dtype=np.int16), signals[indices]).argmax(-1)
    samp = 1 - 2 * samp
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

def gdmae(
    p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, G=16
):
    """Run GDMAE using Gaussian depth sampling and correlation maximization.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Grid size for phase/amplitude search.
    N : int
        Number of parallel queries.
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
    if gaussian_radius is None:
        gaussian_radius = 4 * M
    T = M / sigma
    x = np.arange(-gaussian_radius, gaussian_radius)
    probs = np.exp(-0.5 * (x / T) ** 2)
    prob_sum = probs.sum()
    probs = np.where(x%2==1, probs, 0)
    probs /= prob_sum
    probs[x==0] = 1 - probs.sum()
    indices = np.abs(rng.choice(x, size=N, p=probs))
    indices = np.where(indices <= M, indices, 0)
    queries = np.sum(indices)
    depth = np.max(indices)
    grid = np.arange(M + 1) * np.arcsin(np.sqrt(p)) * 2
    signals = (1 + np.cos(grid)) / 2
    signals = np.stack([signals, 1 - signals]).T
    # Convert Bernoulli outcomes into ±1 sample representation.
    samp = rng.multinomial(np.ones(N, dtype=np.int16), signals[indices]).argmax(-1)
    samp = 1 - 2 * samp
    t_list = np.expand_dims(np.arange(M + 1), -1)
    t_sharp_list = np.expand_dims(np.arange(8 * G), -1)
    cos_samp, sin_samp = samp
    # Coarse search maximizing absolute cosine correlation.
    theta = np.abs(np.mean(
            cos_samp * np.cos(indices * t_list / M * np.pi)
            + sin_samp * np.sin(indices * t_list / M * np.pi),
            -1)
    ).argmax(-1)
    # Local refinement around the coarse estimate.
    t = np.abs(np.mean(
            cos_samp * np.cos(indices * (theta + t_sharp_list / G - 4) / M * np.pi)
            + sin_samp * np.sin(indices * (theta + t_sharp_list / G - 4) / M * np.pi),
            -1)
    ).argmax(-1)
    out = (1 - np.cos((theta + t / G - 4) / M * np.pi)) / 2
    return out, queries, depth
