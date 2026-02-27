import numpy as np
from src.util.gauss_util import generate_gaussian_samples


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
        p, M, N, rng, sigma, gaussian_radius, sample_repeat, sine=True, odd_only=True
    )
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
