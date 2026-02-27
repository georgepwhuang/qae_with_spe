"""
The following code is slighly modified from
https://github.com/qiskit-community/ChebAE/blob/main/chebae.ipynb.
"""

import numpy as np
from scipy.special import eval_chebyt as cheb
from statsmodels.stats.proportion import proportion_confint


def invert(a_int, deg, p):
    """Invert transformed probability bounds back to amplitude bounds.

    Parameters
    ----------
    a_int : float
        Midpoint amplitude used to identify the local Chebyshev branch.
    deg : int
        Chebyshev polynomial degree used for amplification.
    p : float
        Probability value in transformed space, typically a confidence bound.

    Returns
    -------
    float
        Recovered amplitude estimate in the original domain.
    """
    theta_int = np.arccos(a_int)
    c = np.pi / (2 * deg)
    t = np.floor(theta_int / c)
    if t % 2 == 0:
        theta = np.arccos(2 * p - 1) / (2 * deg)
    else:
        theta = 2 * c - np.arccos(2 * p - 1) / (2 * deg)
    k = t // 2
    theta += np.pi * k / deg
    return np.cos(theta)


def find_next_cheb(a_min, a_max, min_deg=0, odd=False):
    """Select the next usable Chebyshev degree for interval refinement.

    Parameters
    ----------
    a_min : float
        Current lower amplitude bound.
    a_max : float
        Current upper amplitude bound.
    min_deg : int, default=0
        Minimum acceptable degree (exclusive lower threshold in the loop).
    odd : bool, default=False
        If True, enforce odd degrees only.

    Returns
    -------
    int or None
        Largest degree above ``min_deg`` that keeps endpoints in one branch,
        or ``None`` if no such degree exists.
    """
    theta_lo = np.arccos(a_max)
    theta_hi = np.arccos(a_min)
    deg = int((np.pi / 2) / (theta_hi - theta_lo))
    if odd and deg % 2 == 0:
        deg += 1
    while deg > min_deg:
        if int(2 * deg * theta_lo / np.pi) == int(2 * deg * theta_hi / np.pi):
            return deg
        if odd:
            deg -= 2
        else:
            deg -= 1
    return None


def max_error_cp(delta, Nshots):
    """Compute worst-case Clopper-Pearson half-width for a fixed shot count.

    Parameters
    ----------
    delta : float
        Significance level passed to the beta confidence interval.
    Nshots : int
        Number of Bernoulli trials used per confidence computation.

    Returns
    -------
    float
        Maximum half-interval width over all count outcomes in
        ``{0, 1, ..., Nshots}``.
    """
    max_error = 0
    for counts in range(0, Nshots + 1):
        lower, upper = proportion_confint(counts, Nshots, method="beta", alpha=delta)
        if (upper - lower) / 2 > max_error:
            max_error = (upper - lower) / 2

    return max_error


def chebae(a_true, eps, delta, rng, nu=8, r=2, Nshots=100, odd=False):
    """Run adaptive Chebyshev Amplitude Estimation (ChebAE).

    Parameters
    ----------
    a_true : float
        Ground-truth amplitude used by the simulator.
    eps : float
        Target additive error for the final amplitude interval.
    delta : float
        Failure probability budget for confidence intervals.
    rng : numpy.random.Generator
        Random number generator used for Bernoulli sampling.
    nu : float, default=8
        Shot-adaptation threshold multiplier.
    r : int, default=2
        Minimum multiplicative growth factor for candidate degrees.
    Nshots : int, default=100
        Batch shot count when not in single-shot adaptive mode.
    odd : bool, default=False
        If True, restrict selected Chebyshev degrees to odd values.

    Returns
    -------
    tuple[float, int, int]
        ``(estimate, queries, depth)`` where ``estimate`` is the final midpoint,
        ``queries`` is total oracle-query cost, and ``depth`` is max degree used.
    """
    # Maximum number of refinement rounds from geometric interval shrinkage.
    T = int(np.ceil(np.log(1 / (2 * eps)) / np.log(r)))
    delta_T = delta / T
    err_max = max_error_cp(delta_T, Nshots)
    a_min, a_max = 0, 1
    num_flips, num_heads = 0, 0
    deg = 1
    queries = 0
    depth = 0
    while a_max - a_min > eps * 2:
        # Try to increase degree while staying in a single invertible Chebyshev branch.
        new_deg = find_next_cheb(a_min, a_max, min_deg=deg * r, odd=odd)
        if new_deg is not None:
            deg = new_deg
            num_flips, num_heads = 0, 0
        # Adaptive shot rule based on confidence-interval propagation through inverse map.
        gap = cheb(deg, a_max) ** 2 - cheb(deg, a_min) ** 2
        if err_max * (a_max - a_min) / gap < nu * eps:
            Nshots_i = 1
        else:
            Nshots_i = Nshots
        prob = cheb(deg, a_true) ** 2
        if depth < deg:
            depth = deg
        # Accumulate Bernoulli outcomes from transformed success probability.
        for _ in range(Nshots_i):
            if rng.random() < prob:
                num_heads += 1
            num_flips += 1
            queries += deg
        p_min, p_max = proportion_confint(
            num_heads, num_flips, method="beta", alpha=delta_T
        )
        # Map confidence interval bounds back to amplitude interval and intersect.
        a_int = np.mean([a_min, a_max])
        a_min_star = invert(a_int, deg, p_min)
        a_max_star = invert(a_int, deg, p_max)
        a_min_star, a_max_star = sorted([a_min_star, a_max_star])
        a_min_star -= 1e-15
        a_max_star += 1e-15
        a_min, a_max = max(a_min, a_min_star), min(a_max, a_max_star)
    return np.mean([a_min, a_max]), queries, depth
