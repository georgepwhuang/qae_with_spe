import numpy as np
from scipy.special import eval_chebyt as cheb
from statsmodels.stats.proportion import proportion_confint


def invert(a_int, deg, p):
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
    max_error = 0
    for counts in range(0, Nshots + 1):
        lower, upper = proportion_confint(counts, Nshots, method="beta", alpha=delta)
        if (upper - lower) / 2 > max_error:
            max_error = (upper - lower) / 2

    return max_error


def chebae(a_true, eps, delta, rng, nu=8, r=2, Nshots=100, odd=False):
    T = int(np.ceil(np.log(1 / (2 * eps)) / np.log(r)))
    delta_T = delta / T
    err_max = max_error_cp(delta_T, Nshots)
    a_min, a_max = 0, 1
    num_flips, num_heads = 0, 0
    deg = 1
    queries = 0
    depth = 0
    while a_max - a_min > eps * 2:
        new_deg = find_next_cheb(a_min, a_max, min_deg=deg * r, odd=odd)
        if new_deg is not None:
            deg = new_deg
            num_flips, num_heads = 0, 0
        gap = cheb(deg, a_max) ** 2 - cheb(deg, a_min) ** 2
        if err_max * (a_max - a_min) / gap < nu * eps:
            Nshots_i = 1
        else:
            Nshots_i = Nshots
        prob = cheb(deg, a_true) ** 2
        if depth < deg:
            depth = deg
        for _ in range(Nshots_i):
            if rng.random() < prob:
                num_heads += 1
            num_flips += 1
            queries += deg
        p_min, p_max = proportion_confint(
            num_heads, num_flips, method="beta", alpha=delta_T
        )
        a_int = np.mean([a_min, a_max])
        a_min_star = invert(a_int, deg, p_min)
        a_max_star = invert(a_int, deg, p_max)
        a_min_star, a_max_star = sorted([a_min_star, a_max_star])
        a_min_star -= 1e-15
        a_max_star += 1e-15
        a_min, a_max = max(a_min, a_min_star), min(a_max, a_max_star)
    return np.mean([a_min, a_max]), queries, depth
