import numpy as np


def power_law(p, eps, beta, rng, n_shots=10):
    theta = np.arcsin(np.sqrt(p))

    theta_grid = np.arange(0, np.pi / 2 + eps, eps)

    k = np.arange(1, np.floor(np.max((eps ** (-2 * beta), np.log10(1 / eps)))) + 1)
    m_k = np.floor(k ** ((1 - beta) / (2 * beta)))
    depths = 2 * m_k + 1
    signals = np.cos(depths * theta) ** 2
    samples = rng.binomial(n_shots, signals)
    nk1 = samples
    nk0 = n_shots - nk1

    log_cos = np.log(np.cos(depths * np.expand_dims(theta_grid, -1)) ** 2 + 1e-15)
    log_sin = np.log(np.sin(depths * np.expand_dims(theta_grid, -1)) ** 2 + 1e-15)
    log_posterior = np.sum(nk1 * log_cos + nk0 * log_sin, -1)

    estimate_idx = np.argmax(log_posterior)
    estimate = theta_grid[estimate_idx]
    estimate = np.sin(estimate) ** 2

    total_queries = np.sum(depths) * n_shots
    max_depth_used = np.max(depths)

    return estimate, total_queries, max_depth_used
