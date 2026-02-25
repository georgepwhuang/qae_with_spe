import numpy as np


def generate_gaussian_samples(
    p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100
):
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
    samp = rng.multinomial(np.ones(N, dtype=np.int16), signals[best_indices]).argmax(-1)
    samp = 1 - 2 * samp
    return samp, best_indices, best_queries, depth


def glsae(p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, G=16):
    samp, indices, queries, depth = generate_gaussian_samples(
        p, M, N, rng, sigma, gaussian_radius, sample_repeat
    )
    t_list = np.expand_dims(np.arange(M + 1), -1)
    theta = np.mean((samp - np.cos(indices * t_list / M * np.pi)) ** 2, axis=1).argmin(
        -1
    )
    t_list = np.expand_dims(np.arange(8 * G), -1)
    t = np.mean(
        (samp - np.cos(indices * (theta + t_list / G - 4) / M * np.pi)) ** 2, axis=1
    ).argmin(-1)
    out = (1 - np.cos((theta + t / G - 4) / M * np.pi)) / 2
    return out, queries, depth


def gdmae(p, M, N, rng, sigma=4, gaussian_radius=None, sample_repeat=100, G=16):
    samp, indices, queries, depth = generate_gaussian_samples(
        p, M, N, rng, sigma, gaussian_radius, sample_repeat
    )
    t_list = np.expand_dims(np.arange(M + 1), -1)
    theta = np.abs(np.mean(samp * np.cos(indices * t_list / M * np.pi), -1)).argmax(-1)
    t_list = np.expand_dims(np.arange(8 * G), -1)
    t = np.abs(
        np.mean(samp * np.cos(indices * (theta + t_list / G - 4) / M * np.pi), -1)
    ).argmax(-1)
    out = (1 - np.cos((theta + t / G - 4) / M * np.pi)) / 2
    return out, queries, depth
