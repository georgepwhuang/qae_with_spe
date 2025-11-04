import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as sla


def esprit_cosine(p, M, N_s, rng, greater_than_half=False):
    queries = np.sum(np.arange(1, M, 2)) * N_s
    grid = np.arange(1, M, 2) * np.arcsin(np.sqrt(p)) * 2
    depth = np.max(np.arange(1, M, 2))
    signals = (1 + np.cos(grid)) / 2
    signals = np.stack([signals, 1 - signals]).T
    samp = rng.multinomial(np.ones((N_s, 1), dtype=np.int16), signals).argmax(-1)
    samp = np.mean(1 - 2 * samp, 0)

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
    phi = la.pinv(U[1:-1]) @ (U[0:-2] + U[2:])
    eigs, _ = la.eig(phi)
    eigs = np.clip(eigs, -2, 2)[0]
    if greater_than_half:
        res = (1 - np.sqrt((1-eigs.real/2)/2))/2
    else:
        res = (1 - np.sqrt((1+eigs.real/2)/2))/2
    return res, queries, depth


def csae(p, q, rng, C=5, cache=None):
    access = np.concat((np.zeros(1), 2 ** np.arange(2 * q))) * 2 + 1
    depth = np.max(access)
    grid = access * np.arcsin(np.sqrt(p)) * 2
    N_s = C * (np.arange(2 * q, -1, -1) + 1)
    N_s[0] *= 2
    queries = np.sum(access * N_s)
    prob = (1 + np.cos(grid)) / 2
    samp = []
    for p, s in zip(prob, N_s):
        r = np.sum(rng.binomial(1, p, s) == 1)/s
        samp.append(r)
    samp = np.array(samp)
    samp = 1 - 2 * samp
    sign = np.sign(np.sin(grid))
    signal = samp + 1j * sign * np.sqrt(1 - samp**2)
    signal = signal * signal[0].conj()
    o_sig = np.copy(signal)
    if cache is None:
        o_arr = np.concat((np.zeros(1), 2 ** np.arange(2 * q)))
        arr = np.copy(o_arr)
        for i in range(q):
            arr = (o_arr - np.tile(arr, (len(o_arr), 1)).T).ravel()
            arr, indices = np.unique(arr, return_index=True)
            signal = np.outer(o_sig, signal.conj()).T
            signal = signal.ravel()[indices]
        final_indices = np.nonzero(np.abs(arr - 2 ** (2 * q - 2)) <= 2 ** (2 * q - 2))
    else:
        for i in range(q):
            indices = cache[i]
            signal = np.outer(o_sig, signal.conj()).T
            signal = signal.ravel()[indices]
        final_indices = cache[q]
    ula_sig = signal[final_indices].ravel()
    covar = la.toeplitz(ula_sig, ula_sig.conj())
    U, _, _ = sla.svds(covar, 1)
    phi = la.pinv(U[:-1]) @ U[1:]
    eigs, _ = la.eig(phi)
    res = -np.angle(eigs[0])
    res = (1 - np.cos(res/2)) / 2
    return res, queries, depth
