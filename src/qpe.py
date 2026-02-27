import numpy as np


def qpe(p, M, N_s, rng):
    """Estimate amplitude via a QPE-inspired phase histogram simulation.

    Parameters
    ----------
    p : float
        True Bernoulli success probability used by simulation.
    M : int
        Total phase-grid parameter; effective register size is ``M // 2``.
    N_s : int
        Number of independent histogram samples.
    rng : numpy.random.Generator
        Random number generator for multinomial sampling.

    Returns
    -------
    tuple[float, int, int]
        ``(estimate, queries, depth)`` where ``estimate`` is the median-amplitude
        statistic, ``queries`` is total query cost, and ``depth`` is max depth.
    """
    M_true = M // 2
    depth = 2 * M_true - 1
    queries = (2 * M_true - 1) * N_s
    omega = np.arcsin(np.sqrt(p))
    # Ideal phase-estimation signal in Fourier basis.
    eigen = np.exp(-2j * np.arange(M_true) * omega) / np.sqrt(M_true)
    sig = np.fft.ifft(eigen, norm="ortho")
    prob = np.abs(sig) ** 2
    # Symmetrize around zero/negative frequencies.
    prob = prob + np.concat((prob[0:1], prob[-1:0:-1]))
    prob /= 2
    samp = rng.multinomial(np.ones(N_s, dtype=np.int16), prob).argmax(-1)
    samp = np.sin(samp / M_true * np.pi) ** 2
    return np.median(samp), queries, depth
