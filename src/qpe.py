import numpy as np


def qpe(p, M, N_s, rng):
    M_true = M//2
    depth = 2 * M_true - 1
    queries = (2 * M_true - 1) * N_s
    omega = np.arcsin(np.sqrt(p))
    eigen = np.exp(-2j * np.arange(M_true) * omega) / np.sqrt(M_true)
    sig = np.fft.ifft(eigen, norm="ortho")
    prob = np.abs(sig) ** 2
    prob = prob + np.concat((prob[0:1], prob[-1:0:-1]))
    prob /= 2
    samp = rng.multinomial(np.ones(N_s, dtype=np.int16), prob).argmax(-1)
    samp = np.sin(samp / M_true * np.pi) ** 2
    return np.median(samp), queries, depth
