import numpy as np
from collections import UserList


def generate_csae_positions(q):
    o_arr = np.concat((np.zeros(1), 2 ** np.arange(2 * q)))
    arr = np.copy(o_arr)
    with open(f"./cache/csae_pos_q={q}.npy", "wb") as f:
        for _ in range(q):
            arr = o_arr - np.tile(arr, (len(o_arr), 1)).T
            arr, indices = np.unique(arr, return_index=True)
            np.save(f, indices)
        final_index = np.nonzero(np.abs(arr - 2 ** (2 * q - 2)) <= 2 ** (2 * q - 2))
        np.save(f, final_index)


def load_csae_positions(q):
    output = []
    with open(f"./cache/csae_pos_q={q}.npy", "rb") as f:
        for _ in range(q):
            output.append(np.load(f))
        output.append(np.load(f))
    return output


class CSAECache(UserList):
    def __init__(self, q):
        positions = load_csae_positions(q)
        super().__init__(positions)


if __name__ == "__main__":
    for i in range(0, 9):
        generate_csae_positions(i)
