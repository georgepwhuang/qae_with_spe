import numpy as np
from collections import UserList


def generate_csae_positions(q):
    """Generate and save CSAE index caches for one schedule size.

    Parameters
    ----------
    q : int
        CSAE schedule parameter controlling number of coarray iterations.

    Returns
    -------
    None
        Writes index arrays to ``./cache/csae_pos_q={q}.npy``.
    """
    o_arr = np.concat((np.zeros(1), 2 ** np.arange(2 * q)))
    arr = np.copy(o_arr)
    with open(f"./cache/csae_pos_q={q}.npy", "wb") as f:
        # Save unique-index maps for each iterative coarray expansion step.
        for _ in range(q):
            arr = o_arr - np.tile(arr, (len(o_arr), 1)).T
            arr, indices = np.unique(arr, return_index=True)
            np.save(f, indices)
        # Save the final selection window used by the ESPRIT stage.
        final_index = np.nonzero(np.abs(arr - 2 ** (2 * q - 2)) <= 2 ** (2 * q - 2))
        np.save(f, final_index)


def load_csae_positions(q):
    """Load precomputed CSAE index caches from disk.

    Parameters
    ----------
    q : int
        CSAE schedule parameter used when cache file was generated.

    Returns
    -------
    list[numpy.ndarray]
        Sequence of per-iteration pruning indices plus final selection indices.
    """
    output = []
    with open(f"./cache/csae_pos_q={q}.npy", "rb") as f:
        for _ in range(q):
            output.append(np.load(f))
        output.append(np.load(f))
    return output


class CSAECache(UserList):
    """List-like container holding loaded CSAE cache arrays.

    Parameters
    ----------
    q : int
        CSAE schedule parameter identifying which cache file to load.
    """
    def __init__(self, q):
        """Initialize cache container from persisted indices.

        Parameters
        ----------
        q : int
            CSAE schedule parameter identifying which cache file to load.
        """
        positions = load_csae_positions(q)
        super().__init__(positions)


if __name__ == "__main__":
    for i in range(0, 9):
        generate_csae_positions(i)
