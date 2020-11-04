## This is based the work of Christopher Flynn fbm in
# https://github.com/crflynn/fbm and Muniandy, S. V., and Lim, S. C., "Modeling
# of locally self-similar processes using multifractional Brownian motion of
# Riemann–Liouville type." Physical Review E 63, no. 4 (2001): 046104.

import numpy as np
from math import gamma

def mgn(t: np.ndarray, H: callable) -> np.ndarray:
    """
    Generates multifractional Gaussian noise with a time-dependent Hurst index
    H(t) in (0,1).
    The method employed is the Muniandy–Lim method, for Riemann–Liouville type
    of Brownian motion.

    Parameters
    ----------
    t: np.ndarray
        Time array of size ``N``.

    H: clalable
        Hurst exponent function H(t) in (0,1). This should be a function of the
        value the Hurst coefficient takes over time.

    Returns
    -------
    mbm: np.ndarray
        A array of size N of multifractional Gaussian noise with varying Hurst
        over time.

    References
    ----------
    .. [Muniandy2001] S. V. Muniandy and S. C. Lim, "Modeling of locally
        self-similar processes using multifractional Brownian motion of
        Riemann–Liouville type." Physical Review E 63(4), 046104, 2001
    """

    # Asserts
    assert callable(H) == True, "Hurst must be a function of time"

    # Extract increment of time
    dt = t[1] - t[0]

    # Size of arrays
    N = t.size


    # Generate a Gaussian noise
    gn = np.random.normal(0.0, 1.0, N) * np.sqrt(dt)

    # Add additional endpoint to time
    t = np.append(t, t[-1] + dt)

    # Preallocated mbm
    mbm = np.zeros_like(t)

    # Generate local arrays for speedup
    h = 2 * H(t)
    norm = 1 / ((h * dt) ** (0.5))

    for k in range(1, N + 1):
        w = (
            ((t[1:k] ** h[k] - t[:k-1] ** h[k]) ** 0.5) * norm[k]
            / gamma((h[k] / 2.) + 0.5)
        )

        mbm[k] = np.sum(gn[:k-1] *  w[:k][::-1])

    # TODO: Check for further speed-ups
    # TODO: Generalise for several outputs. Should be straightforward.
    return np.diff(mbm)
