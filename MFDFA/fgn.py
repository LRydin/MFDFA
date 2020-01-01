## This is based the work of Christopher Flynn fbm in
# https://github.com/crflynn/fbm and Davies, Robert B., and D. S. Harte. “Tests
# for Hurst effect.” Biometrika 74, no. 1 (1987): 95-101.

import numpy as np

def fgn(N: int, H: float) -> np.ndarray:
    """
    Generates fractional Gaussian noise with a Hurst index H in (0,1). If
    H = 1/2 this is simply Gaussian noise.
    The current method employed is the Davies–Harte method, which fails for
    H ≈ 0. A Cholesky decomposition method and the Hosking’s method will be
    implemented in later versions.

    Parameters
    ----------
    N: int
        Size of fractional Gaussian noise to generate.

    H: float
        Hurst exponent H in (0,1).

    Returns
    -------
    f: np.ndarray
        A array of size N of fractional Gaussian noise with a Hurst index H.
    """

    # Asserts
    assert isinstance(N, int), "Size must be an integer number"
    assert isinstance(H, float), "Hurst index must be a float in (0,1)"

    # Generate linspace
    k = np.linspace(0,N-1,N)

    # Correlation function
    cor = 0.5*(abs(k - 1)**(2*H) - 2*abs(k)**(2*H) + abs(k + 1)**(2*H))

    # Eigenvalues of the correlation function
    eigenvals = np.sqrt(
                  np.fft.fft(
                    np.concatenate([cor[:],0,cor[1:][::-1]],axis = None).real
                  )
                )

    # Two normal distributed noises to be convoluted
    gn = np.random.normal(0.0, 1.0, N)
    gn2 = np.random.normal(0.0, 1.0, N)

    # This is the Davies–Harte method
    w = np.concatenate(
        [(eigenvals[0]   / np.sqrt(2*N)) * gn[0],
         (eigenvals[1:N] / np.sqrt(4*N)) *(gn[1:] + 1j*gn2[1:]),
         (eigenvals[N]   / np.sqrt(2*N)) * gn2[0],
         (eigenvals[N+1:]/ np.sqrt(4*N)) *(gn[1:][::-1] - 1j*gn2[1:][::-1])
        ],
        axis = None)

    # Perform fft. Only first N entry are useful
    f = np.fft.fft(w).real[:N] * ( (1.0 / N) ** H )

    # TODO: Check for further speed-ups
    # TODO: Generalise for several outputs. Should be straightforward.
    # TODO: Implement the Cholesky decomposition method
    # TODO: Implement the Hosking’s method
    return f
