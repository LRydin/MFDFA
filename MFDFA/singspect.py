## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

import numpy as np

from numpy.polynomial.polynomial import polyfit

import matplotlib.pyplot as plt

def singularity_spectrum(lag, mfdfa_data, q, lim = [None, None]):
    """
    Extract the slopes of the fluctuation function to futher obtain the
    the singularity strength ``hq`` (or Hölder exponents) and singularity
    spectrum ``Dq`` (or fractal dimension).

    Parameters
    ----------
    lag: np.ndarray of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.ndarray
        Fractal exponents used. Must be more than 2 points.

    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.

    Returns
    -------
    hq: np.ndarray
        Singularity strength ``hq``. The width of this function indicates the
        strength of the multifractality. A width of ``max(hq) - min(hq) ≈ 0``
        means the data is monofractal.

        Dq: np.ndarray
        Singularity spectrum ``Dq``. The location of the maximum of ``Dq`` (with
         ``hq`` as the abscissa) should be 1 and indicates the most prominent
         exponent in the data.

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.
    """

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype = float)

    slopes = np.zeros(len(q))

    # Find slopes of each q-power
    for i in range(len(q)):
        slopes[i] = polyfit(np.log(lag[lim[0]:lim[1]]),
                            np.log(mfdfa_data[lim[0]:lim[1],i]),
                            1)[1]

    # Calculate tau
    tau = _tau(slopes, q)

    # Calculate hq, which needs tau
    hq = _hq(tau, q)

    # Calculate Dq, which needs tau and hq
    Dq = _Dq(tau, hq, q)

    return hq, Dq

def _tau(slopes, q):
    """
    Calculate the multifractal scaling exponents ``tau``.
    """
    return ( q * slopes ) - 1

def _hq(tau, q):
    """
    Calculate the singularity strength or Hölder exponents ``hq``.
    """
    return ( np.gradient(tau) / np.gradient(q) )

def _Dq(tau, hq, q):
    """
    Calculate the singularity spectrum or fractal dimension ``Dq``.
    """
    return q * hq - tau
