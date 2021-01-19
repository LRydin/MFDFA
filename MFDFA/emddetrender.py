# This is based on Dawid Laszuk's PyEMD functions in 'Python implementation of
# Empirical Mode Decomposition algorithm', https://github.com/laszukdawid/PyEMD,
# licenced under the Apache 2.0 Licencing.

# from PyEMD import EMD
import numpy as np

# Import of PyEMD is called inside the function

def detrendedtimeseries(timeseries: np.ndarray, modes: list):
    """
    The function calculates the Intrinsic Mode Functions (IMFs) of a given
    timeseries, subtracts the user-chosen IMFs for detrending, and returns the
    detrended timeseries. Based on based on Dawid Laszuk's PyEMD found at
    https://github.com/laszukdawid/PyEMD

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length ``N``.

    modes: list
        List of integers indicating the indices of the IMFs to be
        subtracted/detrended from the ``timeseries``.

    Returns
    -------
    detrendedTimeseries: np.ndarray
        Detrended 1-dimensional ``timeseries``.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [Huang1998] N. E. Huang, Z. Shen, S. R. Long, M. C. Wu, H. H. Shih, Q.
        Zheng, N.-C. Yen, C. C. Tung, and H. H. Liu, "The empirical mode
        decomposition and the Hilbert spectrum for non-linear and non stationary
        time series analysis", Proc. Royal Soc. London A, Vol. 454, pp. 903-995,
        1998.
    .. [Rilling2003] G. Rilling, P. Flandrin, and P. Goncalves, "On Empirical
        Mode Decomposition and its algorithms", IEEE-EURASIP Workshop on
        Nonlinear Signal and Image Processing NSIP-03, Grado (I), June 2003.
    """

    # Obtain Intrinsic Mode Functions (IMFs) using pyEMD
    IMF = IMFs(timeseries)

    # Subtract the selected IMFs 'modes' from the timeseries
    detrendedTimeseries = timeseries - np.sum(IMF[modes, :], axis = 0)

    return detrendedTimeseries


def IMFs(timeseries: np.ndarray):
    """
    Extract the Intrinsic Mode Functions (IMFs) of a given timeseries.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length ``N``.

    Notes
    -----
    .. versionadded:: 0.3

    Returns
    -------
    IMFs: np.ndarray
        The Intrinsic Mode Functions (IMFs) of the Empirical Mode Decomposition.
        These are of shape ``(..., timeseries.size)``, with the first dimension
        varying depending on the data. Last entry is the residuals.
    """

    # Check if EMD-signal is installed
    missing_library()
    from PyEMD import EMD

    # Initiate pyEMD's EMD function
    emd = EMD()

    # Obtain the Intrinsic Mode Functions (IMFs)
    IMFs = emd(timeseries)

    # Returns the IMFs as a (..., timeseries.size) numpy array.
    return IMFs


def missing_library():
    try:
        import PyEMD.EMD as _EMD
    except ImportError:
        raise ImportError(("PyEMD is required to do Empirical Mode "
                           "decomposition. Please install PyEMD with 'pip "
                           "install EMD-signal'.")
                         )
