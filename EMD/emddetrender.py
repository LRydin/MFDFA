# This is based on Dawid Laszuk's PyEMD functions in 'Python implementation of
# Empirical Mode Decomposition algorithm', https://github.com/laszukdawid/PyEMD,
# licenced under the Apache 2.0 Licencing.

from PyEMD import EMD
import numpy as np

def detrendedtimeseries(timeseries: np.ndarray, IMFmodes: list):
    """
    The function calculates the Intrinsic Mode Functions (IMFs) of a given
    timeseries, subtracts the user-chosen IMFs for detrending, and returns the
    detrended timeseries. Based on based on Dawid Laszuk's PyEMD found at
    https://github.com/laszukdawid/PyEMD

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length ``N``.

    IMFmodes: list
        List of integers indicating the indices of the IMFs to be
        subtracted/detrended from the ``timeseries``.

    Returns
    -------
    detrendedTimeseries: np.ndarray
        Detrended 1-dimensional ``timeseries``.




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
    IMFs = getIMFs(timeseries)

    # tobeSubtracted = timeseries*0
    # for i in range(len(IMFmodes)):
    #     tobeSubtracted = np.add(tobeSubtracted, IMFs[i])

    detrendedTimeseries = timeseries - np.sum(IMFs[IMFmodes, :], axis = 0)

    return detrendedTimeseries

def getIMFs(timeseries: np.ndarray):
    """
    Extract the Intrinsic Mode Functions (IMFs) of a given timeseries.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries of length ``N``.

    Returns
    -------
    IMFs: np.ndarray
        The Intrinsic Mode Functions (IMFs) of the Empirical Mode Decomposition.
    """
    # Initiate pyEMD's EMD function
    emd = EMD()

    # Obtain the Intrinsic Mode Functions (IMFs)
    IMFs = emd(timeseries)

    return IMFs
