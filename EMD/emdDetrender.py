# This is based on Dawid Laszuk's pyEMD functions in 'Python implementation of
# Empirical Mode Decomposition algorithm', https://github.com/laszukdawid/PyEMD,
# licenced under the Apache 2.0 Licencing.

from PyEMD import EMD
import numpy as np

def getDetrendedtimeseries(timeseries: np.ndarray, IMFmodes: list):
    """
        The function calculates the Intrinsic Mode Functions (IMFs) of a given
        timeseries, subtracts the user-chosen IMFs for detrending, and returns
        the detrended timeseries.

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
    """

    # Obtain Intrinsic Mode Functions (IMFs) using pyEMD
    IMFs = getIMFs(timeseries)

    tobeSubtracted = timeseries*0
    for i in range(len(IMFmodes)):
        tobeSubtracted = np.add(tobeSubtracted, IMFs[i])

    detrendedTimeseries = timeseries - tobeSubtracted
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
            The Intrinsic Mode Functions (IMFs) of the Empirical Mode
            Decomposition.
    """
    # Initiate pyEMD's EMD function
    emd = EMD()

    # Obtain the Intrinsic Mode Functions (IMFs)
    IMFs = emd(timeseries)

    return IMFs
