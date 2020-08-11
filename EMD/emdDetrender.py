from PyEMD import EMD
import numpy as np

def getDetrendedtimeseries(timeseries: np.ndarray, IMFIndicesToBeSubtracted: list):
    '''
        @param {np.ndarray} timeseries : A 1d numpy array of real numbers on which the EMD analysis is to be perfomed
        
        @param {list} IMFIndicesToBeSubtracted : A python list (i.e. array) of integers indicating the indices of the IMFs to be subtracted
        
        @return {np.ndarray} detrendedTimeseries : The detrended timeseries based on user's choice of IMFs

        The function calculates the Intrinsic Mode Functions (IMFs) of a given timeseries,
        subtracts the user-chosen IMFs for detrending, and returns the detrended timeseries.
    '''

    IMFs = getIMFs(timeseries)
    tobeSubtracted = timeseries*0
    for i in range(len(IMFIndicesToBeSubtracted)):
        tobeSubtracted = np.add(tobeSubtracted, IMFs[i])
            
    detrendedTimeseries = timeseries - tobeSubtracted
    return detrendedTimeseries

def getIMFs(timeseries):
    emd = EMD()
    IMFs = emd(timeseries)
    return IMFs
