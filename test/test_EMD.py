import numpy as np

import sys
sys.path.append("../")
from MFDFA import detrendedtimeseries, IMFs

def test_EMD():
    for N in [1000, 10000]:

        X = np.cumsum(np.random.normal(size = N, loc = 0))

        Y = detrendedtimeseries(X, [0])

        assert X.shape == Y.shape, "Detrended data shape doesn't match original"

        IMF = IMFs(X)

        assert IMF.shape[1] == Y.shape[0], "IMFs shape doesn't match timeseres"
