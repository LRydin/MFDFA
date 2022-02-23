import numpy as np

import sys
sys.path.append("../")
from MFDFA import RS

def test_RS():
    for N in [1000, 10000]:

        X = np.random.normal(size = N, loc = 0)

        lag = np.unique(
              np.logspace(
              0, np.log10(X.size // 4), 50
              ).astype(int) + 1
            )

        # Testing wrongly shaped lag
        try:
            RS(X, lag = lag.reshape(2,-1))
        except Exception:
            pass

        # Testing lag larger than timeseries
        try:
            MFDFA(X, lag = [3,4,X.size+1])
        except Exception:
            pass

        # Testing wrongly shaped array
        try:
            RS(np.stack((X,X), axis=1), lag = lag)
        except Exception:
            pass

        # Testing simple FA (order = 0)
        lag, rs = RS(X, lag = lag)

        assert rs.ndim == 1, "Output is not 1 dimensional"
