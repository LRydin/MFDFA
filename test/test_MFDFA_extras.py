import numpy as np

import sys
sys.path.append("../")
from MFDFA import MFDFA

def test_MFDFA():
    for N in [1000, 10000]:
        for q_list in [1, 2, 6, 21]:

            X = np.random.normal(size = N, loc = 0)
            q = np.linspace(-10, 10, q_list)

            lag = np.unique(
                  np.logspace(
                  0, np.log10(X.size // 4), 25
                  ).astype(int) + 1
                )

            lag, dfa = MFDFA(X, lag = lag, q = q, order = 0,
              modified = True, stat = False, extensions = {'EMD': [0]})

            assert dfa.ndim == 2, "Output is not 2 dimensional"
            assert dfa.shape[1] <= q.shape[0], "Output shape mismatch"

            lag, dfa, edfa = MFDFA(X, lag = lag, q = q, order = 1,
              modified = True, stat = False,
              extensions = {'eDFA':True, 'EMD': [0]})

            assert dfa.ndim == 2, "Output is not 2 dimensional"
            assert edfa.ndim == 2, "Output is not 2 dimensional"
            assert dfa.shape[1] <= q.shape[0], "Output shape mismatch"
