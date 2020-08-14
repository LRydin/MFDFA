import numpy as np

import sys
sys.path.append("../")
from MFDFA import MFDFA

def test_MFDFA():
    for N in [10000, 1000000]:
        for q_list in [1, 2, 6, 21]:
            for order in [1,2,3,4]:

                X = np.random.normal(size = N, loc = 0)
                q = np.linspace(-10, 10, q_list)

                lag = np.unique(
                      np.logspace(
                      0, np.log10(X.size // 4), 25
                      ).astype(int) + 1 + order
                      )

                lag, dfa = MFDFA(X, lag = lag, q = q, order = 1)

                assert dfa.ndim == 2, "Output is not 2 dimensional"
                assert dfa.shape[1] <= q.shape[0], "Output shape mismatch"

                lag, dfa, dfa_std = MFDFA(X, lag = lag, q = q, order = 1,
                  modified=True, stat=True)

                assert dfa.ndim == 2, "Output is not 2 dimensional"
                assert dfa.shape[1] <= q.shape[0], "Output shape mismatch"

                lag, dfa = MFDFA(X, lag = lag, q = q, order = 1,
                  modified=True, stat=False, extensions = 'eDFA')

                assert dfa.ndim == 2, "Output is not 2 dimensional"
                assert dfa.shape[1] <= q.shape[0], "Output shape mismatch"
