import numpy as np

import sys
sys.path.append("../")

import matplotlib.pyplot as plt

from MFDFA import MFDFA
from MFDFA import singspect

def test_spectrum():
    for N in [1000, 10000]:
        for q_list in [6, 12, 21, 41]:

            X = np.cumsum(np.random.normal(0, 5, size=N))

            q = np.linspace(-10, 10, q_list)
            q = q[q!=0.0]

            lag = np.unique(
                  np.logspace(
                  0, np.log10(X.size // 4), 55
                  ).astype(int) + 3
                )

            lag, dfa = MFDFA(X, lag=lag, q=q, order=1)

            alpha, f  = singspect.singularity_spectrum(lag, dfa, q=q)

            _ = singspect.singularity_spectrum(lag, dfa, q=q, lim=[None, None])
            _ = singspect.singularity_spectrum_plot(alpha, f)

            assert alpha.shape[0] == f.shape[0], "Output shape mismatch"
            assert alpha.shape[0] == q.shape[0], "Output shape mismatch"

            q, tau = singspect.scaling_exponents(lag, dfa, q=q)
            _ = singspect.scaling_exponents_plot(q, tau)
            assert tau.shape[0] == q.shape[0], "Output shape mismatch"

            q, hq = singspect.hurst_exponents(lag, dfa, q=q)
            _ = singspect.hurst_exponents_plot(q, hq)
            assert hq.shape[0] == q.shape[0], "Output shape mismatch"

            try:
                singspect._slopes(lag, dfa, q[0:3])
            except Exception:
                pass
