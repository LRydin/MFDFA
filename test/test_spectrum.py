import numpy as np

import sys
sys.path.append("../")

import matplotlib.pyplot as plt
from scipy.stats import levy_stable

from MFDFA import MFDFA
from MFDFA import singularity_spectrum, singularity_spectrum_plot
from MFDFA import scaling_exponents, scaling_exponents_plot
from MFDFA import hurst_exponents, hurst_exponents_plot


def test_spectrum():
    for N in [1000, 10000]:
        for q_list in [6, 12, 21]:

            alpha = 1.5
            X = levy_stable.rvs(alpha, beta=0, size=N)

            q = np.linspace(-10, 10, q_list)
            q = q[q!=0.0]

            print(q)
            lag = np.unique(
                  np.logspace(
                  0, np.log10(X.size // 4), 25
                  ).astype(int) + 1
                )

            lag, dfa = MFDFA(X, lag = lag, q = q, order = 1)

            hq, Dq  = singularity_spectrum(lag, dfa, q = q)
            _ = singularity_spectrum_plot(hq, Dq);
            assert hq.shape[0] == Dq.shape[0], "Output shape mismatch"
            assert hq.shape[0] == q.shape[0], "Output shape mismatch"

            tau = scaling_exponents(lag, dfa, q = q)
            _ = scaling_exponents_plot(q, tau);
            assert tau.shape[0] == q.shape[0], "Output shape mismatch"

            hq = hurst_exponents(lag, dfa, q = q)
            _ = hurst_exponents_plot(q, hq);
            assert hq.shape[0] == q.shape[0], "Output shape mismatch"