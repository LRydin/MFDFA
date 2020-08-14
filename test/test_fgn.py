import numpy as np

import sys
sys.path.append("../")
from MFDFA import fgn

def test_fgn():
    for H in [0.3, 0.5, 0.7]:
        for N in [1000, 10000]:

            noise = fgn(N, H = H)

            assert noise.size == N, "Generated noise size not N"
