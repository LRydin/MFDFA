# %% codecell
import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

from MFDFA import MFDFA
from MFDFA import fgn


# %% Lets take a fractional Ornstein–Uhlenbeck process Xₜ with a time-dependent
# diffusion or volatility σₜ, some drift of mean reverting term θ(Xₜ), and
# fractional Gaussian noise with Hurst exponent H
#
#                           dXₜ = - θ(Xₜ)Xₜdt + σₜdBᴴₜ.

# Generate some path with a simple Euler–Maruyama integrator

# Integration time and time sampling
t_final = 100
delta_t = 0.001

# The parameters θ and σ
theta = 1
sigma = 0.5

# The time array of the trajectory
time = np.arange(0, t_final, delta_t)

# Initialise the array y
X = np.zeros(time.size)

# Lets use a positively correlated noise H > 1/2
H = 0.7

# Generate the fractional Gaussian noise
dw = (t_final ** H) * fgn(time.size, H = H)

# Integrate the process
for i in range(1,time.size):
    X[i] = X[i-1] - theta*X[i-1]*delta_t + sigma*dw[i]

# Plot the path
plt.plot(time, X)

# %% MFDFA
# Select the segment lengths s, denoted lag here
lag = np.unique(np.logspace(0, np.log10(X.size // 100), 25).astype(int)+1)

# q-variations to calculate
q = np.linspace(1,10,10)

# dfa records the fluctuation function, order = 1 is the order of the polynomial
# fittings used, i.e., DFA1 in this case
lag, dfa = MFDFA(X, lag, q = q, order = 1)

# %% Plots
# Visualise the results in a log-log plot
plt.loglog(lag, dfa, '.');

# %%
# Extract the slopes and compare with H + 1, i.e., 1.7.
polyfit(np.log(lag[:]),np.log(dfa[:]),1)[1]
