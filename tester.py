# %% codecell
import numpy as np
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
t_final = 2000
delta_t = 0.001

# The parameters θ and σ
theta = 1
sigma = 0.25

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
plt.plot(time[:1000], X[:1000])


# %% MFDFA
# Select the segment lengths s, denoted lag here
lag = np.unique(np.logspace(0.5, np.log10(X.size // 100), 50, dtype=int))

# q-variations to calculate
q = 2

# dfa records the fluctuation function using the EMD as a detrending mechanims.
lag, dfa = MFDFA(X, lag, q = q)

# %% Plots
# Visualise the results in a log-log plot
plt.loglog(lag, dfa, 'o-');
plt.loglog(lag, np.var(dw)*lag**(H+1), '--', color='black');

# %%
# Extract the slopes and compare with H + 1, i.e., 1.7.
H_hat = np.polyfit(np.log(lag)[4:20],np.log(dfa[4:20]),1)[0]

print('Estimated H = '+'{:.3f}'.format(H_hat[0]))
