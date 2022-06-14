# created by Leonardo Rydin Gorjão. Most python libraries are standard (e.g. via
# Anaconda). If TeX is not present in the system comment out lines 12 to 15.

import numpy as np

# to install MFDFA just run pip install MFDFA
from MFDFA import MFDFA
from MFDFA import fgn

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pgf.texsystem'] = 'pdflatex'
matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 18,
    'axes.labelsize': 20,'axes.titlesize': 24, 'figure.titlesize' : 28})
matplotlib.rcParams['text.usetex'] = True

colours = ['#1b9e77','#d95f02','#7570b3']

################################## fOU process #################################

# %% Lets take a fractional Ornstein–Uhlenbeck process Xₜ with a time-dependent
# diffusion or volatility σₜ, some drift of mean reverting term θ(Xₜ), and
# fractional Gaussian noise with Hurst exponent H
#
#                           dXₜ = - θ(Xₜ)Xₜdt + σₜdBᴴₜ.

# integration time and time sampling
t_final = 100
delta_t = .001

# The parameters θ and σ
theta = 1
sigma = .5

# The time array of the trajectory
time = np.arange(0, t_final, delta_t)

# Initialise the array y
X1 = np.zeros(time.size)
X2 = np.zeros(time.size)
X3 = np.zeros(time.size)

# Generate the fractional Gaussian noise
dw1 = (t_final ** .3) * fgn(time.size, H=.3)
dw2 = (t_final ** .5) * fgn(time.size, H=.5)
dw3 = (t_final ** .7) * fgn(time.size, H=.7)

# Integrate the process
for i in range(1, time.size):
    X1[i] = X1[i-1] - theta*X1[i-1] * delta_t + sigma * dw1[i]
    X2[i] = X2[i-1] - theta*X2[i-1] * delta_t + sigma * dw2[i]
    X3[i] = X3[i-1] - theta*X3[i-1] * delta_t + sigma * dw3[i]

# %% MFDFA
# Lag s from 3 to 1000 datapoints
lag = np.unique(np.logspace(.6, 3, 118).astype(int))

# q power variations, removing the 0 power
q_list = np.linspace(-10, 10, 41)
q_list = q_list[q_list!=.0]

lag, dfa1 = MFDFA(X1, lag, q=q_list, order=1)
lag, dfa2 = MFDFA(X2, lag, q=q_list, order=1)
lag, dfa3 = MFDFA(X3, lag, q=q_list, order=1)

# %% ################################## FIG01 ##################################
fig, ax = plt.subplots(1,2, figsize=(16,4))
axi1 = fig.add_axes([.31, .2, .18, .38])

ax[0].loglog(lag, dfa1[...,23],'o', markersize=6, color = colours[0],
    markerfacecolor='none', label=r'$H=0.3, q=2$')
ax[0].loglog(lag, dfa2[...,23],'s', markersize=6, color = colours[1],
    markerfacecolor='none', label=r'$H=0.5, q=2$')
ax[0].loglog(lag, dfa3[...,23],'D', markersize=6, color = colours[2],
    markerfacecolor='none', label=r'$H=0.7, q=2$')

ax[0].set_ylim([5e-4,4e1])

axi1.loglog(lag[10:], .02 * lag[10:]**1.3,'--', color = 'black', alpha= .5)
axi1.loglog(lag[10:], .0002 * lag[10:]**1.3,'--', color = 'black', alpha= .5)
axi1.loglog(lag[10::2], dfa1[10::2,[0]],  '^', markersize=6, color=colours[1],
    markerfacecolor='none', label=r'$q=-\!\!10$')
axi1.loglog(lag[10::2], dfa1[10::2,[17]], 'v', markersize=6, color=colours[1],
    markerfacecolor='none', label=r'$q=-\!\!2$')
axi1.loglog(lag[10::2], dfa1[10::2,[23]], '<', markersize=6, color=colours[1],
    markerfacecolor='none', label=r'$q=2$')
axi1.loglog(lag[10::2], dfa1[10::2,[39]], '>', markersize=6, color=colours[1],
    markerfacecolor='none', label=r'$q=10$')

axi1.set_xticks([])
fig.text(.315, .49, r'$H=0.3$', fontsize=20)
fig.text(.365, .225, r'$q=-10,-2,2,10$', fontsize=18)

slopes1 = np.polynomial.polynomial.polyfit(np.log(lag)[5:45],
    np.log(dfa1)[5:45],1)[1]-1
slopes2 = np.polynomial.polynomial.polyfit(np.log(lag)[5:45],
    np.log(dfa2)[5:45],1)[1]-1
slopes3 = np.polynomial.polynomial.polyfit(np.log(lag)[5:45],
    np.log(dfa3)[5:45],1)[1]-1

ax[1].plot(q_list, q_list * slopes1 - 1, 'o', markersize=6, color=colours[0],
    markerfacecolor='none', label=r'$H=0.3$')
ax[1].plot(q_list, q_list * slopes2 - 1, 's', markersize=6, color=colours[1],
    markerfacecolor='none', label=r'$H=0.5$')
ax[1].plot(q_list, q_list * slopes3 - 1, 'D', markersize=6, color=colours[2],
    markerfacecolor='none', label=r'$H=0.7$')

ax[1].plot(q_list, q_list * .3 - 1, '-', color='black', alpha=.7)
ax[1].plot(q_list, q_list * .5 - 1, '-', color='black', alpha=.7)
ax[1].plot(q_list, q_list * .7 - 1, '-', color='black', alpha=.7)

axi2 = fig.add_axes([.802, .2, .18, .38])
axi2.plot(q_list, slopes1, 'o', markersize=6, color=colours[0],
    markerfacecolor='none', label=r'$H=0.3$')
axi2.plot(q_list, slopes2, 's', markersize=6, color=colours[1],
    markerfacecolor='none', label=r'$H=0.5$')
axi2.plot(q_list, slopes3, 'D', markersize=6, color=colours[2],
    markerfacecolor='none', label=r'$H=0.7$')

axi2.plot(q_list, np.ones_like(q_list) * .3, '-', color='black', alpha=.7)
axi2.plot(q_list, np.ones_like(q_list) * .5, '-', color='black', alpha=.7)
axi2.plot(q_list, np.ones_like(q_list) * .7, '-', color='black', alpha=.7)

axi2.set_ylim([.1,.9])
axi2.set_yticks([.3,.5,.7])
axi2.set_xticks([])
axi2.set_ylabel(r'$h(q)$',labelpad=3,fontsize=24)

ax[0].set_ylabel(r'$F_q(s)$',labelpad=3,fontsize=24)
ax[0].set_xlabel(r'segment size $s$',labelpad=3,fontsize=24)

ax[1].set_ylabel(r'$\tau(q)$',labelpad=3,fontsize=24)
ax[1].set_xlabel(r'$q$',labelpad=3,fontsize=24)

ax[0].legend(fontsize=20, loc=2, handlelength=1.1, columnspacing=.6,
    handletextpad=.2)
ax[1].legend(fontsize=20, loc=2, handlelength=1.1, columnspacing=.6,
    handletextpad=.2)

fig.text(.005, .92, r'a)', fontsize=28)
fig.text(.505, .92, r'b)', fontsize=28)

fig.subplots_adjust(left=.07, bottom=.17, right=.99, top=.99, hspace=.06,
    wspace=.15)

# fig.savefig('fig01.pdf', trasparent=True)
