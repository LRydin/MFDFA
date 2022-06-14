# created by Leonardo Rydin Gorjão. Most python libraries are standard (e.g. via
# Anaconda). If TeX is not present in the system comment out lines 13 to 16.

import numpy as np
from scipy.stats import levy_stable

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

# %% ######################### Lévy distributions ##############################

# integration time and time sampling
t_final = 1000
delta_t = .001

# The time array of the trajectory
time = np.arange(0, t_final, delta_t)

alpha = 1.75
X4 = levy_stable.rvs(alpha=alpha, beta=0, size=time.size)

alpha = 1.25
X5 = levy_stable.rvs(alpha=alpha, beta=0, size=time.size)

alpha = .75
X6 = levy_stable.rvs(alpha=alpha, beta=0, size=time.size)

# %%
lag = np.unique(np.logspace(0.6, 3, 118).astype(int))

q_list = np.linspace(-10,10,41)
q_list = q_list[q_list!=0.0]
lag, dfa4 =  MFDFA(X4, lag, q=q_list, order=3)
lag, dfa5 = MFDFA(X5, lag, q=q_list, order=3)
lag, dfa6 = MFDFA(X6, lag, q=q_list, order=3)

# %% ################################## FIG02 ##################################
fig, ax = plt.subplots(1,2, figsize=(16,4))

ax[0].loglog(lag[9::2], dfa5[9::2,[0]], '^', markersize=6,
    markerfacecolor='none', color=colours[1], label=r'$q=-10$')
ax[0].loglog(lag[9::2], dfa5[9::2,[23]],'<', markersize=6,
    markerfacecolor='none', color=colours[1], label=r'$q=2$')
ax[0].loglog(lag[9::2], dfa5[9::2,[10]],'D', markersize=6,
    markerfacecolor='none', color=colours[1], label=r'$q=-5$')
ax[0].loglog(lag[9::2], dfa5[9::2,[29]],'H', markersize=6,
    markerfacecolor='none', color=colours[1], label=r'$q=5$')
ax[0].loglog(lag[9::2], dfa5[9::2,[17]],'v', markersize=6,
    markerfacecolor='none', color=colours[1], label=r'$q=-2$')
ax[0].loglog(lag[9::2], dfa5[9::2,[39]],'>', markersize=6,
    markerfacecolor='none', color=colours[1], label=r'$q=10$')

ax[0].loglog(lag[9::2], 0.03 * lag[9::2]**(1/1.25),'--', color='black')
ax[0].loglog(lag[9::2], 4e4 * lag[9::2]**(1/10),'--', color='black')


ax[0].set_ylabel(r'$F_q(s)$', labelpad=7, fontsize=24)
ax[0].set_xlabel(r'segment size $s$', labelpad=3, fontsize=24)
ax[0].set_ylim([9e-2, 2e5])
fig.text(.18, .22, r'$\alpha = 1.25$', fontsize=22)

slopes4 = np.polynomial.polynomial.polyfit(np.log(lag)[25:75],
    np.log(dfa4)[25:75],1)[1]
slopes5 = np.polynomial.polynomial.polyfit(np.log(lag)[25:75],
    np.log(dfa5)[25:75],1)[1]
slopes6 = np.polynomial.polynomial.polyfit(np.log(lag)[25:75],
    np.log(dfa6)[25:75],1)[1]

ax[1].plot(q_list, slopes4, 'o', markersize=9, markerfacecolor='none',
    color=colours[0], label=r'$\alpha=1.75$')
ax[1].plot(q_list, slopes5, 's', markersize=9, markerfacecolor='none',
    color=colours[1], label=r'$\alpha=1.25$')
ax[1].plot(q_list, slopes6, 'D', markersize=9, markerfacecolor='none',
    color=colours[2], label=r'$\alpha=0.75$')

ax[1].plot(q_list[:20], np.ones(q_list[:20].size)*(1/1.75),'-', color='black',
    alpha=.7)
ax[1].plot(q_list[:20], np.ones(q_list[:20].size)*(1/1.25),'-', color='black',
    alpha=.7)
ax[1].plot(q_list[:20], np.ones(q_list[:20].size)*(1/0.75),'-', color='black',
    alpha=.7)

ax[1].set_ylim([.001, 1.49])
ax[1].set_ylabel(r'$h(q)$',labelpad=3, fontsize=24)
ax[1].set_xlabel(r'$q$',labelpad=3, fontsize=24)

axi2 = fig.add_axes([.85, .6, .135, .37])
axi2.plot(q_list, q_list * slopes4 - 1,'o', markersize=6, color=colours[0],
    markerfacecolor='none')
axi2.plot(q_list, q_list * slopes5 - 1,'s', markersize=6, color=colours[1],
    markerfacecolor='none')
axi2.plot(q_list, q_list * slopes6 - 1,'D', markersize=6, color=colours[2],
    markerfacecolor='none')

axi2.set_ylim([-16, 1.2])
axi2.set_yticks([-12, -6, 0])
axi2.set_xlabel(r'$q$', labelpad=3, fontsize=24)
axi2.set_ylabel(r'$\tau(q)$', labelpad=-5, fontsize=24)

ax[0].legend(loc=4, handletextpad=.3, handlelength=.5, ncol=3,
    columnspacing=.65)
ax[1].legend(loc=3, handletextpad=.3, handlelength=.5, ncol=3,
    columnspacing=.65)

locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
ax[0].yaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                      numticks=100)
ax[0].yaxis.set_minor_locator(locmin)
ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

fig.text(.005, .92, r'a)',fontsize=28)
fig.text(.505, .92, r'b)',fontsize=28)

fig.subplots_adjust(left=.07, bottom=.17, right=.99, top=.99, hspace=.06,
wspace=.15)
# fig.savefig('fig02.pdf', trasparent=True)
