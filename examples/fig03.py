# created by Leonardo Rydin Gorjão. Most python libraries are standard (e.g. via
# Anaconda). If TeX is not present in the system comment out lines 13 to 16.

import numpy as np
import pandas as pd

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

# %% ############################ Sunspot data #################################
sun = pd.read_csv('sunspots.csv', ';', header=None)
time = np.array(sun[0])

# select columns
sun = np.array(sun[5])

# mask -1 as np.nan (-1 means no entry recorded)
sun[sun ==-1.] = np.nan

sun_m = np.ma.masked_invalid(sun)

# %%
lag = np.unique(np.round(np.logspace(.4, 4, 100)))[30:]
q_list = np.linspace(-10,10,41)
q_list = q_list[q_list!=0.0]
lag, dfa_sun =  MFDFA(sun_m, lag, q=q_list, order=1)

# %% ################################## FIG03 ##################################
fig, ax = plt.subplots(1,3, figsize=(16,4))

ax[0].loglog(lag[:], dfa_sun[:,[0]],  '^', markersize=6, markerfacecolor='none',
    color=colours[0], label=r'$q=-10$')
ax[0].loglog(lag[:], dfa_sun[:,[10]], 'D', markersize=6, markerfacecolor='none',
    color=colours[0], label=r'$q=-5$')
ax[0].loglog(lag[:], dfa_sun[:,[17]], 'v', markersize=6, markerfacecolor='none',
    color=colours[0], label=r'$q=-2$')
ax[0].loglog(lag[:], dfa_sun[:,[23]], '<', markersize=6, markerfacecolor='none',
    color=colours[1], label=r'$q=2$')
ax[0].loglog(lag[:], dfa_sun[:,[29]], 'H', markersize=6, markerfacecolor='none',
    color=colours[1], label=r'$q=5$')
ax[0].loglog(lag[:], dfa_sun[:,[39]], '>', markersize=6, markerfacecolor='none',
    color=colours[1], label=r'$q=10$')

ax[0].set_ylabel(r'$F_q(s)$',labelpad=7,fontsize=24)
ax[0].set_xlabel(r'segment size $s$',labelpad=3,fontsize=24)

slopes_sun = np.polynomial.polynomial.polyfit(np.log(lag)[20:55],np.log(dfa_sun)[20:55],1)[1]

ax[1].plot(q_list[:20], slopes_sun[:20],'o', markersize=9,
    markerfacecolor='none', color=colours[0])
ax[1].plot(q_list[20:], slopes_sun[20:],'o', markersize=9,
    markerfacecolor='none', color=colours[1])

ax[1].set_ylim([None,2.5])
ax[1].set_ylabel(r'$h(q)$',labelpad=5,fontsize=24)
ax[1].set_xlabel(r'$q$',labelpad=3,fontsize=24)

axi2 = fig.add_axes([0.52, 0.6, 0.135, .37])
axi2.plot(q_list[:20], q_list[:20]*slopes_sun[:20]-1,'o', markersize=6,
    color=colours[0], markerfacecolor='none')
axi2.plot(q_list[20:], q_list[20:]*slopes_sun[20:]-1,'o', markersize=6,
    color=colours[1], markerfacecolor='none')

axi2.set_xlabel(r'$q$',labelpad=3,fontsize=24)
axi2.set_ylabel(r'$\tau(q)$',labelpad=-3,fontsize=24)
axi2.set_yticks([-20,-10,0,10])

t_sun = q_list * slopes_sun - 1
hq_sun = np.gradient(t_sun) / np.gradient(q_list)
f_sun = q_list * hq_sun - t_sun

ax[2].plot(hq_sun[5:20], f_sun[5:20],'o', markersize=9,
    markerfacecolor='none', color=colours[0])
ax[2].plot(hq_sun[20:], f_sun[20:],'o', markersize=9,
    markerfacecolor='none', color=colours[1])

ax[2].set_xlabel(r'$\alpha$',labelpad=3,fontsize=24)
ax[2].set_ylabel(r'$D(\alpha)$',labelpad=-5,fontsize=24)

ax[0].legend(loc=4, handletextpad=.3, handlelength=.5, ncol=2,
    columnspacing=.65)

locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
ax[0].yaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1,
                                      numticks=100)
ax[0].yaxis.set_minor_locator(locmin)
ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

fig.text(.005, .92, r'a)', fontsize=28)
fig.text(.34, .92, r'b)', fontsize=28)
fig.text(.67, .92, r'c)', fontsize=28)

fig.subplots_adjust(left=.07, bottom=.17, right=.99, top=.99, hspace=.06,
    wspace=.25)
# fig.savefig('fig03.pdf', trasparent=True)
