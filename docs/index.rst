MFDFA's documentation
=====================

.. toctree::
   :maxdepth: 2

:code:`MFDFA` is a python implementation of Multifractal Detrended Fluctuation Analysis, first developed by by Peng *et al.* ¹ and later extended to study multifractality :code:`MFDFA` by Kandelhardt *et al.* ².

Installation
============

For the moment the library is available from TestPyPI, so you can use

::

   pip install -i https://test.pypi.org/simple/ MFDFA

Then on your favourite editor just use

.. code:: python

   from MFDFA import MFDFA

An exemplary one-dimensional fractional Ornstein–Uhlenbeck process
==================================================================

For a more detailed explanation on how to integrate an Ornstein–Uhlenbeck process, see the `kramersmoyal's package <https://github.com/LRydin/KramersMoyal#a-one-dimensional-stochastic-process>`_
You can also follow the [fOU.ipynb](/examples/fOU.ipynb)

Generating a fractional Ornstein–Uhlenbeck process
--------------------------------------------------

This is one method of generating a (fractional) Ornstein–Uhlenbeck process with :math:`H=0.7`, employing a simple Euler–Maruyama integration method

.. code:: python

   # Imports
   from MFDFA import MFDFA
   from MFDFA import fgn
   # where this second library is to generate fractional Gaussian noises

   # integration time and time sampling
   t_final = 500
   delta_t = 0.001

   # Some drift theta and diffusion sigma parameters
   theta = 0.3
   sigma = 0.1

   # The time array of the trajectory
   time = np.arange(0, t_final, delta_t)

   # The fractional Gaussian noise
   H = 0.7
   dB = (t_final ** H) * fgn(N = time.size, H = H)

   # Initialise the array y
   y = np.zeros([time.size])

   # Integrate the process
   for i in range(1, time.size):
      y[i] = y[i-1] - theta * y[i-1] * delta_t + sigma * dB[i]

And now you have a fractional process with a self-similarity exponent :math:`H=0.7`

Using the :code:`MFDFA`
-----------------------
To now utilise the :code:`MFDFA`, we take this exemplary process and run the (multifractal) detrended fluctuation analysis. For now lets consider only the monofractal case, so we need only :math:`q = 2`.

.. code:: python

   # Select a band of lags, which usually ranges from
   # very small segments of data, to very long ones, as
   lag = np.logspace(0.7, 4, 30).astype(int)
   # Notice these must be ints, since these will segment
   # the data into chucks of lag size

   # Select the power q
   q = 2

   # The order of the polynomial fitting
   order = 1

   # Obtain the (MF)DFA as
   lag, dfa = MFDFA(y, lag = lag, q = q, order = order)


Now we need to visualise the results, which can be understood in a log-log scale. To find *H* we need to fit a line to the results in the log-log plot

.. code:: python

   # To uncover the Hurst index, lets get some log-log plots
   plt.loglog(lag, dfa, 'o', label='fOU: MFDFA q=2')

   # And now we need to fit the line to find the slope. We will
   # fit the first points, since the results are more accurate
   # there. Don't forget that if you are seeing in log-log
   # scales, you need to fit the logs of the results
   np.polyfit(np.log(lag[:15]), np.log(dfa[:15]),1)[0]

   # Now what you should obtain is: slope = H + 1

.. image:: /_static/fig1.png
  :height: 450
  :align: center
  :alt: MFDFA of a fractional Ornstein–Uhlenbeck process


References
----------

¹ Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., & Goldberger, A. L. (1994). *Mosaic organization of DNA nucleotides*. `Physical Review E, 49(2), 1685–1689 <https://doi.org/10.1103/PhysRevE.49.1685>`_

² Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E., Havlin, S., Bunde, A., & Stanley, H. E. (2002). *Multifractal detrended fluctuation analysis of nonstationary time series*. `Physica A: Statistical Mechanics and Its Applications, 316(1-4), 87–114 <https://doi.org/10.1016/S0378-4371(02)01383-3>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
