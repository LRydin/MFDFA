Employing Empirical Mode Decompositions for detrending
------------------------------------------------------

`Empirical Mode Decomposition <https://en.wikipedia.org/wiki/Hilbert%E2%80%93Huang_transform>`_ (EMD), or maybe more correctly described, the Hilbert─Huang transform is a transformation analogous to a Fourier or Hilbert transform that decomposes a one-dimensional timeseries or signal into its Intrinsic Mode Functions (IMFs).
For our purposes, we simply want to employ EMD to detrend a timeseries.

.. warning::

   To use this feature, you need to first install `PyEMD <https://github.com/laszukdawid/PyEMD>`_ (EMD-signal) with
   
   ::

      pip install EMD-signal

Understanding :code:`MFDFA`'s :code:`EMD` detrender
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take a timeseries :code:`y` and extract the Intrinsic Mode Functions (IMFs)

.. code:: python

   # Import
   from MFDFA import IMFs

   # Extract the IMFs simply by employing
   IMF = IMFs(y)

From here one obtains a :code:`(..., y.size)`. Best now to study the different IMFs is to plot them and the timeseries :code:`y`

.. code:: python

   # Import
   import matplotlib.pyplot as plt

   # Plot the timeseries and the IMFs 6,7, and 8
   plt.plot(X, color='black')
   plt.plot(np.sum(IMF[[6,7,8],:], axis=0).T)

Using :code:`MFDFA` with :code:`EMD`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To now perform the multifractal detrended fluctuation analysis, simply insert the IMFs desired to be subtracted from the timeseries. This will also for :code:`order = 0`, not to do any polynomial detrending.

.. code:: python

   # Select a band of lags, which usually ranges from
   # very small segments of data, to very long ones, as
   lag = np.logspace(0.7, 4, 30).astype(int)

   # Obtain the (MF)DFA by declaring the IMFs to subtract
   # in a list in the dictionary of the extensions
   lag, dfa = MFDFA(y, lag = lag, extensions = {"EMD": [6,7,8]})
