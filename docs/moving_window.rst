Moving window for segmentation
------------------------------

For short timeseries the segmentation of the data—especially for large lags—results in bad statistics, e.g. if a timeseries has `2048` datapoints and one wishes to study the flucutation analysis up to a lag of `512`, only 4 segmentations of the data are possible for the lag `512`. Instead one can use an moving window over the timeseries to obtain better statistics at large lags.

Using :code:`MFDFA`'s :code:`window` extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To utilise a moving window one has to declare the moving windows step-size, i.e., the number of data points the window will move over the data. Say we wish to increase the statistics of the aforementioned example to include a moving window moving `32` steps (so one has 64 segments at a lag of `512`)

.. code:: python

   # Select a band of lags, which usually ranges from
   # very small segments of data, to very long ones, as
   lag = np.logspace(0.7, 4, 30).astype(int)

   # Obtain the (MF)DFA by declaring the IMFs to subtract
   # in a list in the dictionary of the extensions
   lag, dfa, edfa = MFDFA(y, lag = lag, extensions = {'window': 32})
