Extended Detrended Fluctuation Analysis
---------------------------------------

In the publication `Detrended fluctuation analysis of cerebrovascular responses to abrupt changes in peripheral arterial pressure in rats.  <https://doi.org/10.1016/j.cnsns.2020.105232>`_ the authors introduce a new metric similar to the conventional Detrended Fluctuation Analysis (DFA) which they denote *Extended* Detrended Fluctuation Analysis (eDFA), which relies on extracting the difference of the minima and maxima for each segmentation of the data, granting a new power-law exponent to study, i.e., as in eq. (5) in the paper

.. math::
   \mathrm{d}F (n) = \mathrm{max}[F(n)] - \mathrm{min}[F(n)],

which in turn results in

.. math::
   \mathrm{d}F(n) \sim n^\beta.


Using :code:`MFDFA`'s :code:`eDFA` extension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To obtain the :code:`eDFA`, simply set the extension to :code:`True` and add a new output function, here denoted :code:`edfa`

.. code:: python

   # Select a band of lags, which usually ranges from
   # very small segments of data, to very long ones, as
   lag = np.logspace(0.7, 4, 30).astype(int)

   # Obtain the (MF)DFA by declaring the IMFs to subtract
   # in a list in the dictionary of the extensions
   lag, dfa, edfa = MFDFA(y, lag = lag, extensions = {'eDFA': True})
