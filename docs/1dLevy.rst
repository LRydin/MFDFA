Multifractality in one dimensional distributions
================================================

To show how multifractality can be studied, let us take a sample of random numbers of a symmetric `Lévy distribution <https://en.wikipedia.org/wiki/L%C3%A9vy_distribution>`_.


Univariate random numbers from a Lévy stable distribution
---------------------------------------------------------

To obtain a sample of random numbers of Lévy stable distributions, use `scipy`'s `levy_stable`. In particular, take an :math:`\alpha`-stable distribution, with :math:`\alpha=1.5`

.. code:: python

   # Imports
   from MFDFA import MFDFA
   from scipy.stats import levy_stable

   # Generate 100000 points
   alpha = 1.5
   X = levy_stable.rvs(alpha=alpha, beta = 0, size=10000)


For :code:`MFDFA` to detect the multifractal spectrum of the data, we need to vary the parameter :math:`q\in[-10,10]` and exclude :math:`0`. Let us also use a quadratic polynomial fitting by setting :code:`order=2`

.. code:: python

   # Select a band of lags, which are ints
   lag = np.unique(np.logspace(0.5, 3, 100).astype(int))

   # Select a list of powers q
   q_list = np.linspace(-10,10,41)
   q_list = q_list[q_list!=0.0]

   # The order of the polynomial fitting
   order = 2

   # Obtain the (MF)DFA as
   lag, dfa = MFDFA(y, lag = lag, q = q_list, order = order)


Again, we plot this in a double logarithmic scale, but now we include 6 curves, from 6 selected :math:`q={-10,-5-2,2,5,10}`. Include as well are the theoretical curves for :math:`q=-10`, with a slope of :math:`1/\alpha=1/1.5` and :math:`q=10`, with a slope of :math:`1/q=1/10`


.. image:: /_static/fig2.png
  :height: 450
  :align: center
  :alt: Plot of the MFDFA of a Lévy stable distribution for a few q values.
