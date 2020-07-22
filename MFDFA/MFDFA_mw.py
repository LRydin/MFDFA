## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval

def MFDFA_mw(timeseries: np.ndarray, lag: np.ndarray=None, order: int=1,
          q: np.ndarray=2, window: int=1, stat: bool=False,
          extensions: list=['None'], modified: bool=False) -> np.ndarray:
    """
    Moving window method for Multifractal Detrended Fluctuation Analysis of a
    timeseries. This is particularly valuabe for short time series. Notice that
    this increases the run time of the MFDFA by ``N!``.

    Considers a moving window over the timeseries instead of segmenting the
    timeseries in disjoint parts. For more information about MFDFA, read the
    MFDFA documentation.


     generates
    a fluctuation function F²(q,s), with s the segment size and q the q-powers,
    Take a timeseries Xₜ, find the integral Yₜ = cumsum(Xₜ), and segment the
    timeseries into Nₛ segments of size s.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries ``(N, 1)``. The timeseries of length ``N``.

    lag: np.ndarray of ints
        An array with the window sizes to calculate (ints). Notice
        ``min(lag) > order + 1`` given a polynomial fit of order ``m`` needs at
        least ``m`` points. The results are meaningless for 'order = m' and for
        lag > size of data / 4 since there is low statistics with < 4 windows
        to divide the timeseries.

    order: int (default = 1)
        The order of the polynomials to approximate. ``order = 1`` is the DFA1,
        which is a least-square fit of the data with a first order polynomial (a
        line), ``order = 2`` is a second-order polynomial, etc.

    q: np.ndarray (default = 2)
        Fractal exponent to calculate. Array in ``[-10,10]``. The values = 0
        will be removed, since the code does not converge there. ``q = 2`` is
        the standard Detrended Fluctuation Analysis as is set a default.

    window: int (default = 1)
        int > 0 with the number of steps the window shoud move over the data.
        ``window = 1`` will move window by ``1`` step.

    stat: bool (default = False)
        Calculates the standard deviation associated with each segment's
        averaging.

    extensions: list (default = 'None')
        Include some of the recent added functionalities to DFA and MFDFA.
        Currently implemented are:
            ``'eDFA'`` - A method to evaluate the strength of multifractality.

    modified: bool (default = False)
        For data with the Hurst index ≈ 0, i.e., strongly anticorrelated, a
        standard MFDFA will result in inacurate results, thus a further
        integration of the timeseries yields a modified scaling coefficient.

    Returns
    -------
    lag: np.ndarray of ints
        Array of lags, realigned, preserving only different lags and with
        entries > order + 1

    f: np.ndarray
        An array of shape ``(size(lag), size(q))`` of variances over the
        indicated ``lag`` windows and the indicated ``q``-fractal powers.

    f_std: np.ndarray
        Present only if ``stat = True``. An array of shape
        ``(size(lag), size(q))`` of the standard deviations associated with each
        averaging of ``f``, over the indicated ``lag`` windows and the indicated
        ``q``-fractal powers.
    """

    # Force lag to be ints, ensure lag > order + 1
    lag = lag[lag > order + 1]
    lag = np.round(lag).astype(int)

    # Assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    # Assert window is int and > 0
    assert isinstance(window, int), "'window' is not integer"
    assert window > 0, "'window' is not > 0"

    timeseries = timeseries.reshape(-1,1)
    # Size of array
    N = timeseries.shape[0]

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype = float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -.1) + (q > .1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)

    # x-axis
    X = np.linspace(1, lag.max(), lag.max())

    # "Profile" of the series
    Y = np.cumsum(timeseries - np.mean(timeseries))

    # Cumulative "profile" for strongly anticorrelated data:
    if modified == True:
        Y = np.cumsum(Y - np.mean(Y))

    # Return f of (fractal)-variances
    f = np.empty((0, q.size))

    if stat == True:
        f_std = np.empty((0, q.size))

    if 'eDFA' in extensions:
        f_eDFA = np.empty((0, q.size))

    # Loop over elements in lag and use an moving window
    # Notice that given one has to slip the timeseries into diferent segments of
    # length lag(), so some elements at the end of the array might be missing.

    for i in lag:
        F = np.empty(0)
        for j in range(0, i-1, window):

            # subtract j points as the moving window shortens the data
            N_0 = N - j

            # Reshape into (N_0/lag, lag)
            Y_ = Y[j:N - N_0 % i].reshape((N - N_0 % i) // i, i)

            # Perform a polynomial fit to each segments
            p = polyfit(X[:i], Y_.T, order)

            # Subtract the trend from the fit and calculate the variance
            F = np.append(F, np.var(Y_ - polyval(X[:i], p), axis = 1) )

        # Caculate the Multi-Fractal Detrended Fluctuation Analysis
        f = np.append(f,
              np.float_power(
                np.mean(np.float_power(F, q / 2), axis = 1) / 2,
              1 / q.T),
            axis = 0)


        # Caculate standard deviation associated with each mean
        if stat == True:
            f_std = np.append(f_std,
                  np.float_power(
                    np.std(np.float_power(F, q / 2), axis = 1) / 2,
                  1 / q.T),
                axis = 0)

        if 'eDFA' in extensions:
            f_eDFA = np.append(f_eDFA, eDFA(F))


    if stat == False:
        return lag, f
    if stat == True:
        return lag, f, f_std

def eDFA(F: np.ndarray) -> np.ndarray:
    """
    In the reference indicated below a measure of nonstationarity was added by
    including a subsequent calculation of the extrema of the DFA. Denote
    :math:`dF_q^2(s)` the difference of the extrema at each segment, i.e.,

    .. math::

        dF_q^2(s) = \max[F_q^2(s)] - \min[F_q^2(s)]

    Parameters
    ----------
    F: np.ndarray
        Fluctuation function

    Returns
    -------
    res: np.ndarray
        Difference of `max` and `min`.

    References
    ----------
    'Detrended fluctuation analysis of cerebrovascular responses to abrupt
    changes in peripheral arterial pressure in rats', A.N. Pavlov, A.S.
    Abdurashitov, A.A. Koronovskii, Jr., O.N. Pavlova, O.V.
    Semyachkina-Glushkovskaya, J. Kurths, CNSNS 105232,
    doi:10.1016/j.cnsns.2020.105232
    """

    return np.max(F) - np.min(F)

# TODO: Add log calculator for q ≈ 0
