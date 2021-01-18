## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from .emddetrender import detrendedtimeseries

def MFDFA(timeseries: np.ndarray, lag: np.ndarray=None, order: int=1,
          q: np.ndarray=2, stat: bool=False, modified: bool=False,
          extensions: dict={'EMD':False, 'eDFA':False, 'window':False}) -> np.ndarray:
    """
    Multifractal Detrended Fluctuation Analysis of timeseries. MFDFA generates
    a fluctuation function F²(q,s), with s the segment size and q the q-powers,
    Take a timeseries Xₜ, find the integral Yₜ = cumsum(Xₜ), and segment the
    timeseries into Nₛ segments of size s.

    .. math::

        F^2(v,s) = \dfrac{1}{s} \sum_{i=1}^s [Y_{(v-1)s + i} - y_{v,i}]^2,
        ~\mathrm{for}~v=1,2, \dots, N_s,

    with :math:`y_{v,i}` the polynomial fittings of order m. Having obtained
    the variances of each (detrended) segment, average over s and increase s, to
    obtain the fluctuation function :math:`F_q^2(s)` depending on the segment
    length.

    .. math::

        F_q^2(s) = \Bigg\{\dfrac{1}{N_s} \sum_{v=1}^{N_s}
        [F^2(v,s)]^{q/2}\Bigg\}^{1/q}

    The fluctuation function :math:`F_q^2(s)` can now be plotted in a log-log
    scale, the slope of the fluctuation function :math:`F_q^2(s)` vs the
    s-segment size is the self-similarity scaling :math:`h(q)`

    .. math::

        F_q^2(s) \sim s^{h(q)}.

    If :math:`H ≈ 0` in a monofractal series, use a second integration
    step by setting :code:`modified = True`.

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
        line), ``order = 2`` is a second-order polynomial, etc..
        ``order = 0`` skips the detrending process and hence gives the
        nondetrended fluctuation functions, i.e., simply Fluctuation Analysis.

    q: np.ndarray (default = 2)
        Fractal exponent to calculate. Array in ``[-10,10]``. The values = 0
        will be removed, since the code does not converge there. ``q = 2`` is
        the standard Detrended Fluctuation Analysis as is set a default.

    stat: bool (default = False)
        Calculates the standard deviation associated with each segment's
        averaging.

    modified: bool (default = False)
        For data with the Hurst index ≈ 0, i.e., strongly anticorrelated, a
        standard MFDFA will result in inacurate results, thus a further
        integration of the timeseries yields a modified scaling coefficient.

    extensions: dict
     - ``EMD``: list (default ``False``)
        If not ``None``, requires a list of indices of the user-chosen IMFs
        obtained from an (externally performed) EMD analysis. The indexing
        starts from ``0``. Will enforce ``order = 0`` since there is no need
        for a polynomial detrending.
     - ``eDFA``: bool (default ``False``)
        A method to evaluate the strength of multifractality. Calls function
        `eDFA()`.
     - ``window``: bool (default ``False``)
        A moving window for smaller timeseries. Set ``window`` as int > 0 with
        the number of steps the window shoud move over the data. ``window = 1``
        will move window by ``1`` step. Since the timeseries is segmented at
        each lag lenght, any window choise > lag is only segmented once.

    Returns
    -------
    lag: np.ndarray of ints
        Array of lags, realigned, preserving only different lags and with
        entries > order + 1

    f: np.ndarray
        A array of shape (size(lag),size(q)) of variances over the indicated
        lag windows and the indicated q-fractal powers.

    References
    ----------
    .. [Peng1994] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons, H. E.
        Stanley, and A. L. Goldberger. "Mosaic organization of DNA nucleotides."
        Phys. Rev. E, 49(2), 1685–1689, 1994.
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.
    """

    # Force lag to be ints, ensure lag > order + 1
    lag = lag[lag > order + 1]
    lag = np.round(lag).astype(int)

    # Assert if timeseries is 1 dimensional
    if timeseries.ndim > 1:
        assert timeseries.shape[1] == 1, "Timeseries needs to be 1 dimensional"

    timeseries = timeseries.reshape(-1,1)

    # Size of array
    N = timeseries.shape[0]

    # Assert if window is given, that it is int and > 0
    window = False
    if 'window' in extensions:
        if extensions['window'] != False:
            window = extensions['window']
            assert isinstance(window, int), "'window' is not integer"
            assert window > 0, "'window' is not > 0"


    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype = float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -.1) + (q > .1)]

    # Reshape q to perform np.float_power
    q = q.reshape(-1, 1)

    # x-axis needed for polyfit
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

    # Check which extensions are requested
    if ('eDFA', True) in extensions.items():
        f_eDFA = np.empty((0, q.size))

    if 'EMD' in extensions:
        if extensions['EMD'] != False:
            # assert the dictionary entry is a list
            assert isinstance(extensions['EMD'], list), 'list IMFs to detrend'

            # Detrending of the timeseries using EMD with given IMFs in a list
            Y = detrendedtimeseries(Y, extensions['EMD'])

            # Force order = 0 since the data is detrended with EMD, i.e., no
            # need to do polynomial fittings anymore
            order = 0

    # Loop over elements in lag
    # Notice that given one has to split the timeseries into different segments
    # of length lag(), some elements at the end of the array might be
    # missing. The same procedure is run in reverse—if not using an moving
    # window—where elements at the begining of the series are discarded instead.
    for i in lag:

        # Standard option
        if window == False:
            # Reshape into (N/lag, lag)
            Y_ = Y[:N - N % i].reshape((N - N % i) // i, i)
            Y_r = Y[N % i:].reshape((N - N % i) // i, i)

            # If order = 0 one gets simply Fluctuation Analysis (FA), or if one is
            # using the EMD setting the data is detrended and no polynomial fitting
            # is needed.
            if order == 0:
                # Skip detrending
                F = np.append(np.var(Y_, axis=1), np.var(Y_r, axis=1))

            else:
                # Perform a polynomial fit to each segments
                p = polyfit(X[:i], Y_.T, order)
                p_r = polyfit(X[:i], Y_r.T, order)

                # Subtract the trend from the fit and calculate the variance
                F = np.append(
                    np.var(Y_ - polyval(X[:i], p), axis = 1),
                    np.var(Y_r - polyval(X[:i], p_r), axis = 1)
                )

        # For short timeseries, using a moving window instead of segmenting the
        # timeseries. Notice the number of operations is considerably larger
        # depending on the moving window displacement.
        if window != False:

            F = np.empty(0)
            for j in range(0, i-1, window):

                # subtract j points as the moving window shortens the data
                N_0 = N - j

                # Reshape into (N_0/lag, lag)
                Y_ = Y[j:N - N_0 % i].reshape((N - N_0 % i) // i, i)

                # If order = 0 one gets simply Fluctuation Analysis (FA), or if one
                # is using the EMD setting the data is detrended and no polynomial
                # fitting is needed.
                if order == 0:
                    # Skip detrending
                    F = np.append(F, np.var(Y_, axis=1))

                else:
                    # Perform a polynomial fit to each segments
                    p = polyfit(X[:i], Y_.T, order)

                    # Subtract the trend from the fit and calculate the variance
                    F = np.append(F, np.var(Y_ - polyval(X[:i], p), axis = 1))

        # Caculate the Multifractal (Non)-Detrended Fluctuation Analysis
        f = np.append(f,
              np.float_power(
                np.mean(np.float_power(F, q / 2), axis = 1),
              1 / q.T),
            axis = 0)

        # Caculate standard deviation associated with each mean
        if stat == True:
            f_std = np.append(f_std,
                      np.float_power(
                        np.std(np.float_power(F, q / 2), axis = 1),
                      1 / q.T),
                    axis = 0)

        if ('eDFA', True) in extensions.items():
            f_eDFA = np.append(f_eDFA, eDFA(F))


    if stat == False:
        if ('eDFA', True) in extensions.items():
            return lag, f, np.vstack(f_eDFA)
        else:
            return lag, f
    if stat == True:
        if ('eDFA', True) in extensions.items():
            return lag, f, f_std, np.vstack(f_eDFA)
        else:
            return lag, f, f_std

def eDFA(F: np.ndarray) -> np.ndarray:
    """
    In the reference indicated below a measure of nonstationarity was added by
    including a subsequent calculation of the extrema of the DFA. Denoted
    :math:`dF_q^2(s)` the difference of the extrema at each segment, i.e.,

    .. math::

        dF_q^2(s) = \max[F_q^2(s)] - \min[F_q^2(s)]

    Parameters
    ----------
    F: np.ndarray
        Fluctuation function given by the ``MFDFA()``.

    Returns
    -------
    res: np.ndarray
        Difference of `max` and `min`.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [Pavlov2020] A. N. Pavlov, A. S. Abdurashitov, A. A. Koronovskii Jr., O.
        N. Pavlova, O. V. Semyachkina-Glushkovskaya, and J. Kurths. "Detrended
        fluctuation analysis of cerebrovascular responses to abrupt changes in
        peripheral arterial pressure in rats." CNSNS 85, 105232, 2020
    """

    return np.max(F) - np.min(F)

# TODO: Add log calculator for q ≈ 0
