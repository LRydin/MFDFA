## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from EMD.emdDetrender import getDetrendedtimeseries

def MFDFA(
        timeseries: np.ndarray,
        lag: np.ndarray=None,
        order: int=1,
        q: np.ndarray=2,
        modified: bool=False,
        emdConfig: dict={
            "detrendByEMD":False,
            "chosenIMFIndices":[]
        }
    ) -> np.ndarray:
    """
    Multi-Fractal Detrended Fluctuation Analysis of timeseries. MFDFA generates
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

    If :math:`H \approx 0` in a monofractal series, use a second integration
    step by setting :code:`modified = True`.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries (N, 1). The timeseries of length N.

    lag: np.ndarray of ints
        An array with the window sizes to calculate (ints). Notice
        min(lag) > order + 1 because to fit a polynomial of order m one needs at
        least m points. The results are meaningless for 'order = m' and for
        lag > size of data / 4 since there is low statistics with < 4 windows
        to divide the timeseries.

    order: int
        The order of the polynomials to approximate. 'order = 1' is the DFA1,
        which is a least-square fit of the data with a first order polynomial (a
        line), 'order = 2' is a second-order polynomial, etc..
        order = 0 skips the detrending process and hence gives the Nondetrended
        fluctuatioin functions.

    q: np.ndarray
        Fractal exponent to calculate. Array in [-10,10]. The values = 0 will be
        removed, since the code does not converge there. q = 2 is the standard
        Detrended Fluctuation Analysis as is set a default.

    modified: bool
        For data with the Hurst exponent ≈ 0, i.e., strongly anticorrelated, a
        standard MFDFA will result in inacurate results, thus a further
        integration of the timeseries yields a modified scaling coefficient.
    
    emdConfig: dict
        "detrendByEMD": bool
            If True, then detrends the timeseries by Emperical Mode Decomposition (EMD) 
            analysis based on the IMF-indices listed in emdConfig["chosenIMFIndices"] 
        "chosenIMFIndices": list
            Indices of the user-chosen IMFs obtain from an (externally performed) EMD
            analysis. The indexing starts from 0.

    Returns
    -------
    lag: np.ndarray of ints
        Array of lags, realigned, preserving only different lags and with
        entries > order + 1

    f: np.ndarray
        A array of shape (size(lag),size(q)) of variances over the indicated
        lag windows and the indicated q-fractal powers.
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

    # Loop over elements in lag
    # Notice that given one has to slip the timeseries into diferent segments of
    # length lag(), so some elements at the end of the array might be missing.
    # The same procedure it run in reverse, were elements at the begining of the
    # series are discared instead
    for i in lag:
        if  emdConfig["detrendByEMD"] == False: 
            # Reshape into (N/lag, lag)
            Y_ = Y[:N - N % i].reshape((N - N % i) // i, i)
            Y_r = Y[N % i:].reshape((N - N % i) // i, i)

            if order == 0:
                # Skip detrending
                F = np.var(Y_, axis=1)
                F_r =  np.var(Y_r, axis=1)
            else:
                # Perform a polynomial fit to each segments
                p = polyfit(X[:i], Y_.T, order)
                p_r = polyfit(X[:i], Y_r.T, order)

                # Subtract the trend from the fit and calculate the variance
                F = np.var(Y_ - polyval(X[:i], p), axis = 1)
                F_r = np.var(Y_r - polyval(X[:i], p_r), axis = 1)
        
        else:
            # detrend using EMD
            IMFsToBeSubtracted = emdConfig["chosenIMFIndices"]
            Y_detrended = getDetrendedtimeseries(Y, IMFsToBeSubtracted)
            
            # Reshape into (N/lag, lag)
            Y_ = Y_detrended[:N - N % i].reshape((N - N % i) // i, i)
            Y_r = Y_detrended[N % i:].reshape((N - N % i) // i, i)

            # calculate the variance
            F = np.var(Y_, axis=1)
            F_r =  np.var(Y_r, axis=1)

        # Caculate the Multi-Fractal Nondetrended or Detrended Fluctuation Analysis
        f = np.append(f,
              np.float_power(
                np.mean( np.float_power(F, q / 2), axis = 1) / 2,
              1 / q.T)
              + np.float_power(
                np.mean( np.float_power(F_r, q / 2), axis = 1) / 2,
              1 / q.T),
            axis = 0)

    return lag, f

# TODO: Add log calculator for q ≈ 0
