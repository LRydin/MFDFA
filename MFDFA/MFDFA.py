## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended  fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
import matplotlib.pyplot as plt

def MFDFA(timeseries: np.ndarray, lag: np.ndarray=None, order: int=1,
          q: np.ndarray=2, modified: bool=False, error: bool=False,
          overlap: bool=True) -> np.ndarray:
    """
    Multi-Fractal Detrended Fluctuation Analysis of timeseries.

    Parameters
    ----------
    timeseries: np.ndarray
        A 1-dimensional timeseries (N, 1). The timeseries of length N.

    lag: np.ndarray of ints
        An array with the window sizes to calculate (ints). Notice
        min(lag) > order + 1 because to fit a polynomial of order m one needs at
        least m points. The results are meaningless for 'order = m' and for
        lag ≈ size of data / 4 since there is low statistics with only 4 windows
        to divide the timeseries.

    order: int
        The order of the polynomials to approximate. 'order = 1' is the DFA1,
        which is a least-square fit of the data with a first order polynomial (a
        line), 'order = 2' is a second-order polynomial, etc..

    q: np.ndarray
        Fractal exponent to calculate. Array in [-10,10]. The values = 0 will be
        removed, since the code does not converge there. q = 2 is the standard
        Detrended Fluctuation Analysis as is set a default.

    modified: bool
        For data with the Hurst exponent ≈ 0, i.e., strongly anticorrelated, a
        standard MFDFA will result in inacurate results, thus a further
        integration of the timeseries yields a modified scaling coefficient.

    error: bool
        Output standard deviations of calculation. If error = True, output is a
        tuple.

    overlap: bool=True
        [to be implemented] for short timeseries, allows overlap of windows.

    Returns
    -------
    lag: np.ndarray of ints
        Array of lags, realigned and with entries > order + 1

    f: np.ndarray
        A array of shape (size(lag),size(q)) of variances over the indicated
        lag windows and the indicated q-fractal powers.

    f_std: np.ndarray
        A array of shape (size(lag),size(q)) of the standard deviations of the
        averaging of the windows to account for the errors in the calculation.
        Requires error = True.
    """

    # Force lag to be ints

    lag = lag[lag > order + 1]
    lag = np.round(lag).astype(int)

    # Size of array
    N = timeseries.size

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

    # Return f_std of errors
    if error == True:
        f_std = np.empty((0, q.size))

    # Loop over elements in lag
    # Notice that given one has to slip the timeseries into diferent segments of
    # length lag(), so some elements at the end of the array might be missing.
    # The same procedure it run in reverse, were elements at the begining of the
    # series are discared instead
    for i in lag:
        # Reshape into (N/lag, lag)
        Y_ = Y[:N - N % i].reshape((N - N % i) // i, i)
        Y_r = Y[N % i:].reshape((N - N % i) // i, i)

        # Perform a polynomial fit to each segments
        p = polyfit(X[:i], Y_.T, order)
        p_r = polyfit(X[:i], Y_r.T, order)

        # Subtract the trend from the fit and calculate the variance
        F = np.var(Y_ - polyval(X[:i], p), axis = 1)
        F_r = np.var(Y_r - polyval(X[:i], p_r), axis = 1)

        # Caculate the Multi-Fractal Detrended Fluctuation Analysis
        f = np.append(f,
              np.float_power(
                np.mean( np.float_power(F, q / 2), axis = 1) / 2,
              1 / q.T)
              + np.float_power(
                np.mean( np.float_power(F_r, q / 2), axis = 1) / 2,
              1 / q.T),
            axis = 0)

        # if error = True calculates the errors
        if error == True:
            f_std = np.append(f_std,
                  np.float_power(
                    np.std( np.float_power(F, q / 2), axis = 1) / 2,
                  1 / q.T)
                  + np.float_power(
                    np.std( np.float_power(F_r, q / 2), axis = 1) / 2,
                  1 / q.T),
                axis = 0)

        # @Francisco Magia a fazer aqui?

    if error == False:
        return lag, f
    elif error == True:
        return lag, f, f_std

def MFDFA_plot(lag: np.ndarray, f: np.ndarray) -> None:
    """
    Log-log of lag and DFA function

    Parameters
    ----------
    lag: np.ndarray of ints
        x-axis of the plot, with the window sizes in logarithmic scale.

    f: np.ndarray
        The array of variances over the indicated windows.
    """

    plt.loglog(lag, f, ',')
    plt.xlabel('window size')
    plt.xlabel('variances')

    return
