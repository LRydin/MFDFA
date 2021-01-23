## This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

import numpy as np

from numpy.polynomial.polynomial import polyfit

def singularity_spectrum(lag: np.array, mfdfa: np.ndarray, q: np.array,
                        lim: list=[None, None], interpolate: int=False):
    """
    Extract the slopes of the fluctuation function to futher obtain the
    the singularity strength ``hq`` (or Hölder exponents) and singularity
    spectrum ``Dq`` (or fractal dimension).

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.

    interpolate: int (default False)
        Interpolates the ``q`` space to smoothed the singularity spectrum. Not
        yet implemented.

    Returns
    -------
    tau: np.array
        Scaling exponents ``tau``. A usually increasing function of ``q`` from
        which the fractality of the data can be determined by its shape. A truly
        linear tau indicates monofractality, whereas a curved one (usually
        curving around small ``q`` values) indicates multifractality.

    Notes
    -----
    .. versionadded:: 0.4.1

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.
    """

    # Calculate tau
    tau = scaling_exponents(lag, mfdfa, q, lim, interpolate)

    # Calculate hq, which needs tau
    hq = hurst_exponents(lag, mfdfa, q, lim, interpolate)

    # Calculate Dq, which needs tau and hq
    Dq = _Dq(tau, hq, q)

    return hq, Dq

def scaling_exponents(lag: np.array, mfdfa: np.ndarray, q: np.array,
                      lim: list=[None, None], interpolate: int=False):
    """
    Calculate the multifractal scaling exponents ``tau``.

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.

    interpolate: int (default False)
        Interpolates the ``q`` space to smoothed the singularity spectrum. Not
        yet implemented.

    Returns
    -------
    hq: np.array
        Singularity strength ``hq``. The width of this function indicates the
        strength of the multifractality. A width of ``max(hq) - min(hq) ≈ 0``
        means the data is monofractal.

    Dq: np.array
        Singularity spectrum ``Dq``. The location of the maximum of ``Dq`` (with
         ``hq`` as the abscissa) should be 1 and indicates the most prominent
         exponent in the data.

    Notes
    -----
    .. versionadded:: 0.4.1

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.
    """

    # Calculate the slopes
    slopes = _slopes(lag, mfdfa, q, lim, interpolate)

    return ( q * slopes ) - 1

def hurst_exponents(lag: np.array, mfdfa: np.ndarray, q: np.array,
                    lim: list=[None, None], interpolate: int=False):
    """
    Calculate the generalised Hurst exponents 'hq' from the

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default = [None, None])
        List of lower and upper lag limits. If none, the polynomial fittings
        include the full range of lag.

    interpolate: int (default False)
        Interpolates the ``q`` space to smoothed the singularity spectrum. Not
        yet implemented.

    Returns
    -------
    hq: np.array
        Singularity strength ``hq``. The width of this function indicates the
        strength of the multifractality. A width of ``max(hq) - min(hq) ≈ 0``
        means the data is monofractal.

    Notes
    -----
    .. versionadded:: 0.4.1

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica A,
        316(1-4), 87–114, 2002.
    """

    # Calculate tau
    tau = scaling_exponents(lag, mfdfa, q, lim, interpolate)

    return ( np.gradient(tau) / np.gradient(q) )

def _slopes(lag: np.array, mfdfa: np.ndarray, q: np.array,
            lim: list=[None, None], interpolate: int=False):
    """
    Extra the slopes of each q power obtained with MFDFA to later produce either
    the singularity spectrum or the multifractal exponents.

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype = float)

    # Ensure mfdfa has the same q-power entries as q
    if mfdfa.shape[1] != q.shape[0]:
        raise ValueError(
            "Fluctuation function and q powers don't match in dimension.")

    # Allocated array for slopes
    slopes = np.zeros(len(q))

    # Find slopes of each q-power
    for i in range(len(q)):
        slopes[i] = polyfit(np.log(lag[lim[0]:lim[1]]),
                            np.log(mfdfa[lim[0]:lim[1],i]),
                            1)[1]

    return slopes

def _Dq(tau, hq, q):
    """
    Calculate the singularity spectrum or fractal dimension ``Dq``.

    Notes
    -----
    .. versionadded:: 0.4.1
    """
    return q * hq - tau

def singularity_spectrum_plot(hq, Dq):
    """
    Plots the singularity spectrum.

    Parameters
    ----------
    hq: np.array
        Singularity strength ``hq`` as calculated with `singularity_spectrum`.

    Dq: np.array
        Singularity spectrum ``Dq`` as calculated with `singularity_spectrum`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    Notes
    -----
    .. versionadded:: 0.4.1
    """

    fig, ax = _plotter(hq, Dq)

    ax.set_ylabel(r'Dq')
    ax.set_xlabel(r'hq')

    return fig

def scaling_exponents_plot(q, tau):
    """
    Plots the scaling exponents, which is conventionally given with ``q`` in the
    abscissa and ``tau`` in the ordinates.

    Parameters
    ----------
    tau: np.array
        Scaling exponents ``tau`` as calculated with `scaling_exponents`.

    q: np.array
        Singularity spectrum ``Dq`` as calculated with `singularity_spectrum`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    fig, ax = _plotter(q, tau)

    ax.set_ylabel(r'tau')
    ax.set_xlabel(r'q')

    return fig

def hurst_exponents_plot(q, hq):
    """
    Plots the generalised Hurst exponents ``hq`` in the ordinates with ``q``
    in the abscissa.

    Parameters
    ----------
    tau: np.array
        Generalised Hurst coefficients ``hq`` as calculated with
        `hurst_exponents`.

    q: np.array
        Singularity spectrum ``Dq`` as calculated with `singularity_spectrum`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    fig, ax = _plotter(q, hq)

    ax.set_ylabel(r'hq')
    ax.set_xlabel(r'q')

    return fig

def _plotter(x, y):
    """
    Plot helper function.

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    # Check if matplotlib is installed
    _missing_library()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)

    ax.plot(x, y, lw = 2, color = 'black')

    fig.tight_layout()

    return fig, ax

def _missing_library():
    try:
        import matplotlib.pyplot as _plt
    except ImportError:
        raise ImportError(("matplotlib is required to do output the singularity"
                           " spectrum plot. Please install matplotlib with 'pip"
                           " install matplotlib' or another convinient way.")
                         )
