# This is based on Kantelhardt, J. W., Zschiegner, S. A., Koscielny-Bunde, E.,
# Havlin, S., Bunde, A., & Stanley, H. E., Multifractal detrended fluctuation
# analysis of nonstationary time series. Physica A, 316(1-4), 87-114, 2002 as
# well as on nolds (https://github.com/CSchoel/nolds) and on work by  Espen A.
# F. Ihlen, Introduction to multifractal detrended fluctuation analysis in
# Matlab, Front. Physiol., 2012, https://doi.org/10.3389/fphys.2012.00141

from typing import Tuple

import numpy as np
from numpy.polynomial.polynomial import polyfit

__all__ = [
    'singularity_spectrum',
    'scaling_exponents',
    'hurst_exponents',
    'singularity_spectrum_plot',
    'scaling_exponents_plot',
    'hurst_exponents_plot'
]


def singularity_spectrum(lag: np.array, mfdfa: np.ndarray, q: np.array,
                         lim: list = [False, False], interpolate: int = False
                         ) -> Tuple[np.array, np.array]:
    """
    Extract the slopes of the fluctuation function to further obtain the
    singularity strength `α` and singularity spectrum `f(α)`. It is important to
    note that `α` will be centred `>1` for most cases because of the increase in
    regularity in MFDFA.

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default `[int(lag.size // 1.5), int(lag.size // 8)]`)
        List of lower and upper lag limits. If you wish to consider the full
        range, use `None` to unbound the limits (lower or upper) and thus
        consider the full lag, e.g., `lim=[None, None]`.

    interpolate: int (default False)
        Interpolates the `q` space to smoothed the singularity spectrum. Not
        yet implemented.

    Returns
    -------
    alpha: np.array
        Singularity strength `α`. The width of this function indicates the
        strength of the multifractality. A width of `max(α) - min(α) ≈ 0`
        means the data is monofractal.

    f: np.array
        Singularity spectrum `f(α)`. The location of the maximum of `f(α)`
        (with `α` as the abscissa) should be 1 and indicates the most
        prominent fractal scale in the data.

    Notes
    -----
    .. versionadded:: 0.4.1

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica
        A, 316(1-4), 87–114, 2002.
    """

    # if no lower limit is given
    if lim[0] is False:
        lim[0] = int(lag.size // 8)

    # if no upper limit is given
    if lim[1] is False:
         lim[1] = int(lag.size // 1.5)

    # clean q
    q = _clean_q(q)

    # Calculate tau
    _, tau = scaling_exponents(lag, mfdfa, q, lim, interpolate)

    # Calculate α, which needs tau
    alpha = np.gradient(tau) / np.gradient(q)

    # Calculate Dq, which needs tau and q
    f = _falpha(tau, alpha, q)

    return alpha, f


def scaling_exponents(lag: np.array, mfdfa: np.ndarray, q: np.array,
                      lim: list = [False, False], interpolate: int = False
                      ) -> Tuple[np.array, np.array]:
    """
    Calculate the multifractal scaling exponents `τ(q)`, which is given by

    .. math::

       \tau(q) = qh(q) - 1.

    To evaluate the scaling exponent `τ(q)`, plot it vs `q`. If the
    relation between `τ(q)` is linear, the data is monofractal. If
    not, the data is multifractal. Note that these measures rarely match the
    theoretical expectation,  thus a variation of ± 0.25 is absolutely
    reasonable.

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default `[int(lag.size // 1.5), int(lag.size // 8)]`)
        List of lower and upper lag limits. If none, the polynomial fittings
        will be restrict to half the maximal lag and discard the first lag
        point.

    interpolate: int (default False)
        Interpolates the `q` space to smoothed the singularity spectrum. Not
        yet implemented.

    Returns
    -------
    q: np.array
        The `q` powers.

    tau: np.array
        Scaling exponents `τ(q)`. A usually increasing function of `q` from
        which the fractality of the data can be determined by its shape. A
        truly linear tau indicates monofractality, whereas a curved one
        (usually curving around small `q` values) indicates multifractality.


    Notes
    -----
    .. versionadded:: 0.4.1

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica
        A, 316(1-4), 87–114, 2002.
    """

    # if no lower limit is given
    if lim[0] is False:
        lim[0] = int(lag.size // 8)

    # if no upper limit is given
    if lim[1] is False:
         lim[1] = int(lag.size // 1.5)

    # clean q
    q = _clean_q(q)

    # Calculate the slopes
    slopes = _slopes(lag, mfdfa, q, lim, interpolate)

    return q, (q * slopes) - 1


def hurst_exponents(lag: np.array, mfdfa: np.ndarray, q: np.array,
                    lim: list = [False, False], interpolate: int = False
                    ) -> Tuple[np.array, np.array]:
    """
    Calculate the generalised Hurst exponents `h(q)` from MFDFA, which
    are simply the slopes of each DFA for various `q` values.

    Note that these measures rarely match the theoretical expectation,
    thus a variation of ± 0.25 is absolutely reasonable. It is important to
    note that `h(q)` will have values `>1` for most cases because of the
    increase in regularity in MFDFA.

    Parameters
    ----------
    lag: np.array of ints
        An array with the window sizes which where used in MFDFA.

    mdfda: np.ndarray
        Matrix of the fluctuation function from MFDFA

    q: np.array
        Fractal exponents used. Must be more than 2 points.

    lim: list (default `[int(lag.size // 1.5), int(lag.size // 8)]`)
        List of lower and upper lag limits. If you wish to consider the full
        range, use `None` to unbound the limits (lower or upper) and thus
        consider the full lag, e.g., `lim=[None, None]`.

    interpolate: int (default False)
        Interpolates the `q` space to smoothed the singularity spectrum. Not
        yet implemented.

    Returns
    -------
    q: np.array
        The `q` powers.

    hq: np.array
        Singularity strength `h(q)`. The width of this function indicates the
        strength of the multifractality. A width of `max(h(q)) - min(h(q)) ≈ 0`
        means the data is monofractal.

    Notes
    -----
    .. versionadded:: 0.4.1

    References
    ----------
    .. [Kantelhardt2002] J. W. Kantelhardt, S. A. Zschiegner, E.
        Koscielny-Bunde, S. Havlin, A. Bunde, H. E. Stanley. "Multifractal
        detrended fluctuation analysis of nonstationary time series." Physica
        A, 316(1-4), 87–114, 2002.
    """

    # if no lower limit is given
    if lim[0] is False:
        lim[0] = int(lag.size // 8)

    # if no upper limit is given
    if lim[1] is False:
         lim[1] = int(lag.size // 1.5)

    # clean q
    q = _clean_q(q)

    # Calculate the slopes
    hq = _slopes(lag, mfdfa, q, lim, interpolate)

    return q, hq


def _slopes(lag: np.array, mfdfa: np.ndarray, q: np.array,
            lim: list = [None, None], modified=True, interpolate: int = False
            ) -> np.array:
    """
    Extra the slopes of each `q` power obtained with MFDFA to later produce
    either the singularity spectrum or the multifractal exponents.

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    # if no lower limit is given
    if lim[0] is False:
        lim[0] = int(lag.size // 8)

    # if no upper limit is given
    if lim[1] is False:
         lim[1] = int(lag.size // 1.5)

    # clean q
    q = _clean_q(q)

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    # Ensure mfdfa has the same q-power entries as q
    if mfdfa.shape[1] != q.shape[0]:
        raise ValueError(
            "Fluctuation function and q powers don't match in dimension."
        )

    # Allocated array for slopes
    slopes = np.zeros(len(q))

    # Find slopes of each q-power
    for i in range(len(q)):
        slopes[i] = polyfit(np.log(lag[lim[0]:lim[1]]),
                            np.log(mfdfa[lim[0]:lim[1], i]),
                            1
                            )[1]

    return slopes


def _falpha(tau, alpha, q) -> np.array:
    """
    Calculate the singularity spectrum or fractal dimension `f(α)`.

    Notes
    -----
    .. versionadded:: 0.4.1
    """
    return q * alpha - tau


# Plotters


def singularity_spectrum_plot(alpha, f) -> np.array:
    """
    Plots the singularity spectrum.

    Parameters
    ----------
    alpha: np.array
        Singularity strength `α` as calculated with `singularity_spectrum`.

    f: np.array
        Singularity spectrum `f(α)` as calculated with `singularity_spectrum`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    ax: figure axes.
        Returns the axes of the figure.

    Notes
    -----
    .. versionadded:: 0.4.1
    """

    fig, ax = _plotter(alpha, f)

    ax.set_ylabel(r'f(α)')
    ax.set_xlabel(r'α')

    return fig, ax


def scaling_exponents_plot(q, tau) -> Tuple['plt.fig', 'plt.Axes']:
    """
    Plots the scaling exponents, which is conventionally given with `q` in the
    abscissa and `τ` in the ordinates.

    Parameters
    ----------
    q: np.array
        Singularity spectrum `f(α)` as calculated with `singularity_spectrum`.

    tau: np.array
        Scaling exponents `τ` as calculated with `scaling_exponents`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    ax: figure axes.
        Returns the axes of the figure.

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    fig, ax = _plotter(q, tau)

    ax.set_ylabel(r'tau')
    ax.set_xlabel(r'q')

    return fig, ax


def hurst_exponents_plot(q, hq) -> Tuple['plt.fig', 'plt.Axes']:
    """
    Plots the generalised Hurst exponents `h(q)` in the ordinates with `q`
    in the abscissa.

    Parameters
    ----------
    q: np.array
        Singularity spectrum `f(α)` as calculated with `singularity_spectrum`.

    hq: np.array
        Generalised Hurst coefficients `h(q)` as calculated with
        `hurst_exponents`.

    Returns
    -------
    fig: matplotlib fig
        Returns the figure, useful if one wishes to use fig.savefig(...).

    ax: figure axes.
        Returns the axes of the figure.

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    fig, ax = _plotter(q, hq)

    ax.set_ylabel(r'h(q)')
    ax.set_xlabel(r'q')

    return fig, ax


def _clean_q(q) -> np.array:

    # Fractal powers as floats
    q = np.asarray_chkfinite(q, dtype=float)

    # Ensure q≈0 is removed, since it does not converge. Limit set at |q| < 0.1
    q = q[(q < -.1) + (q > .1)]

    # Reshape q to perform np.float_power
    q = q.flatten()

    return q


def _plotter(x: np.array, y: np.array) -> Tuple['plt.fig', 'plt.Axes']:
    """
    Plot helper function.

    Notes
    -----
    .. versionadded:: 0.4.1

    """

    # Check if matplotlib is installed
    _missing_library()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)

    ax.plot(x, y, 'o', color='black')

    fig.tight_layout()

    return fig, ax


def _missing_library() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            ("'matplotlib' is required to output the singularity "
             "spectrum plots. Please install 'matplotlib'."
             )
        )

    return
