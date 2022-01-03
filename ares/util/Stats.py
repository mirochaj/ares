"""

Stats.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct 11 17:13:38 MDT 2014

Description:

"""

import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.interpolate import griddata
from scipy.stats import binned_statistic
from scipy.optimize import fmin, curve_fit

root2 = np.sqrt(2.)
rootpi = np.sqrt(np.pi)

_normal = lambda x, p0, p1, p2: p0 * np.exp(-(x - p1)**2 / 2. / p2**2)
_normal_skew = lambda x, p0, p1, p2, p3: 2 * _normal(x, p0, p1, p2) \
    * 0.5 * (1. + erf((x - p1) * p3 / root2 / p2))

def symmetrize_errors(mu, err, operation='min'):
    """
    Take asymmetric errorbars and return symmetrized version.
    """
    if type(err) not in [int, float]:
        err1 = err[0]
        err2 = err[1]
    else:
        err1 = err2 = err

    logphi_ML = mu
    logphi_lo_tmp = logphi_ML - err1   # log10 phi
    logphi_hi_tmp = logphi_ML + err2   # log10 phi

    phi_lo = 10**logphi_lo_tmp
    phi_hi = 10**logphi_hi_tmp

    err1 = 10**logphi_ML - phi_lo
    err2 = phi_hi - 10**logphi_ML

    if operation == 'min':
        return np.min([err1, err2])
    elif operation == 'max':
        return np.max([err1, err2])
    elif operation == 'mean':
        return np.mean([err1, err2])
    else:
        raise NotImplementedError('help')

def Gauss1D(x, pars):
    """
    Parameters
    ----------
    pars[0] : float
        Baseline / offset.
    pars[1] : float
        Amplitude / normalization.
    pars[2] : float
        Mean
    pars[3] : float
        Variance.
    """
    return pars[0] + pars[1] * np.exp(-(x - pars[2])**2 / 2. / pars[3])

def get_nu(sigma, nu_in, nu_out):

    if nu_in == nu_out:
        return sigma

    guess = nu_in / nu_out

    # Define a 1-D Gaussian with variable variance
    pdf = lambda x, s: np.exp(-x**2 / 2. / s**2) / np.sqrt(s**2 * 2 * np.pi)

    # Integral (relative to total) from -sigma to sigma
    integral = lambda s: quad(lambda x: 1e12 * pdf(x, s=s),
        -abs(sigma), abs(sigma), epsrel=1e-15, epsabs=1e-15)[0] / 1e12

    # Minimize difference between integral (as function of variance) and
    # specified area (i.e., supplied confidence interval).
    to_min = lambda y: abs(integral(y) - nu_in)

    # Solve above equation to obtain the 1-sigma error-bar.
    s1 = fmin(to_min, guess * sigma, disp=False)[0]

    # Define a new PDF using this 1-sigma error. Now, the unknown is
    # the interval over which we must integrate to obtain the
    # desired area, nu_out.
    f_s_out = lambda s_out: quad(lambda x: pdf(x, s=s1),
        -abs(s_out), abs(s_out))[0]

    to_min = lambda y: abs(f_s_out(y) - nu_out)

    s_out = fmin(to_min, guess * sigma, disp=False)[0]

    return s_out

def error_1D(x, y, nu=0.68, limit=None):
    """
    Compute 1-D (possibly asymmetric) errorbar for input PDF.

    Parameters
    ----------
    x : np.ndarray
        bins
    y : np.ndarray
        PDF
    nu : float
        Integrate out until nu-% of the likelihood has been enclosed.

    Example
    -------
    For a Gaussian, should recover 1, 2, and 3 sigma easily:
    >>> x = np.linspace(-20, 20, 10000)
    >>> y = np.exp(-x**2 / 2. / 1.)
    >>> print error_1D(x, y, 0.6827), error_1D(x, y, 0.9545), error_1D(x, y, 0.9973)

    Returns
    -------
    If tailed is None:
    Tuple containing (maximum likelihood value, (lower errorbar, upper errorbar)).

    """

    tot = np.sum(y)

    cdf = np.cumsum(y) / float(tot)

    if limit is None:
        percent = (1. - nu) / 2.
        p1, p2 = percent, 1. - percent

        x1 = np.interp(p1, cdf, x)
        x2 = np.interp(p2, cdf, x)
        xML = x[np.argmax(y)]

        return xML, (xML - x1, x2 - xML)
    elif limit == 'lower':
        return np.interp((1. - nu), cdf, x), (None, None)
    elif limit == 'upper':
        return np.interp(nu, cdf, x), (None, None)
    else:
        raise ValueError('Invalid input value for limit.')

def error_2D(x, y, z, bins, nu=[0.95, 0.68], weights=None, method='raw'):
    """
    Find 2-D contour given discrete samples of posterior distribution.

    Parameters
    ----------
    x : np.ndarray
        Array of samples in x.
    y : np.ndarray
        Array of samples in y.
    bins : np.ndarray, (2, Nsamples)

    method : str
        'raw', 'nearest', 'linear', 'cubic'


    """

    if method == 'raw':
        nu, levels = _error_2D_crude(z, nu=nu)
    else:

        # Interpolate onto new grid
        grid_x, grid_y = np.meshgrid(bins[0], bins[1])
        points = np.array([x, y]).T
        values = z

        grid = griddata(points, z, (grid_x, grid_y), method=method)

        # Mask out garbage points
        mask = np.zeros_like(grid, dtype='bool')
        mask[np.isinf(grid)] = 1
        mask[np.isnan(grid)] = 1
        grid[mask] = 0

        nu, levels = _error_2D_crude(grid, nu=nu)

    return nu, levels

def _error_2D_crude(L, nu=[0.95, 0.68]):
    """
    Integrate outward at "constant water level" to determine proper
    2-D marginalized confidence regions.

    ..note:: This is fairly crude -- the "coarse-ness" of the resulting
        PDFs will depend a lot on the binning.

    Parameters
    ----------
    L : np.ndarray
        Grid of likelihoods.
    nu : float, list
        Confidence intervals of interest.

    Returns
    -------
    List of contour values (relative to maximum likelihood) corresponding
    to the confidence region bounds specified in the "nu" parameter,
    in order of decreasing nu.
    """

    if type(nu) in [int, float]:
        nu = np.array([nu])

    # Put nu-values in ascending order
    if not np.all(np.diff(nu) > 0):
        nu = nu[-1::-1]

    peak = float(L.max())
    tot = float(L.sum())

    # Counts per bin in descending order
    Ldesc = np.sort(L.ravel())[-1::-1]

    Lencl_prev = 0.0

    # Will correspond to whatever contour we're on
    j = 0

    # Some preliminaries
    contours = [1.0]
    Lencl_running = []

    # Iterate from high likelihood to low
    for i in range(1, Ldesc.size):

        # How much area (fractional) is contained in bins at or above the current level?
        Lencl_now = L[L >= Ldesc[i]].sum() / tot

        # Keep running list of enclosed (integrated) likelihoods
        Lencl_running.append(Lencl_now)

        # What contour are we on?
        Lnow = Ldesc[i]

        # Haven't hit next contour yet
        if Lencl_now < nu[j]:
            pass
        # Just passed a contour
        else:

            # Interpolate to find contour more precisely
            Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                [Ldesc[i-1], Ldesc[i]])

            # Save relative to peak
            contours.append(Linterp / peak)

            j += 1

            if j == len(nu):
                break

        Lencl_prev = Lencl_now

    # Return values that match up to inputs
    return nu[-1::-1], np.array(contours[-1::-1])

def correlation_matrix(cov):
    """
    Compute correlation matrix.

    Parameters
    ----------
    x : list
        Each element is an array of data for that dimension.
    mu : list
        List of mean values in each dimension.

    """

    rho = np.zeros_like(cov)
    N = rho.shape[0]
    for i in range(N):
        for j in range(N):
            rho[i,j] = cov[i,j] / np.sqrt(cov[i,i] * cov[j,j])

    return rho

def bin_e2c(bins):
    """
    Convert bin edges to bin centers.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return 0.5 * (bins[1:] + bins[:-1])

def bin_c2e(bins):
    """
    Convert bin centers to bin edges.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return np.concatenate(([bins[0] - 0.5 * dx], bins + 0.5 * dx))

def skewness(data):
    return np.sum((data - np.mean(data))**3) \
        / float(data.size) / np.std(data)**3

def kurtosis(data):
    return np.sum((data - np.mean(data))**4) \
        / float(data.size) / np.std(data)**4

_dist_opts = ['normal', 'lognormal', 'skewnormal',
    'normal-pars', 'lognormal-pars', 'skewnormal-pars', 'pdf', 'cdf']

def quantify_scatter(x, y, xbin_c, weights=None, inclusive=False,
    method_avg='avg', method_std='std', cdf_lim=0.7, pdf_bins=50):
    """
    Quantify the scatter in some relationship between two variables, x and y.

    Parameters
    ----------
    x : np.ndarray
        Independent variable
    y : np.array
        Dependent variable
    xbin_c : np.ndarray
        Bin centers for `x`
    weights : np.ndarray
        Can weight each samples by some factor.
    inclusive : bool
        Include samples above or below bounding bins in said bins.
    method_avg : str
        How to quantify the average y value in each x bin.
        Options: 'avg', 'median', 'mode'
    method_std : str, float
        How to quantify the spread in y in each x bin.
        Options: 'std', 'normal', 'lognormal', 'bounds', 'pdf', float
        If a float is provided, assume it's a percentile, e.g.,
        method_std=0.68 will return boundaries of region containing 68% of
        samples.
    cdf_lim : float
        If fitting a normal or log-normal function to the distribution in each
        bin, include the PDF up until this value of the CDF when fitting the
        distribution. Essentially a kludge to exclude long tails in fit, to
        better capture peak and width of (main part of) the distribution.

    """

    xbin_e = bin_c2e(xbin_c)

    if weights is None:
        have_weights = False
    else:
        if np.all(np.diff(np.diff(weights)) == 0):
            have_weights = False
        else:
            have_weights = True

    if not have_weights:

        if method_std in ['std', 'sum'] and method_avg == 'avg':

            print("Deferring to scipy.stats.binned_statistic since weights=None.")

            yavg, _b, binid = binned_statistic(x, y, statistic='mean',
                bins=xbin_e)
            ysca, _b, binid = binned_statistic(x, y, statistic=method_std,
                bins=xbin_e)
            N, _b, binid = binned_statistic(x, y, statistic='count',
                bins=xbin_e)

            return xbin_c, yavg, ysca, N

    ysca = []
    yavg = []
    N = []
    for i, lo in enumerate(xbin_e):
        if i == len(xbin_e) - 1:
            break

        # Upper edge of bin
        hi = xbin_e[i+1]

        if inclusive and i == 0:
            ok = x < hi
            ok = np.logical_and(ok, np.isfinite(y))
        elif inclusive and i == len(xbin_e) - 1:
            ok = x >= lo
            ok = np.logical_and(ok, np.isfinite(y))
        else:
            ok = np.logical_and(x >= lo, x < hi)
            ok = np.logical_and(ok, np.isfinite(y))

        f = y[ok==1]

        # What to do when there aren't any samples in a bin?
        # Move on, that's what. Add masked elements.
        if (f.size == 0) or (weights[ok==1].sum() == 0):

            yavg.append(-np.inf)
            if method_std == 'bounds' or type(method_std) in [int, float, np.float64]:
                ysca.append((-np.inf, -np.inf))
            elif type(method_std) in [list, tuple]:
                ysca.append((-np.inf, -np.inf))
            else:
                ysca.append(-np.inf)

            N.append(0)
            continue

        # If we made it here, we've got some samples.
        # Record the number of samples in this bin, the average value,
        # and some measure of the scatter.
        N.append(sum(ok==1))

        if method_avg == 'avg':
            yavg.append(np.average(f, weights=weights[ok==1]))
        elif method_avg == 'median':
            # Weighted median -> use cdf of weights, convert to y-value after.
            # First: rank-order in y value.
            ix = np.argsort(f)
            cdf = np.cumsum(weights[ok==1][ix]) / np.sum(weights[ok==1][ix])
            yavg.append(f[ix][np.argmin(np.abs(cdf - 0.5))])
        else:
            raise NotImplemented("Haven't implemented method_avg={} in this case!".format(method_avg))

        if method_std == 'std':
            ysca.append(np.std(f))
        elif method_std == 'sum':
            ysca.append(np.sum(f * weights[ok==1]))
        elif method_std in _dist_opts:

            if method_std.startswith('lognormal'):
                pdf, ye = np.histogram(np.log10(y[ok==1]), density=1,
                    weights=weights[ok==1], bins=pdf_bins)
            else:
                pdf, ye = np.histogram(y[ok==1], density=1,
                    weights=weights[ok==1], bins=pdf_bins)

            yc = bin_e2c(ye)

            if method_std == 'pdf':
                ysca.append((yc, pdf))
                continue

            cdf = np.cumsum(pdf) / np.sum(pdf)

            if method_std == 'cdf':
                ysca.append((yc, cdf))
                continue

            # Compute median to use as initial guess
            med = np.interp(0.5, cdf, yc)
            std = np.nanstd(y[ok==1]) if method_std.startswith('norm') \
                else np.nanstd(np.log10(y[ok==1]))

            # Make sure we go a little past the peak in the fit.
            yc_fit = yc[cdf <= cdf_lim]
            pdf_fit = pdf[cdf <= cdf_lim]

            # If CDF very sharp, just use all of it.
            if len(pdf_fit) < 3 + int('skewnormal' in method_std):
                yc_fit = yc
                pdf_fit = pdf

            if 'skew' in method_std:
                _model = _normal_skew
                p0 = [pdf.max(), med, std, 1.1]
            else:
                _model = _normal
                p0 = [pdf.max(), med, std]

            print("guesses: {}".format(p0))

            try:
                pval, pcov = curve_fit(_model, yc_fit, pdf_fit,
                    p0=p0, maxfev=100000)
            except RuntimeError:
                print("Gaussian fit failed!")
                pval = [-np.inf] * (3 + int('skewnormal' in method_std))

            if '-pars' in method_std:
                ysca.append(pval)
            else:
                ysca.append(pval[2])

        elif method_std == 'bounds':
            ysca.append((np.min(f), np.max(f)))
        elif type(method_std) in [int, float, np.float64]:
            q1 = 0.5 * 100 * (1. - method_std)
            q2 = 100 * method_std + q1
            lo, hi = np.percentile(f, (q1, q2))
            ysca.append((lo, hi))
        elif type(method_std) in [list, tuple]:
            q1 = 100 * method_std[0]
            q2 = 100 * method_std[1]
            lo, hi = np.percentile(f, (q1, q2))
            ysca.append((lo, hi))
        else:
            raise NotImplemented('help')

    return np.array(xbin_c), np.array(yavg), np.array(ysca), np.array(N)

def bin_samples(x, y, xbin_c, weights=None, limits=False, percentile=None,
    return_N=False, inclusive=False):
    """
    Bin samples in x and y.

    This is for backward compatibility. It's just a wrapper around
    `quantify_scatter` defined above.
    """

    if limits:
        return quantify_scatter(x, y, xbin_c, weights=weights,
            method_std='bounds', inclusive=inclusive)
    elif percentile is not None:
        a, b, c, d = quantify_scatter(x, y, xbin_c, weights=weights,
            method_std=percentile, inclusive=inclusive)

        if not return_N:
            return a, b, c
        else:
            return a, b, c, d
    else:
        return quantify_scatter(x, y, xbin_c, weights=weights,
            method_std='std', inclusive=inclusive)
