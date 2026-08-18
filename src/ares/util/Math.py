"""

Math.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:52:13 MDT 2014

Description:

"""

import numpy as np
from ..physics.Constants import nu_0_mhz
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d as interp1d_scipy

try:
    from mcfit import P2xi, xi2P

    import warnings
    warnings.filterwarnings("ignore",
        message="The default value of lowring has been changed to False, ")

    have_mcfit = True
except ImportError:
    have_mcfit = False


_numpy_kwargs = {'left': None, 'right': None}

def integrate_with_subgrid_interp(x, y, bound_lo, bound_hi, method='trapz_trapz',
    brute_force_for_single_pt=False, axis=-1):
    """
    Sometimes doing numpy.trapezoid isn't good enough.

    In particular, if the grid is relatively coarse *and* we're not 
    integrating over the full domain, and we just associate the integration
    bounds with the nearest grid points, we can accrue noticeable numerical
    error.

    This routine is designed to help.

    Basically, we treat the 'bulk' of the integral using the trapezoidal rule
    and then correct for the extra area under the curve that is off of our
    grid points.
    
    Parameters
    ----------
    x : np.ndarray 
        Array of x values.
    y : np.ndarray
        Function values corresponding to given x values. Note: this can 
        be more than 1-D, see `axis` parameter.
    bound_lo : int, float
        Lower limit of integration
    bound_hi : int, float
        Upper limit of integration.
    method : str 
        Current options: 'trapz_nn', 'trapz_cubic', 'quad_cubic'
    brute_force_for_single_pt : bool
        In the rare case where bound_lo and bound_hi are both in the same grid 
        point, just fit the full curve with a cubic spline and integrate using
        quad. The alternative(s) can be very inaccurate.

    Returns 
    -------
    Integral of function y(x) between `bound_lo` and `bound_hi`.
    """

    ##
    # Automatically flip arrays and/or bounds
    if bound_lo > bound_hi:
        lo = bound_hi * 1.
        bound_hi = bound_lo * 1.
        bound_lo = lo
    
    if not np.all(np.diff(x) > 0):
        assert axis == 0, "Need to generalize for axis>0!"
        x = x[-1::-1]
        y = y[-1::-1]

        assert np.all(np.diff(x) > 0), "Non-monotonic x values!"
        
    # We don't do extrapolation here    
    assert bound_lo >= x.min(), \
            f"Hey! bound_lo={bound_lo:.3e}, must be >= min(x)={x.min()}"
    assert bound_hi <= x.max(), \
            f"Hey! bound_hi={bound_hi:.3e}, must be <= max(x)={x.max()}"

    ##
    # Otherwise, we're keeping it simpler/faster.
    i_lo = np.argmin(np.abs(x - bound_lo))
    i_hi = np.argmin(np.abs(x - bound_hi))

    # We force the bounding indices to contain `bound_lo` and `bound_hi`.
    # So, our corrections *remove* extra bits of the integrand.
    # Nothing special here, just a convention.
    if (x[i_lo] > bound_lo) and i_lo >= 1:
        i_lo -= 1
    if (x[i_hi] < bound_hi) and i_hi < (len(x) - 1):
        i_hi += 1

    ##
    # Brute force. Setup interpolant and then integrate using quad.
    if (method == 'cubic_quad') or ((i_lo == i_hi) and brute_force_for_single_pt):

        if y.ndim > 1:
            raise NotImplemented('help')
        else:
            f = interp1d(x, y, kind='cubic', axis=axis)
            return quad(f, bound_lo, bound_hi)[0]

    ##
    # Special case: both points in same grid point.
    if (i_lo == i_hi):
        # In this case, just doing a rectangular integration as 
        # we have no knowledge of broader function shape.
        if method == 'trapz_nn':
            return y[i_lo] * (bound_hi - bound_lo)
        else:
            pass
            # Will get dealt with below.

    # Recall: i_lo and i_hi extend past our integration bounds.
    # First we'll compute this integral
    if method.startswith('trapz'):
        full = np.trapezoid(y[i_lo:i_hi+1], x=x[i_lo:i_hi+1], axis=axis)
    elif method.startswith('simps'):
        full = simpson(y[i_lo:i_hi+1], x=x[i_lo:i_hi+1], axis=axis)

    # Equivalent to 'nn' correction
    if '_' not in method:
        return full

    # From here on its all about how we correct for the extra little 
    # slivers of the integral we need to remove. We're calling these 
    # the 'ears' because they hang off the sides.
    corr_method = method.split('_')[-1]

    ##
    # General case.
    if corr_method == 'nn':
        return full
    elif corr_method == 'trapz':
        # Recall: i_lo and i_hi extend past our integration bounds.
        # First we'll compute this integral

        # Trapezoid of left 'ear'
        dydx_l = (y[i_lo+1] - y[i_lo]) / (x[i_lo+1] - x[i_lo])
        y_bound_lo = y[i_lo] + (bound_lo - x[i_lo]) * dydx_l
        rect_l = (bound_lo - x[i_lo]) * y_bound_lo
        corr_l = rect_l + 0.5 * (bound_lo - x[i_lo]) * (y[i_lo] - y_bound_lo)

        # Trapezoid of right 'ear'
        dydx_r = (y[i_hi] - y[i_hi-1]) / (x[i_hi] - x[i_hi-1])
        y_bound_hi = y[i_hi-1] + (bound_hi - x[i_hi-1]) * dydx_r
        rect_r = (x[i_hi] - bound_hi) * y_bound_hi
        corr_r = rect_r + 0.5 * (x[i_hi] - bound_hi) * (y[i_hi] - y_bound_hi)
    
        # Subtract of ears and we're done.
        return full - corr_l - corr_r
    else:
        raise NotImplementedError('help')


def interp1d(x, y, kind='linear', fill_value=0.0, bounds_error=False,
    force_scipy=False, **kwargs):

    if 'axis' in kwargs:
        force_scipy = True

    if (kind == 'linear') and (not force_scipy):
        kw = _numpy_kwargs.copy()
        for kwarg in ['left', 'right']:
            if kwarg in kwargs:
                kw[kwarg] = kwargs[kwarg]
            else:
                kw[kwarg] = fill_value

        return lambda xx: np.interp(xx, x, y, **kw)
    elif (kind == 'cubic') or force_scipy:
        for kwarg in ['left', 'right']:
            if kwarg in kwargs:
                del kwargs[kwarg]
        return interp1d_scipy(x, y, kind='cubic', bounds_error=bounds_error,
            fill_value=fill_value, **kwargs)
    else:
        raise NotImplemented("Don\'t understand interpolation method={}".format(method))

class interp1d_wrapper(object):
    """
    Wrap interpolant and use boundaries as floor and ceiling.
    """
    def __init__(self, x, y, kind):
        self._x = x
        self._y = y
        self._interp = interp1d(x, y, kind=kind, bounds_error=False)

        self.limits = self._x.min(), self._x.max()

    def __call__(self, xin):

        if type(xin) in [int, float, np.float64]:
            if xin < self.limits[0]:
                x = self.limits[0]
            elif xin > self.limits[1]:
                x = self.limits[1]
            else:
                x = xin
        else:
            x = xin.copy()
            x[x < self.limits[0]] = self.limits[0]
            x[x > self.limits[1]] = self.limits[1]

        return self._interp(x)

def forward_difference(x, y):
    """
    Compute the derivative of y with respect to x via forward difference.

    Parameters
    ----------
    x : np.ndarray
        Array of x values
    y : np.ndarray
        Array of y values

    Returns
    -------
    Tuple containing x values and corresponding y derivatives.

    """

    return x[0:-1], (np.roll(y, -1) - y)[0:-1] / np.diff(x)

def central_difference(x, y, keep_size=False):
    """
    Compute the derivative of y with respect to x via central difference.

    Parameters
    ----------
    x : np.ndarray
        Array of x values
    y : np.ndarray
        Array of y values

    Returns
    -------
    Tuple containing x values and corresponding y derivatives.

    """

    dydx = ((np.roll(y, -1) - np.roll(y, 1)) \
        / (np.roll(x, -1) - np.roll(x, 1)))

    if keep_size:
        xout = x
        yout = dydx.copy()
        #
        yout[0] = (y[1] - y[0]) / (x[1] - x[0])
        yout[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    else:
        xout = x[1:-1]
        yout = dydx[1:-1]

    return xout, yout

def smooth(y, width, kernel='boxcar'):
    """
    Smooth 1-D function `y` using boxcar of width `kernel` (in pixels).

    Kernel options: 'boxcar' and 'gaussian'
    """

    assert width % 2 == 1

    s = width - 1
    kern = np.zeros_like(y)

    if kernel == 'boxcar':
        kern[kern.size//2 - s//2: kern.size//2 + s//2+1] = \
            np.ones(width) / float(width)
    elif kernel == 'gaussian':
        x0 = kern.size // 2
        xx = np.arange(0, len(y))
        kern = np.exp(-0.5 * (xx - x0)**2 / width**2) / width / np.sqrt(2 * np.pi)
    else:
        raise NotImplemented('help')

    # Chop off regions within boxcar size of edges
    result = np.convolve(y, kern, mode='same')
    result[0:width] = y[0:width]
    result[-width:] = y[-width:]

    return result

class LinearNDInterpolator(object):
    def __init__(self, axes, data, fill_values=None):
        """
        Create linear interpolation object.

        Parameters
        ----------
        axes : list
            Coordinates of sampled points for each axis of data.
        data : np.ndarray
            Sampled values.
        fill_values : float, list
            Values to return for coordinates outside the table. By default,
            will return values at the table boundaries.

        Example
        -------
        Interpolate in 1D:
        >>> x = np.arange(10)
        >>> y = x**2
        >>> interp = LinearNDInterpolator(x, y)
        >>> interp(5.4)

        Interpolate in 2D:
        >>> x = y = np.arange(10)
        >>> xx, yy = np.meshgrid(x, y)
        >>> z = xx**2 + yy**2
        >>> interp = LinearNDInterpolator([x, y], z)
        >>> interp([5.4, 5.9])

        Interpolate in 3D:
        >>> x = y = z = np.arange(10)
        >>> xx, yy, zz = np.meshgrid(x, y, z)
        >>> w = xx**2 + yy**2 + zz**2
        >>> interp = LinearNDInterpolator([x, y, z], w)
        >>> interp([5.4, 5.9, 7.1])

        """

        self.axes = np.array(axes)
        self.data = data

        self.dims = self.data.shape

        if len(self.axes.squeeze().shape) == 1:
            self.Nd = 1
        else:
            self.Nd = len(self.axes)

        if self.Nd == 1:
            self._init_1d()
        elif self.Nd >= 2:
            self._init_Nd()
        else:
            raise NotImplemented('Haven\'t implemented interpolation for N>3')

    def __call__(self, points):
        """
        Interpolate!

        Parameters
        ----------
        points : float, np.ndarray
            Can only be a float if we're interpolating in 1D.
            Otherwise, must be an array with ND elements.
        """

        if self.Nd == 1:
            return self._interp_1d(points)
        elif self.Nd == 2:
            return self._interp_2d(points)
        elif self.Nd == 3:
            return self._interp_3d(points)
        else:
            raise NotImplemented('Haven\'t implemented interpolation for N>3')

    def _init_1d(self):
        """
        Setup arrays for 1D interpolation.

        Use numpy, but do something numpy doesn't do, which is check to make
        sure x-values are ascending.
        """
        if np.all(np.diff(self.axes) > 0):
            return

        self.axes = self.axes[0,-1::-1]
        self.data = self.data[-1::-1]

    def _init_Nd(self):
        self.daxes = np.array([np.diff(axis) for axis in self.axes])
        self.axes_min = np.array([np.min(axis) for axis in self.axes])

        tmp = np.zeros(self.Nd)
        for i in range(self.Nd):
            if not np.allclose(self.daxes[i] - self.daxes[i][0],
                np.zeros_like(self.daxes[i])):
                raise ValueError('Values must be evenly spaced!')
            tmp[i] = self.daxes[i][0]

        self.daxes = tmp.copy()

    def _interp_1d(self, points):
        """ Interpolate using numpy for one-dimensional case. """

        return np.interp(points, self.axes, self.data)

    def _interp_2d(self, points):
        """ Interpolate in 2D. """

        i_n = np.digitize(points[0], self.axes[0])
        i_m = np.digitize(points[1], self.axes[1])

        x1 = self.axes[0][i_n]
        x2 = self.axes[0][i_n+1]
        y1 = self.axes[1][i_m]
        y2 = self.axes[1][i_m+1]

        f11 = self.data[i_n][i_m]
        f21 = self.data[i_n+1][i_m]
        f12 = self.data[i_n][i_m+1]
        f22 = self.data[i_n+1][i_m+1]

        final = (f11 * (x2 - points[0]) * (y2 - points[1]) + \
            f21 * (points[0] - x1) * (y2 - points[1]) + \
            f12 * (x2 - points[0]) * (points[1] - y1) + \
            f22 * (points[0] - x1) * (points[1] - y1)) / (x2 - x1) / (y2 - y1)

        return final

    def _get_indices_3d(self, points):
        # Smaller indices
        i_s = np.digitize(points[0], self.axes[0])
        j_s = np.digitize(points[1], self.axes[1])
        k_s = np.digitize(points[2], self.axes[2])

        # Bracketing coordinates
        if i_s < 0:
            i_s = i_b = 0
        elif i_s >= (self.dims[0] - 1):
            i_s = i_b = -1
        else:
            i_b = i_s + 1
        if j_s < 0:
            j_s = j_b = 0
        elif j_s >= (self.dims[1] - 1):
            j_s = j_b = -1
        else:
            j_b = j_s + 1
        if k_s < 0:
            k_s = k_b = 0
        elif k_s >= (self.dims[2] - 1):
            k_s = k_b = -1
        else:
            k_b = k_s + 1

        # Bracketing values
        x_s, y_s, z_s = self.axes[0,i_s], self.axes[1,j_s], self.axes[2,k_s]
        x_b, y_b, z_b = self.axes[0,i_b], self.axes[1,j_b], self.axes[2,k_b]

        # Distance between supplied value and smallest value in table
        x_d = (points[0] - x_s) / self.daxes[0]
        y_d = (points[1] - y_s) / self.daxes[1]
        z_d = (points[2] - z_s) / self.daxes[2]

        return [i_s, j_s, k_s], [i_b, j_b, k_b], [x_d, y_d, z_d]

    def _interp_3d(self, points):
        """ Interpolate in 3D. """

        ijk_s, ijk_b, xyz_d = self._get_indices_3d(points)

        i_s, j_s, k_s = ijk_s
        i_b, j_b, k_b = ijk_b
        x_d, y_d, z_d = xyz_d

        i1 = self.data[i_s,j_s,k_s] * (1. - z_d) + self.data[i_s,j_s,k_b] * z_d
        i2 = self.data[i_s,j_b,k_s] * (1. - z_d) + self.data[i_s,j_b,k_b] * z_d

        j1 = self.data[i_b,j_s,k_s] * (1. - z_d) + self.data[i_b,j_s,k_b] * z_d
        j2 = self.data[i_b,j_b,k_s] * (1. - z_d) + self.data[i_b,j_b,k_b] * z_d

        w1 = i1 * (1. - y_d) + i2 * y_d
        w2 = j1 * (1. - y_d) + j2 * y_d

        final = w1 * (1. - x_d) + w2 * x_d

        return final


def get_cf_from_ps_tab(k, ps, **kwargs):
    assert have_mcfit, "Must install mcfit! See `use_mcfit` parameter."

    cf_func = P2xi(k, **kwargs)
    R, cf = cf_func(ps, extrap=True)

    if R[1] < R[0]:
        return R[-1::-1], cf[-1::-1]
    else:
        return R, cf

def get_ps_from_cf_tab(R, cf, **kwargs):
    assert have_mcfit, "Must install mcfit! See `use_mcfit` parameter."

    ps_func = xi2P(R, **kwargs)
    k, ps = ps_func(cf, extrap=True)

    if k[1] < k[0]:
        return k[-1::-1], ps[-1::-1]
    else:
        return k, ps

def get_cf_from_ps_func(R, f_ps, kmin=1e-4, kmax=5000., rtol=1e-5, atol=1e-5):
    cf = np.zeros_like(R)
    for i, RR in enumerate(R):

        # Split the integral into an easy part and a hard part
        kcrit = 1. / RR

        # Re-normalize integrand to help integration
        norm = 1. / f_ps(kmax)

        # Leave sin(k*R) out -- that's the 'weight' for scipy.
        integrand = lambda kk: norm * 4 * np.pi * kk**2 * f_ps(kk) / kk / RR
        integrand_full = lambda kk: integrand(kk) * np.sin(kk * RR)

        # Do the easy part of the integral
        cf[i] = quad(integrand_full, kmin, kcrit,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1)[0] / norm

        # Do the hard part of the integral using Clenshaw-Curtis integration
        cf[i] += quad(integrand, kcrit, kmax,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1,
            weight='sin', wvar=RR)[0] / norm

    # Our FT convention
    cf /= (2 * np.pi)**3

    return cf

def get_ps_from_cf_func(k, f_cf, Rmin=1e-2, Rmax=1e3, rtol=1e-5, atol=1e-5):

    ps = np.zeros_like(k)
    for i, kk in enumerate(k):

        # Split the integral into an easy part and a hard part
        Rcrit = 1. / kk

        # Re-normalize integrand to help integration
        norm = 1. / f_cf(Rmax)

        # Leave sin(k*R) out -- that's the 'weight' for scipy.
        integrand = lambda RR: norm * 4 * np.pi * RR**2 * f_cf(RR) / kk / RR
        integrand_full = lambda RR: integrand(RR) * np.sin(kk * RR)

        # Do the easy part of the integral
        ps[i] = quad(integrand_full, Rmin, Rcrit,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1)[0] / norm

        # Do the hard part of the integral using Clenshaw-Curtis integration
        ps[i] += quad(integrand, Rcrit, Rmax,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1,
            weight='sin', wvar=kk)[0] / norm

    return ps


# Backward compatibility
get_cf_from_ps = get_cf_from_ps_func
get_ps_from_cf = get_ps_from_cf_func
