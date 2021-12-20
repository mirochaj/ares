"""

Math.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:52:13 MDT 2014

Description:

"""

import numpy as np
from ..physics.Constants import nu_0_mhz
from scipy.interpolate import interp1d as interp1d_scipy

_numpy_kwargs = {'left': None, 'right': None}

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

def central_difference(x, y):
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
        / (np.roll(x, -1) - np.roll(x, 1)))[1:-1]

    return x[1:-1], dydx

def five_pt_stencil(x, y):
    """
    Compute the first derivative of y wrt x using five point method.
    """

    h = abs(np.diff(x)[0])

    num = -np.roll(y, -2) + 8. * np.roll(y, -1) \
          - 8. * np.roll(y, 1) + np.roll(y, 2)

    return x[2:-2], num[2:-2] / 12. / h

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
