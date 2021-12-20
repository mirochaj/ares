"""

test_util_stats.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Tue 24 Mar 2020 22:11:31 EDT

Description:

"""

import numpy as np
from ares.util.Stats import Gauss1D, error_1D, error_2D, bin_e2c, bin_c2e, \
    quantify_scatter, bin_samples, symmetrize_errors, get_nu, \
    skewness, kurtosis, correlation_matrix

def test(tol=1e-1):

    # Make unit Gaussian, test that we recover 68% confidence region
    x = np.arange(-5, 5, 0.001)
    y = Gauss1D(x, pars=[0, 1., 0., 1.])
    assert y.max() == 1

    mu, (lo, hi) = error_1D(x, y)
    assert abs(mu) < tol
    assert abs(1. - lo) < tol
    assert abs(1. - hi) < tol

    # Make some bins
    xe = np.arange(0, 1, 0.05)
    xc = bin_e2c(xe)
    assert np.allclose(xc, xe[0:-1] + 0.025, rtol=tol)
    assert np.allclose(xe, bin_c2e(xc))

    # Make some fake data to test scatter analysis on.
    y = d = x + 2 + np.random.normal(scale=0.3, size=x.size)
    xb = x[0:-1:100]
    _xb, yb, s, N = quantify_scatter(x, y, xb)

    slope = np.mean(np.diff(yb) / np.diff(xb))

    assert abs(np.mean(s) - 0.3) < tol, "Something wrong with scatter."
    assert abs(slope - 1.) < tol, "Not recovering mean slope."

    # Test quantify_scatter with weights
    w = np.random.rand(y.size)
    _xb2, yb2, s2, N2 = quantify_scatter(x, y, xb, weights=w)

    # Backward compatibility
    _xb3, yb3, s3, N3 = bin_samples(x, y, xb)
    assert np.allclose(yb, yb3), "Problem w/ bin_samples."

    # Other random stuff
    y = np.random.rand(100)
    err = np.random.rand(200).reshape(2, 100)

    err_mi = symmetrize_errors(y, err, operation='min')
    err_ma = symmetrize_errors(y, err, operation='max')
    err_me = symmetrize_errors(y, err, operation='mean')

    # Error bars
    nu_out = get_nu(1., 0.68, 0.68)
    assert nu_out == 1

    nu_out = get_nu(1., 0.68, 0.95)
    assert abs(nu_out - 2) < 1e-1, nu_out

    # 2-D contours
    x = np.random.normal(size=1000)
    y = np.random.normal(size=1000)
    bins = np.arange(0, 1, 0.05)
    h, xedge, yedge = np.histogram2d(x, y, bins)

    err2d = error_2D(x, y, h, bins, nu=[0.95, 0.68], method='raw')

    sk = skewness(d)
    ku = kurtosis(d)
    corr = correlation_matrix(np.diag(np.random.rand(10)))

if __name__ == '__main__':
    test()
