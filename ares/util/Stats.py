"""

Stats.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct 11 17:13:38 MDT 2014

Description: 

"""

import numpy as np
from scipy.optimize import fmin
from scipy.integrate import quad
from scipy.interpolate import griddata

def symmetrize_errors(mu, err, operation='min'):
    
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

def GaussND(x, mu, cov):
    """
    Return value of multi-variate Gaussian at point x (same shape as mu).
    """
    N = len(x)
    norm = 1. / np.sqrt((2. * np.pi)**N * np.linalg.det(cov))
    icov = np.linalg.inv(cov)    
    score = np.dot(np.dot((x - mu).T, icov), (x - mu))

    return norm * np.exp(-0.5 * score)

def get_nu(sigma, nu_in, nu_out):
    """
    Parameters
    ----------
    sigma : float
        1-D Gaussian error on some parameter.
    nu_in : float
        Percent of likelihood enclosed within given sigma.
    nu_out : float
        Percent of likelihood
    
    Example
    -------
    Convert 68% error-bar of 0.5 (arbitrary) to 95% error-bar:
    >>> get_nu(0.5, 0.68, 0.95)
    
    Returns
    -------
    sigma corresponding to nu_out.
    
    """
    
    if nu_in == nu_out:
        return sigma
    
    var_in = sigma**2
    
    # 1-D Gaussian with variable variance
    pdf = lambda x, vv: np.exp(-x**2 / 2. / vv)
        
    # Integral (relative to total) from -sigma to sigma
    # Allows us to convert input sigma to 1-D error-bar
    integral = lambda vr: quad(lambda x: pdf(x, vv=var_in), -sigma, sigma)[0] \
        / (np.sqrt(abs(var_in)) * np.sqrt(2 * np.pi))
    
    # Minimize difference between integral (as function of variance) and
    # desired area
    to_min = lambda y: abs(integral(y) - nu_in)
    
    # Input sigma converted to 
    var = fmin(to_min, 0.5 * sigma**2, disp=False)[0]
    
    pdf_1sigma = lambda x: np.exp(-x**2 / 2. / var)
    
    integral = lambda sigma: quad(lambda x: pdf_1sigma(x), 
        -sigma, sigma)[0] \
        / (np.sqrt(abs(var)) * np.sqrt(2 * np.pi))
    
    to_min = lambda y: abs(integral(y) - nu_out)
    
    return fmin(to_min, np.sqrt(var), disp=False)[0]

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
    
def rebin(bins):
    """
    Take in an array of bin edges and convert them to bin centers.        
    """

    bins = np.array(bins)
    result = np.zeros(bins.size - 1)
    for i, element in enumerate(result):
        result[i] = (bins[i] + bins[i + 1]) / 2.

    return result

