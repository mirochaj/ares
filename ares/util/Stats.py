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
    
    # 1-D Gaussian with variable variance
    pdf = lambda x, var: np.exp(-x**2 / 2. / var)
        
    # Integral (relative to total) from -sigma to sigma
    # Allows us to convert input sigma to 1-D error-bar
    integral = lambda var: quad(lambda x: pdf(x, var=var), -sigma, sigma)[0] \
        / (np.sqrt(abs(var)) * np.sqrt(2 * np.pi))
    
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

def error_1D(x, y, nu=0.68, discrete=True):
    """
    Compute 1D (possibly asymmetric) errorbar.

    If the number of samples in x and y is small, this will not be very
    accurate.

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
    Errorbar (asymmetric).

    """    

    if discrete:
        tot = float(np.sum(y))
    else:
        tot = np.trapz(y, x)

    # Number of elements in histogram
    K = len(y)

    # Maximum likelihood point    
    iML = np.argmax(y)

    # Start just right of the peak
    k = iML+1

    # Begin
    area = 0.0
    while (area < nu) and (k < K):

        Lr = y[k]  # likelihood at right point

        # Determine location of point left of peak with same likelihood
        if discrete:
            l = np.argmin(np.abs(y[0:iML-1] - Lr))
        else:
            xl = np.interp(Lr, y[0:iML-1], x[0:iML-1])
            l = int(np.argmin(np.abs(xl - x)))

        if discrete:
            area = np.sum(y[l:k+1]) / tot
        else:
            area = np.trapz(y[l:k+1] / tot, x[l:k+1])

        k += 1

    k -= 1

    return x[iML] - x[l], x[k] - x[iML]    

def rebin(bins):
    """
    Take in an array of bin edges and convert them to bin centers.        
    """

    bins = np.array(bins)
    result = np.zeros(bins.size - 1)
    for i, element in enumerate(result):
        result[i] = (bins[i] + bins[i + 1]) / 2.

    return result

