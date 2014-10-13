"""

Math.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:52:13 MDT 2014

Description: 

"""

import numpy as np
from scipy.interpolate import interp1d
from ..physics.Constants import nu_0_mhz

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

def take_derivative(z, field, wrt='z'):
    """ Evaluate derivative of `field' with respect to `wrt' at z. """

    # Take all derivatives wrt z, convert afterwards
    x = z
    y = field
    xp, fp = central_difference(x, y)

    if wrt == 'nu':
        fp *= -1.0 * (1. + xp)**2 / nu_0_mhz
    elif wrt == 'logt':
        spline = interp1d(x, y, kind='linear')
        fp *= -2.0 * (1. + xp) / spline(xp) / 3.
    elif wrt == 'z':
        pass
    else:
        print 'Unrecognized option for wrt.'

    # x-values must be monotonically increasing - fix dep. on 'wrt'
    if not np.all(np.diff(xp) > 0):
        xp, fp = list(xp), list(fp)
        xp.reverse()
        fp.reverse()
        xp, fp = np.array(xp), np.array(fp)        

    return z, np.interp(z, xp, fp)    
