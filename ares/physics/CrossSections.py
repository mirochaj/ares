"""
CrossSections.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-10-25.

Description: Compute the bound-free absorption cross-sections via the fitting
formulae of Verner et al. 1996.
     
"""

from numpy import sqrt

E_th = [13.6, 24.6, 54.4]

# HI is the first 7-element sub-array in 'params', HeI is the second, and 
# HeII is the third.  In order, the coefficients in these arrays are:
# E_0, sigma_0, y_a, P, y_w, y_0, y_1
params = [[4.298e-1, 5.475e4, 3.288e1, 2.963, 0.0, 0.0, 0.0],
          [13.61, 9.492e2, 1.469, 3.188, 2.039, 4.434e-1, 2.136],
          [1.72, 1.369e4, 3.288e1, 2.963, 0.0, 0.0, 0.0]]

def PhotoIonizationCrossSection(E, species=0):
    """ 
    Compute bound-free absorption cross section for hydrogen or helium.

    Parameters
    ----------
    E : float
        Photon energy (eV)
    species : int
        Species ID number. HI = 0 (default), HeI = 1, HeII = 2
        
    Returns
    -------
    Cross section in cm**2 using fits from Verner et al. (1996).    
    
    References
    ----------
    Verner, D.A., Ferland, G.J., Korista, K.T., and Yaklovlev, D.G., ApJ, 465, 487
    
    """
    
    if E < E_th[species]:
        return 0.0
    
    x = (E / params[species][0]) - params[species][5]
    y = sqrt(x**2 + params[species][6]**2)
    F_y = ((x - 1.0)**2 + params[species][4]**2) * \
        y**(0.5 * params[species][3] - 5.5) * \
        (1.0 + sqrt(y / params[species][2]))**-params[species][3]
                                
    return params[species][1] * F_y * 1e-18
    
sigma0 = PhotoIonizationCrossSection(E_th[0])
def ApproximatePhotoIonizationCrossSection(E, species=0):
    if E < E_th[species]:
        return 0.0
    return sigma0 * (E_th[species] / E)**3                         
    

    