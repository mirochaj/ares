"""
McBride et al. 2009.
"""

#def Macc(z, Mh):
#    return 3. * (Mh / 1e10)**1.127 * ((1. + z) / 7.)**2.5
    
    
#def Macc(z, Mh):
#    """
#    Equation 8 from McBride et al. (2009), in high-z limit.
#    
#    ..note:: This is the *mass* accretion rate, not the baryon accretion rate.
#    """
#    return 42. * (Mh / 1e12)**1.127 * (1. + 1.17 * z) * (1. + z)**1.5
    
    
def Mofz(z, zform, Mmin, beta, gamma):
    """
    Mass of a halo as a function of redshift given its formation redshift,
    initial mass, and the phenomenological parameters beta and gamma.
    
    ..note:: This is for individual halos! beta and gamma are fitted parameters,
        and in general seem to depend on mass.
        
    """
    
    return Mmin * ((1. + z) / (1. + zform))**beta \
        * np.exp(-gamma * (z - zform))
    
def dMdz(z, zform, Mmin, beta, gamma):
    return Mofz(z, zform, Mmin, beta, gamma) * (beta / (1. + z) - gamma)
    
def Macc(z, Mh):
    """
    Equation 9 from McBride et al. (2009).
    
    ..note:: This is the *median* MAH, not the mean.
    
    """
    
    return 24.1 * (Mh / 1e12)**1.094 * (1. + 1.75 * z) * (1. + z)**1.5

def Mofz_ensemble(z):
    pass
    