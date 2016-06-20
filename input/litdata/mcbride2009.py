"""
McBride et al. 2009.
"""

#def MAR(z, Mh):
#    """
#    Equation 8 from McBride et al. (2009), in high-z limit.
#    
#    ..note:: This is the *DM* accretion rate, not the baryon accretion rate.
#    """
#    return 42. * (Mh / 1e12)**1.127 * (1. + 1.17 * z) * (1. + z)**1.5
            
def MAR(z, Mh):
    """
    Equation 9 from McBride et al. (2009).
    
    ..note:: This is the *median* MAH, not the mean.
    
    """
    
    return 24.1 * (Mh / 1e12)**1.094 * (1. + 1.75 * z) * (1. + z)**1.5


    