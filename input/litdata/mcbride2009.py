"""
McBride et al. 2009.
"""

#def Macc(z, Mh):
#    return 3. * (Mh / 1e10)**1.127 * ((1. + z) / 7.)**2.5
    
    
def Macc(z, Mh):
    """
    Equation 8 from McBride et al. (2009), in high-z limit.
    """
    return 42. * (Mh / 1e12)**1.127 * (1. + 1.17 * z) * (1. + z)**1.5
    
#def Macc(z, Mh):
#    """
#    Equation 9 from McBride et al. (2009).
#    """
#    return 42.1 * (Mh / 1e12)**1.094 * (1. + 1.75 * z) * (1. + z)**1.5

    