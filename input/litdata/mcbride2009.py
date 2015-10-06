"""
McBride et al. 2009.
"""

def Macc(z, Mh):
    return 3. * (Mh / 1e10)**1.127 * ((1. + z) / 7.)**2.5
    
    
