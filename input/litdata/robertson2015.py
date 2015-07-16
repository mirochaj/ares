"""
Robertson, B. E., Ellis, R. S., Furlanetto, S. R., & Dunlop, J. S. 2015, ApJ,
802, L19

Notes
-----

"""

sfrd_pars = \
{
 'a': 0.01376,
 'b': 3.26,
 'c': 2.59,
 'd': 5.68,
}

sfrd_err = \
{
 'a': 0.001,
 'b': 0.21,
 'c': 0.14,
 'd': 0.19,
}
    
def _SFRD(z, a=None, b=None, c=None, d=None):
    return a * (1. + z)**b / (1. + ((1. + z) / c)**d) 
    
def SFRD(z, **kwargs):
    if not kwargs:
        kwargs = sfrd_pars
        
    return _SFRD(z, **kwargs)

