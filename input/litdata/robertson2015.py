"""
Robertson, B. E., Ellis, R. S., Furlanetto, S. R., & Dunlop, J. S. 2015, ApJ,
802, L19

Notes
-----

"""

sfrd_pars = \
{
 'a': 0.01306,
 'b': 3.66,
 'c': 2.28,
 'd': 5.29,
}

sfrd_err = \
{
 'a': 0.001,
 'b': 0.21,
 'c': 0.14,
 'd': 0.19,
}
    
def _sfrd(z, a=None, b=None, c=None, d=None):
    return a * (1. + z)**b / (1 + ((1 + z) / c)**d)    
    
def sfrd(z):
    return _sfrd(z, **sfrd_pars)
    