"""
Madau, P., & Dickinson, M. 2014, ARA&A, 52, 415
"""

pars_ml = \
{
 'a': 0.015,
 'b': 2.7,
 'c': 2.9,
 'd': 5.6,
}

#pars_err = \
#{
# 'a': 0.001,
# 'b': 0.21,
# 'c': 0.14,
# 'd': 0.19,
#}

def _SFRD(z, a=None, b=None, c=None, d=None):
    return a * (1. + z)**b / (1 + ((1 + z) / c)**d)

def SFRD(z):
    return _SFRD(z, **pars_ml)
    
info = \
{
 'Lmin': '0.03 * Lstar',
}    
    
