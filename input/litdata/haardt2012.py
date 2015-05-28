"""
Haardt, F., & Madau, P. 2012, ApJ, 746, 125

Notes
-----

"""

pars_ml = \
{
 'a': 6.9e-3,
 'b': 0.14,
 'c': 2.2,
 'd': 5.29,
}

pars_err = \
{
 'a': 0.001,
 'b': 0.21,
 'c': 0.14,
 'd': 0.19,
}

def sfrd(z):
    return (6.9e-3 + 0.14 * (z / 2.2)**1.5) / (1. + (z / 2.7)**4.1)

def _qso_sed_uv():
    pass

def _qso_emissivity_uv():
    pass

def _qso_sed_xray():
    pass

def _qso_emissivity_xray():
    pass