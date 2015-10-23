"""
Haardt, F., & Madau, P. 2012, ApJ, 746, 125

Notes
-----

"""

import os
import numpy as np
from ares.physics.Constants import h_p, c, erg_per_ev

_input = os.getenv('ARES') + '/input/hm12'

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

def _read_UVB():
    
    fn = 'UVB.out'
    skip = 20
    
    f = open('%s/%s' % (_input, fn), 'r')

    data = []
    for i, line in enumerate(f):
        if i < skip:
            continue
        
        if i == 20:
            z = np.array(map(float, line.split()))
            continue
            
        data.append(map(float, line.split()))

    return z, np.array(data)

def MetaGalacticBackground():
    z, data = _read_UVB()
    
    # Reshape data so this function looks like an 
    # ares.simulations.MetaGalacticBackground object
    
    dT = data.T
    wavelengths = dT[0]
    E = h_p * c / (wavelengths / 1e8) / erg_per_ev
    fluxes = dT[1:]

    return z[-1::-1], E[-1::-1], fluxes[-1::-1,-1::-1]

def SFRD(z, **kwargs):
    return (6.9e-3 + 0.14 * (z / 2.2)**1.5) / (1. + (z / 2.7)**4.1)

def _qso_sed_uv():
    pass

def _qso_emissivity_uv():
    pass

def _qso_sed_xray():
    pass

def _qso_emissivity_xray():
    pass