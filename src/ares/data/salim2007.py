"""

salim2007.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Tue Jul 21 18:58:04 2026

Description:

"""

import numpy as np
from . import read

cosmo = \
{
 'hubble_0': 0.7,
 'omega_m_0': 0.3,
 'omega_l_0': 0.7,
}
units = {'ssfr': 'log10(ssfr)', 'mass': 'log10(mass)'}
redshifts = [0.1]

umach = read('umachine_dr1')
_data = umach.get_data('ssfr', sources=['salim07'])['salim07']

data = {}
data['ssfr'] = {}
data['ssfr'][0.1] = \
{
 'mass': _data[(0.005, 0.2)][0],
 'ssfr': _data[(0.005, 0.2)][1],
 'err': 0.28 * np.ones_like(_data[(0.005, 0.2)][0]),
}
