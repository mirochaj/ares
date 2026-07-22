"""

moustakas2013.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Tue Jul 21 16:26:10 2026

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
units = {'phi': 'density', 'mass': 'log10(mass)'}
redshifts = [0.1]

umach = read('umachine_dr1')
m13 = umach.get_data('smf', sources=['moustakas'])['moustakas']

data = {}
data['smf_tot'] = {}
data['smf_tot'][0.1] = \
{
 'mass': m13[(0.01, 0.2)][0],
 'phi': m13[(0.01, 0.2)][1],
 'err': m13[(0.01, 0.2)][2],
}

