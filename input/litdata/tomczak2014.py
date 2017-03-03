"""
Tomczak et al., 2014, ApJ, 783, 85
"""

import numpy as np

info = \
{
 'reference':'Tomczak et al., 2014, ApJ, 783, 85',
 'data': 'Table 1', 
}

redshifts = [3]
wavelength = 1600.

ULIM = -1e10

fits = {}

# From Louis: this is total! See paper for red/blue breakdown
_x =  np.array([9.50,9.75,10.0,10.25,10.5,10.75,11.0,11.25,11.50]) + 0.15 # TO SALPETER
_y =  -1 * np.array([2.65,2.78,3.02,3.21,3.35,3.74,4.00,4.14,4.73])
_ehi = [0.06,0.07,0.08,0.09,0.10,0.13,0.18,0.17,0.31]
_elo = [0.07,0.08,0.09,0.10,0.13,0.17,0.25,0.28,2.00]

# Table 1
tmp_data = {}
tmp_data['smf'] = \
{
 3: {'M': list(10**_x),
     'phi': _y,
     'err': zip(*(_elo, _ehi)),
    },            
}

units = {'smf': 'log10'}

data = {}
data['smf'] = {}
for key in tmp_data['smf']:
    mask = []
    for element in tmp_data['smf'][key]['err']:
        if element == ULIM:
            mask.append(1)
        else:
            mask.append(0)
    
    mask = np.array(mask)
    
    data['smf'][key] = {}
    data['smf'][key]['M'] = np.ma.array(tmp_data['smf'][key]['M'], mask=mask) 
    data['smf'][key]['phi'] = np.ma.array(tmp_data['smf'][key]['phi'], mask=mask) 
    data['smf'][key]['err'] = tmp_data['smf'][key]['err']








