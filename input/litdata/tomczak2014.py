"""
Tomczak et al., 2014, ApJ, 783, 85
"""

import numpy as np

info = \
{
 'reference':'Tomczak et al., 2014, ApJ, 783, 85',
 'data': 'Table 1', 
}

redshifts = [2,3]
wavelength = 1600.

ULIM = -1e10

fits = {}

# From Louis: this is total! See paper for red/blue breakdown
_x3 =  np.array([9.50,9.75,10.0,10.25,10.5,10.75,11.0,11.25,11.50]) + 0.15 # TO SALPETER
_y3 =  -1 * np.array([2.65,2.78,3.02,3.21,3.35,3.74,4.00,4.14,4.73])
_ehi3 = [0.06,0.07,0.08,0.09,0.10,0.13,0.18,0.17,0.31]
_elo3 = [0.07,0.08,0.09,0.10,0.13,0.17,0.25,0.28,2.00]

_x2 =  np.array([9.0,9.25,9.50,9.75,10.0,10.25,10.5,10.75,11.0,11.25,11.50]) + 0.15 
_y2 =  -1 * np.array([2.20,2.31,2.41,2.54,2.67,2.76,2.87,3.03,3.13,3.56,4.27])
_ehi2 = [0.05,0.05,0.05,0.06,0.06,0.06,0.07,0.08,0.08,0.10,0.12]
_elo2 = [0.06,0.06,0.06,0.06,0.07,0.07,0.08,0.09,0.10,0.13,0.15]

# Table 1
tmp_data = {}
tmp_data['smf'] = \
{
 2: {'M': list(10**_x2),
     'phi': _y2,
     'err': list(zip(*(_ehi2, _elo2))),
    },
 3: {'M': list(10**_x3),
     'phi': _y3,
     'err': list(zip(*(_ehi3, _elo3))),
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








