"""
Bowler et al., 2020, MNRAS

Table 6. 
"""

import numpy as np

info = \
{
 'reference': 'Bowler et al., 2020, MNRAS',
 'data': 'Table 6', 
 'fits': 'Table 7', 
}

redshifts = np.array([8., 9.])
wavelength = 1500.

ULIM = -1e10

# Table 6
tmp_data = {}
tmp_data['lf'] = \
{
 8.: {'M': [-21.65, -22.15, -22.90],
        'phi': [2.95e-6, 0.58e-6, 0.14e-6],
        'err': [0.98e-6, 0.33e-6, 0.06e-6],
       },
 9.: {'M': [-21.9, -22.9],
        'phi': [0.84e-6, 0.16e-6],
        'err': [0.49e-6, 0.11e-6],
       },       
}

units = {'lf': 1.}

data = {}
data['lf'] = {}
for key in tmp_data['lf']:
    #mask = np.array(tmp_data['lf'][key]['err']) == ULIM
    N = len(tmp_data['lf'][key]['M'])
    mask = np.array([tmp_data['lf'][key]['err'][i] == ULIM for i in range(N)])
    
    data['lf'][key] = {}
    data['lf'][key]['M'] = np.ma.array(tmp_data['lf'][key]['M'], mask=mask) 
    data['lf'][key]['phi'] = np.ma.array(tmp_data['lf'][key]['phi'], mask=mask) 
    data['lf'][key]['err'] = tmp_data['lf'][key]['err']
