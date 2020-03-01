"""
Morishita et al., 2018, ApJ

Table 6. 
"""

import numpy as np

info = \
{
 'reference': 'Morishita et al., 2018, MNRAS',
 'data': 'Table 3', 
}

redshifts = np.array([9., 10.])
wavelength = 1500.

ULIM = -50.

# Table 6
tmp_data = {}
tmp_data['lf'] = \
{
 10.: {'M': [-23., -22., -21.],
        'phi': [-6.1, -5.9, -4.6],
        'err': [(0.5, 0.8), ULIM, ULIM],
       },
 9.: {'M': [-23., -22., -21.],
        'phi': [-5.9, -5.9, -5.4],
        'err': [ULIM, (0.5, 0.8), (0.5, 0.8)],
       },       
}

units = {'lf': 'log10'}

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
