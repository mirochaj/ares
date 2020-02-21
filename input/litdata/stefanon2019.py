"""
Stefanon et al., 2019, ApJ, 883, 99

Table 6. 
"""

import numpy as np

info = \
{
 'reference': 'Stefanon et al., 2019, ApJ, 883, 99',
 'data': 'Table 6', 
}

redshifts = np.array([8., 9.])
wavelength = 1500.

ULIM = -1e10

# Table 6
tmp_data = {}
tmp_data['lf'] = \
{
 8.: {'M': [-22.55, -22.05, -21.55],
        'phi': [0.76e-6, 1.38e-6, 4.87e-6],
        'err': [(0.74e-6, 0.41e-6), (1.09e-6, 0.66e-6), (2.01e-6, 1.41e-6)],
       },
 9.: {'M': [-22.35, -22.00, -21.60, -21.20],
        'phi': [0.43e-6, 0.43e-6, 1.14e-6, 1.64e-6],
        'err': [(0.99e-6, 0.36e-6), (0.98e-6, 0.36e-6), (1.5e-6, 0.73e-6),
                (2.16e-6, 1.06e-6)],
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
