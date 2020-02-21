"""
McLure et al., 2013, MNRAS, 432, 2696

Table 2. 
"""

import numpy as np

info = \
{
 'reference': 'McLure et al., 2013, MNRAS, 432, 2696',
 'data': 'Table 2', 
 'fits': 'Table 3', 
}

redshifts = np.array([7., 8.])
wavelength = 1500.

ULIM = -1e10

# Table 6
tmp_data = {}
tmp_data['lf'] = \
{
 7.: {'M': list(np.arange(-21, -16.5, 0.5)),
        'phi': [0.00003, 0.00012, 0.00033, 0.00075, 0.0011, 0.0021, 0.0042,
        0.0079, 0.011],
        'err': [0.000001, 0.00002, 0.00005, 0.00009, 0.0002, 0.0006, 0.0009,
        0.0019, 0.0025],
       },
 8.: {'M': list(np.arange(-21.25, -16.75, 0.5)),
        'phi': [0.000008, 0.00003, 0.0001, 0.0003, 0.0005, 0.0012, 0.0018,
            0.0028, 0.0050],
        'err': [0.000003, 0.000009, 0.00003, 0.00006, 0.00012, 0.0004,
            0.0006, 0.0008, 0.0025],
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
