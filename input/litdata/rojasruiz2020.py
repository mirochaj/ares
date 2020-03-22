"""
Rojas-Ruiz et al., 2020, ApJ accepted

https://arxiv.org/abs/2002.06209

Table 2 
"""

import numpy as np

info = \
{
 'reference': 'Rojas-Ruiz et al., 2020, ApJ',
 'data': 'Table 2', 
 'label': 'Rojas-Ruiz+ (2020)'
}

redshifts = np.array([8., 9.])
zrange = np.array([(7., 8.4), (8.4, 11.)])
wavelength = 1500.

ULIM = -1e10

# Table 6
tmp_data = {}
tmp_data['lf'] = \
{
 8.: {'M': [-23, -22, -21],
        'phi': [1.0175e-6, 2.9146e-6, 17.932e-6],
        'err': [(2.752e-6, 0.253e-6), (6.39e-6, 0.969e-6), (36.767e-6, 7.298e-6)],
       },
 9.: {'M': [-23, -22, -21],
        'phi': [5.6904e-6, 7.0378e-6, 40.326e-6],
        'err': [(11.184e-6, 2.413e-6), (13.413e-6, 3.037e-6), (81.276e-6, 18.802e-6)],
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
