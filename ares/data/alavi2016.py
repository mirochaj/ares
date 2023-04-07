"""
Alavi et al., 2016, ApJ, 832, 56

Table 1.
"""

import numpy as np

info = \
{
 'reference': 'Alavi et al., 2016, ApJ, 832, 56',
 'data': 'Table 2 (LBGs only)', 
}

# u, g, and r dropouts, respectively
redshifts = [1.65, 2.0, 2.7]
wavelength = 1500.

ULIM = -1e10

tmp_data = {}
tmp_data['lf'] = \
{
 1.65: {'M': list(np.arange(-18, -12, 1)),
       'phi': [0.527e-2, 0.733e-2, 1.329e-2, 1.150e-2, 8.566e-2, 10.477e-2],
       'err': [(0.287e-2, 0.513e-2), (0.351e-2, 0.579e-2), (0.574e-2, 0.899e-2), 
               (0.743e-2, 1.517e-2), (4.663e-2, 8.332e-2), (8.665e-2, 24.098e-2)],           
      }, 
 2: {'M': list(np.arange(-20, -12, 1)),
        'phi': [0.058e-2, 0.199e-2, 0.768e-2, 1.617e-2, 3.556e-2, 5.736e-2, 
                20.106e-2, 25.922e-2],
        'err': [(0.032e-2, 0.057e-2), (0.062e-2, 0.085e-2), (0.131e-2, 0.156e-2),
                (0.228e-2, 0.263e-2), (0.560e-2, 0.656e-2), (1.271e-1, 1.592e-2),
                (5.504e-2, 7.269e-2), (16.746e-2, 34.192e-2)],           
       },                    
 2.7: {'M': list(np.arange(-20, -12, 1)),
        'phi': [0.112e-2, 0.186e-2, 0.409e-2, 1.309e-2, 4.218e-2, 7.796e-2,
                32.106e-2, 24.187e-2],
        'err': [(0.045e-2, 0.067e-2), (0.058e-2, 0.079e-2), (0.089e-2, 0.110e-2),
                (0.192e-2, 0.223e-2), (0.691e-2, 0.814e-2), (2.06e-2, 2.69e-2),
                (13.870e-2, 21.716e-2), (20.003e-2, 55.631e-2)],   
       },
}

units = {'lf': 1., 'wavelength': 1500.}

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

