"""
Van der Burg et al., 2010, A&A, 523, A74

Table 1.
"""

import numpy as np

info = \
{
 'reference': 'Van der Burg et al., 2010, A&A, 523, A74',
 'data': 'Table 1', 
}

# u, g, and r dropouts, respectively
redshifts = [3, 4, 5]
wavelength = 1500.

ULIM = -1e10

tmp_data = {}
tmp_data['lf'] = \
{
 3: {'M': list(np.arange(-23.2, -18.5, 0.3)),
       'phi': [0.001e-3, 0.001e-3, 0.007e-3, 0.022e-3, 0.057e-3, 0.113e-3,
               0.254e-3, 0.497e-3, 0.788e-3, 1.188e-3, 1.745e-3, 2.240e-3,
               2.799e-3, 3.734e-3, 4.720e-3, 3.252e-3],
       'err': [0.001e-3, 0.001e-3, 0.002e-3, 0.007e-3, 0.020e-3, 0.028e-3, 
               0.027e-3, 0.061e-3, 0.110e-3, 0.267e-3, 0.377e-3, 0.373e-3, 
               0.519e-3, 0.863e-3, 0.866e-3, 1.508e-3],           
      }, 
 4: {'M': list(np.arange(-22.6, -18.5, 0.3)),
        'phi': [0.004e-3, 0.016e-3, 0.035e-3, 0.086e-3, 0.160e-3, 0.287e-3,
                0.509e-3, 0.728e-3, 1.006e-3, 1.465e-3, 1.756e-3, 2.230e-3,
                2.499e-3, 3.038e-3],
        'err': [0.002e-3, 0.003e-3, 0.005e-3, 0.010e-3, 0.027e-3, 0.060e-3,
                0.061e-3, 0.067e-3, 0.040e-3, 0.147e-3, 0.063e-3, 0.305e-3,
                0.564e-3, 0.370e-3],           
       },                    
 5: {'M': list(np.arange(-22.6, -19., 0.3)),
        'phi': [0.002e-3, 0.010e-3, 0.032e-3, 0.065e-3, 0.121e-3, 0.234e-3,
                0.348e-3, 0.494e-3, 0.708e-3, 1.123e-3, 1.426e-3, 1.624e-3,
                1.819e-3],
        'err': [0.002e-3, 0.002e-3, 0.010e-3, 0.015e-3, 0.016e-3, 0.028e-3,
                0.025e-3, 0.050e-3, 0.030e-3, 0.211e-3, 0.229e-3, 0.095e-3,
                0.630e-3],   
       },
}

units = {'lf': 1., 'wavelength': 1500.}

data = {}
data['lf'] = {}
for key in tmp_data['lf']:
    N = len(tmp_data['lf'][key]['M'])
    mask = np.array([tmp_data['lf'][key]['err'][i] == ULIM for i in range(N)])
    
    #mask = []
    #for element in tmp_data['lf'][key]['err']:
    #    if element == ULIM:
    #        mask.append(1)
    #    else:
    #        mask.append(0)
    #
    #mask = np.array(mask)
    
    data['lf'][key] = {}
    data['lf'][key]['M'] = np.ma.array(tmp_data['lf'][key]['M'], mask=mask) 
    data['lf'][key]['phi'] = np.ma.array(tmp_data['lf'][key]['phi'], mask=mask) 
    data['lf'][key]['err'] = tmp_data['lf'][key]['err']


