"""
Reddy & Steidel, 2009, ApJ, 692, 778

Table 1.
"""

import numpy as np

info = \
{
 'reference': 'Reddy & Steidel, 2009, ApJ, 692, 778',
 'data': 'Table 2', 
}

# u, g, and r dropouts, respectively
redshifts = [2.3, 3.05]
wavelength = 1700.

ULIM = -1e10

data = {}
data['lf'] = \
{# has h70's built-in
 2.3: {'M': list(np.arange(-22.58, -18.33+0.5, 0.5)),
       'phi': [0.004e-3, 0.035e-3, 0.142e-3, 0.341e-3, 1.246e-3, 2.030e-3,
               3.583e-3, 7.171e-3, 8.188e-3, 12.62e-3],
       'err': [0.003e-3, 0.007e-3, 0.016e-3, 0.058e-3, 0.083e-3, 0.196e-3,
               0.319e-3, 0.552e-3, 0.777e-3, 1.778e-3],           
      }, 
 3.05: {'M': list(np.arange(-22.77, -18.77+0.5, 0.5)),
        'phi': [0.003e-3, 0.030e-3, 0.085e-3, 0.240e-3, 0.686e-3, 1.530e-3,
                2.934e-3, 4.296e-3, 5.536e-3],
        'err': [0.001e-3, 0.013e-3, 0.032e-3, 0.104e-3, 0.249e-3, 0.273e-3,
                0.333e-3, 0.432e-3, 0.601e-3],
       },                    
}

units = {'phi': 1., 'wavelength': 1500.}

