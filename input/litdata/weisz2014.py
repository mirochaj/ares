"""
Weisz, Johnson, & Conroy, 2014, ApJ, 794, L3

Table 1.
"""

import numpy as np

info = \
{
 'reference': 'Reddy & Steidel, 2009, ApJ, 692, 778',
 'data': 'Table 1', 
}

# u, g, and r dropouts, respectively
redshifts = [0.75, 1.25, 2, 3, 4, 5]
wavelength = 1700.

ULIM = -1e10

data = {}
data['lf'] = \
{# has h70's built-in
 0.75: {'M': list(np.arange(-13.44, -1.44+3, 3)),
       'phi': [],
       'err': [],           
      }, 
 1.25: {'M': list(np.arange(-15.53, -3.53+3, 3)),
        'phi': [],
        'err': [],
       },   
 2: {'M': list(np.arange(-11.75, -2.75+3, 3)),
         'phi': [],
         'err': [],
        },
 3: {'M': list(np.arange(-13.92, -4.92+3, 3)),
          'phi': [0.0333, 0.0956, 0.2162, 1.4591],
          'err': [0.0189, 0.0454, 0.0899, 0.8935],
         },         

 4: {'M': list(np.arange(-13.82, -4.82+3, 3)),
           'phi': [0.3184, 1.8523, 5.9041, 4.5149],
           'err': [0.1510, 0.7706, 2.6702, 3.91],
          },  
 5: {'M': [-15.66, -12.16, -8.66, -5.16],
           'phi': [0.1608, 1.1793, 3.8774, 2.573],
           'err': [0.1206, 0.5334, 1.4540, 2.1117],
          },                                  
}

units = {'lf': 1., 'wavelength': 1500.}

