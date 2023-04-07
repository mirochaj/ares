"""

behroozi2013.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Mar  2 13:31:44 PST 2017

Description: 

"""

import numpy as np

redshifts = [7.]

wavelength = 1500.

fits = {}

fits['lf'] = {}

# Table 5
fits['lf']['pars'] = \
{
 'Mstar': [-20.89], 
 'pstar': [10**3.54],
 'alpha': [-2.04],
}

fits['lf']['err'] = \
{
 'Mstar': [0.66], 
 'pstar': [0.15],
 'alpha': [0.465],
}

# Table 4
data = {}
data['lf'] = \
{
 7.: {'M': list(np.arange(-20.25, -14.75, 0.5)),
      'phi': [-3.4184, -3.0263, -2.9044, -2.7418, -2.3896, -2.1032,
              -1.8201, -1.7548, -1.6044, -1.4012, -1.4012],
      'err': [ 0.1576,  0.1658,  0.1431,  0.1332,  0.1401, 0.1990,
               0.1940,  0.1893,  0.2117,  0.3123,  0.3122],
      },
}


units = {'lf': 'log10'}


