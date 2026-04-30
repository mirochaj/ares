"""
Stefanon et al., 2017, ApJ, 843, 36
"""

import numpy as np

info = \
{
 'reference': 'Stefanon et al., 2017, ApJ, 843, 36',
 'data': 'Table 2',
 'imf': ('chabrier', (0.1, 100.)),
 'other': 'data from arxiv version',
 'label': 'Stefanon+ (2017)',
}

redshifts = [4., 5., 6., 7.]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 2
tmp_data = {}
tmp_data['smf'] = \
{
 4: {'M': list(10**np.arange(8.84, 10.04, 0.1)) + list(10**np.arange(10.09, 11.70, 0.15)),
     'phi': [537e-5, 477e-5, 405e-5, 316e-5, 258e-5, 252e-5, 238e-5, 162e-5,
             155e-5, 109e-5, 71e-5, 61e-5, 29e-5, 18.5e-5, 11.3e-5, 6.7e-5,
             4.1e-5, 2.9e-5, 2.2e-5, 1.20e-5, 0.37e-5, 0.15e-5, 0.075e-5],
     'err': [(147e-5, 141e-5), (115e-5, 114e-5), 94e-5, 73e-5, 60e-5,
             (59e-5, 58e-5), (56e-5, 55e-5), (39e-5, 38e-5), 37e-5, 27e-5,
             (19e-5, 18e-5), (28e-5, 27e-5), 13e-5, (8.3e-5, 8.2e-5),
             5e-5, 3e-5, 1.9e-5, (1.4e-5, 1.3e-5), 1e-5, (0.64e-5, 0.59e-5),
             (0.3e-5, 0.23e-5), (0.21e-5, 0.12e-5), (0.174e-5, 0.07e-5)],
    },
 5: {'M': list(10**np.array([9.44, 9.64, 9.84, 10.09, 10.38, 10.68])),
     'phi': [208e-5, 91e-5, 20.5e-5, 7.8e-5, 4.4e-5, 0.95e-5],
     'err': [(125e-5, 102e-5), (42e-5, 39e-5), (9.2e-5, 8.5e-5),
             (3.7e-5, 3.4e-5), (2.5e-5, 2.1e-5), (1.29e-5, 0.7e-5)],
    },
 6: {'M': list(10**np.array([9.60, 9.94, 10.47])),
      'phi': [65e-5, 8.1e-5, 0.34e-5],
      'err': [(53e-5, 40e-5), (7.4e-5, 5.3e-5), (0.79e-5, 0.32e-5)],
     },
 7: {'M': [10**9.84],
      'phi': [7.4e-5],
      'err': [(10.4e-5, 6e-5)],
     },


}

units = {'smf': 1.}

data = {}
data['smf'] = {}
for key in tmp_data['smf']:
    mask = []
    for element in tmp_data['smf'][key]['err']:
        if element == ULIM:
            mask.append(1)
        else:
            mask.append(0)

    mask = np.array(mask)

    data['smf'][key] = {}
    data['smf'][key]['M'] = np.ma.array(tmp_data['smf'][key]['M'], mask=mask)
    data['smf'][key]['phi'] = np.ma.array(tmp_data['smf'][key]['phi'], mask=mask)
    data['smf'][key]['err'] = tmp_data['smf'][key]['err']
