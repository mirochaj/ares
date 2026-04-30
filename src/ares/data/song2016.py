"""
Song et al., 2016, ApJ, 825, 5
"""

import numpy as np

info = \
{
 'reference': 'Song et al., 2016, ApJ, 825, 5',
 'data': 'Table 2',
 'fits': 'Table 1',
 'imf': ('salpeter', (0.1, 100)),
}

redshifts = [4, 5, 6, 7, 8]
wavelength = 1600.

ULIM = -1e10

fits = {}

#fits['smf'] = {}
#
#fits['smf']['pars'] = \
#{
# 'Mstar': [],
# 'pstar': [],
# 'alpha': [],
#}
#
#fits['smf']['err'] = \
#{
# 'Mstar': [],
# 'pstar': [],  # should be asymmetric!
# 'alpha': [],
#}

# Table 5
tmp_data = {}
tmp_data['smf'] = \
{
 4: {'M': list(10**np.arange(7.25, 11.75, 0.5)),
     'phi': [-1.57, -1.77, -2.00, -2.22, -2.52, -2.91, -3.37, -4.00, -4.54],
     'err': [(0.21, 0.16), (0.15, 0.14), (0.13, 0.10), (0.09, 0.09), (0.09, 0.09),
             (0.12, 0.05), (0.09, 0.12), (0.20, 0.25), (0.34, 0.55)],
    },
 5: {'M': list(10**np.arange(7.25, 11.75, 0.5)),
     'phi': [-1.47, -1.72, -2.01, -2.33, -2.68, -3.12, -3.47, -4.12, -4.88],
     'err': [(0.24, 0.21), (0.20, 0.20), (0.16, 0.16), (0.15, 0.10),
             (0.07, 0.14), (0.09, 0.11), (0.16, 0.14), (0.25, 0.38), (0.4, 0.61)],
    },
 6: {'M': list(10**np.arange(7.25, 10.75, 0.5)),
       'phi': [-1.47, -1.81, -2.26, -2.65, -3.14, -3.69, -4.27],
       'err': [(0.35, 0.32), (0.23, 0.28), (0.21, 0.16), (0.15, 0.15),
               (0.12, 0.11), (0.12, 0.13), (0.38, 0.86)],
      },
 7: {'M': list(10**np.arange(7.25, 11.25, 0.5)),
       'phi': [-1.63, -2.07, -2.49, -2.96, -3.47, -4.11, -4.61, -5.24],
       'err': [(0.54, 0.54), (0.45, 0.41), (0.38, 0.32), (0.32, 0.30),
               (0.32, 0.35), (0.41, 0.57), (0.72, 0.82), (0.90, 0.57)],
      },
 8: {'M': list(10**np.arange(7.25, 10.25, 0.5)),
       'phi': [-1.73, -2.28, -2.88, -3.45, -4.21, -5.31],
       'err': [(1.01, 0.84), (0.84, 0.64), (0.75, 0.57), (0.57, 0.60),
               (0.63, 0.78), (1.01, 1.64)],
      },
}

units = {'smf': 'log10'}

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
