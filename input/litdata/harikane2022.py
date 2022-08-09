"""
Finkelstein et al., 2015, ApJ, 810, 71
"""

import numpy as np

info = \
{
 'reference': 'Harikane et al., 2022',
 'data': 'Table 7',
 'label': 'Harikane+',
}

redshifts = [9, 12, 17]
wavelength = 1500.

ULIM = -1e10

fits = {}

fits['lf'] = {}

fits['lf']['pars'] = \
{
 'Mstar': [],
 'pstar': [],
 'alpha': [],
}

fits['lf']['err'] = \
{
 'Mstar': [],
 'pstar': [],
 'alpha': [],
}

# Table 5
tmp_data = {}
tmp_data['lf'] = \
{
 9 : {'M': np.arange(-23.03, -17.03, 1),
      'phi': np.array([6.95e-5, 7.70e-5, 4.08e-5, 4.08e-5, 3.01e-4, 5.36e-4]),
      'err': [ULIM, ULIM, [9.59e-5, 3.92e-5], [9.59e-5, 3.92e-5],
            [2.19e-4, 1.84e-4], [4.39e-4, 3.68e-4]],
 },
 12 : {'M': np.arange(-23.23, -17.23, 1),
       'phi': np.array([5.91e-6, 6.51e-6, 4.48e-6, 1.29e-5, 1.46e-5, 1.03e-4]),
       'err': [ULIM, ULIM, [10.34e-6, 3.82e-6], [1.72e-5, 0.87e-5],
            [1.95e-5, 0.99e-5], [1.06e-4, 0.67e-4]],
 },
 17 : {'M': np.array([-23.59, -20.59]),
     'phi': np.array([2.15e-6, 5.39e-6]),
     'err': [ULIM, [7.16e-6, 3.57e-6]],
 },
}

#for redshift in tmp_data['lf'].keys():
#    for i in range(len(tmp_data['lf'][redshift]['M'])):
#        tmp_data['lf'][redshift]['phi'][i] *= 1e-3
#
#        if tmp_data['lf'][redshift]['err'][i] == ULIM:
#            continue
#
#        tmp_data['lf'][redshift]['err'][i][0] *= 1e-3
#        tmp_data['lf'][redshift]['err'][i][1] *= 1e-3
#        tmp_data['lf'][redshift]['err'][i] = \
#            tuple(tmp_data['lf'][redshift]['err'][i])

units = {'lf': 1.}

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
