import numpy as np

info = \
{
 'reference': 'Bouwens et al., 2009',
 'data': 'Table 4',
 'label': 'Bouwens+ (2009)',
}

redshifts = [2.5]
wavelength = None
units = {'beta': 1.}

_data = \
{
 2.5: {'M': np.array([-21.73, -20.73, -19.73, -18.73]),
     'beta': [-1.18, -1.58, -1.54, -1.88],
     'err': [0.17, 0.1, 0.06, 0.05],
     'sys': [0.15] * 4,
 },
}

data = {}
data['beta'] = {}
for key in _data:
    data['beta'][key] = {}
    for element in _data[key]:
        data['beta'][key][element] = np.array(_data[key][element])
