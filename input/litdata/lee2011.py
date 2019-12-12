import numpy as np

info = \
{
 'reference': 'Lee et al., 2011',
 'data': 'Section 5.3, Figure 5'
}

redshifts = [4]
wavelength = 1700.
units = {'beta': 1.}


_data = \
{
 # Really z~3.7, samples 1, 3, 6
 4: {'M': [-23.3, -22.18, -21.43],
     'beta': [-1.16, -1.37, -1.78],
     'err': [0.31, 0.14, 0.28],
 },
}


data = {}
data['beta'] = {}
for key in _data:
    data['beta'][key] = {}
    for element in _data[key]:
        data['beta'][key][element] = np.array(_data[key][element])