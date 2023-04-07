"""
Gruppioni et al., 2020, A&A, 643, 8G
"""

import numpy as np

info = \
{
 'reference': 'Gruppioni et al., 2020, A&A, 643, 8G',
 'data': 'Tables 2 & 3',
 'label': 'Gruppioni+ (2020)',
}

# Really coarse redshift bins
redshifts = np.array([4, 5.25])
zbins = np.array([[3.5, 4.5], [4.5, 6.]])

ULIM = -1e10

# Table 3
tmp_data = {}
tmp_data['lf_tot'] = \
{
 4: {'L': [11.75, 12.00, 12.25, 12.50],
   'phi': [-3.37, -3.37, -4.10, -4.10],
   'err': [(0.40, 0.82), (0.40, 0.58), (0.59, 0.78), (0.52, 0.78)],
      },
 5.25: {'L': [11.25, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00, 13.25],
   'phi': [-3.94, -3.91, -3.45, -3.38, -3.79, -4.16, -3.76, -3.63, -4.21],
   'err': [(0.55, 0.78), (0.54, 0.78), (0.38, 0.53), (0.34, 0.48), (0.41, 0.63),
        (0.59, 0.78), (0.55, 0.78), (0.43, 0.79), (0.55, 0.78)],
      },
}

# Table 2
tmp_data['lf_250'] = \
{
 4: {'L': [10, 10.25, 10.50, 10.75],
   'phi': [-3.52, -3.36, -3.68, -4.10],
   'err': [(0.59, 0.76), (0.48, 0.47), (0.48, 0.45), (0.68, 0.76)],
      },
 5.25: {'L': [10.25, 10.50, 10.75, 11.00, 11.25, 11.50, 11.75],
   'phi': [-3.49, -3.21, -3.61, -3.52, -3.61, -3.91, -3.91],
   'err': [(0.45, 0.65), (0.30, 0.37), (0.47, 0.44), (0.41, 0.58),
        (0.34, 0.47), (0.46, 0.62), (0.46, 0.62)],
      },
}

units = {'lf_tot': 'log10', 'L': 'log10'}

data = {}
data['lf_tot'] = {}
data['lf_250'] = {}
for lf in ['lf_tot', 'lf_250']:
    for key in tmp_data[lf]:
        #mask = np.array(tmp_data['lf'][key]['err']) == ULIM
        N = len(tmp_data[lf][key]['L'])
        mask = np.array([tmp_data[lf][key]['err'][i] == ULIM \
            for i in range(N)])

        data[lf][key] = {}
        data[lf][key]['L'] = np.ma.array(tmp_data[lf][key]['L'],
            mask=mask)
        data[lf][key]['phi'] = np.ma.array(tmp_data[lf][key]['phi'],
            mask=mask)
        data[lf][key]['err'] = tmp_data[lf][key]['err']
