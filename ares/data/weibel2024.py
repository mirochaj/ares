"""
Weibel et al., 2024, MNRAS, 533, 1808
"""

import numpy as np

info = \
{
 'reference':'Weibel et al., 2024, MNRAS, 533, 1808',
 'data': 'Tables 2 and 3',
 'imf': ('Kroupa', (None, None)),
 'link': "https://ui.adsabs.harvard.edu/abs/2024MNRAS.533.1808W/abstract",
}

redshifts = [4, 5, 6, 7, 8, 9]

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['smf_tot'] = \
{
 4: {'M': list(10**np.arange(7.75, 12.25, 0.5)),
     'phi': [-1.57, -1.97, -2.38, -2.74, -3.17, -3.68, -4.23, -4.78, -5.91],
     'err': [(0.10, 0.12), (0.06, 0.06), (0.04, 0.05), (0.06, 0.06), (0.07, 0.08),
             (0.09, 0.11), (0.14, 0.19), (0.19, 0.31), (0.54, 1.13)],
    },
 5: {'M': list(10**np.arange(8.25, 12.25, 0.5)),
     'phi': [-2.00, -2.38, -2.89, -3.35, -4.04, -4.87, -5.80, -5.87],
     'err': [(0.09, 0.12), (0.06, 0.07), (0.08, 0.10), (0.10, 0.13), (0.14, 0.19),
             (0.23, 0.43), (0.54, 2.55), (0.55, np.inf)],
    },
 6: {'M': list(10**np.arange(8.25, 12.25, 0.5)),
     'phi': [-2.24, -2.65, -3.26, -3.85, -4.44, -5.26, -5.38, -5.82],
     'err': [(0.12, 0.17), (0.09, 0.11), (0.11, 0.15), (0.15, 0.21), (0.20, 0.35),
             (0.35, np.inf), (0.42, np.inf), (0.56, np.inf)],
    },
 7: {'M': list(10**np.arange(8.25, 12.25, 0.5)),
     'phi': [-2.40, -2.70, -3.35, -3.96, -4.35, -4.78, -5.38, -5.69],
     'err': [(0.15, 0.24), (0.14, 0.20), (0.14, 0.21), (0.19, 0.33), (0.25, 0.58),
             (0.38, np.inf), (0.43, np.inf), (0.55, np.inf)],
    },
 8: {'M': list(10**np.arange(8.75, 12.25, 0.5)),
     'phi': [-3.00, -3.64, -4.09, -4.33, -4.78, -5.54, -5.66],
     'err': [(0.18, 0.28), (0.19, 0.33), (0.24, 0.55), (0.30, 1.39), (0.45, np.inf),
             (0.57, np.inf), (0.56, np.inf)],
    },   
 9: {'M': list(10**np.arange(8.75, 12.25, 0.5)),
     'phi': [-3.39, -3.81, -4.35, -4.79, -5.27, -5.61, -5.61],
     'err': [(0.25, 0.64), (0.24, 0.52), (0.31, 1.54), (0.40, np.inf), (0.54, np.inf),
             (0.64, np.inf), (0.61, np.inf)],
    },      

 
}


units = {'smf_tot': 'log10', 'smf': 'log10'}

data = {}
data['smf_tot'] = {}
for group in ['smf_tot']:

    for key in tmp_data[group]:

        if key not in tmp_data[group]:
            continue

        subdata = tmp_data[group]

        mask = []
        for element in subdata[key]['err']:
            if element == ULIM:
                mask.append(1)
            else:
                mask.append(0)

        mask = np.array(mask)

        data[group][key] = {}
        data[group][key]['M'] = np.ma.array(subdata[key]['M'], mask=mask)
        data[group][key]['phi'] = np.ma.array(subdata[key]['phi'], mask=mask)
        data[group][key]['err'] = tmp_data[group][key]['err']

# Make `smf` and `smf_tot` interchangeable
data['smf'] = data['smf_tot']
