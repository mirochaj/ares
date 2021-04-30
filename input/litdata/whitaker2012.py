"""
Whitaker, K. E., et al. 2012, ApJ, 754, L29
http://arxiv.org/abs/1205.0547

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 5.

"""

import numpy as np

info = \
{
 'reference':'Whitaker, K. E., et al. 2012, ApJ, 754, L29',
 'data': 'Behroozi, Table 5', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [0.25, 0.75, 1.25, 1.75, 2.25]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 0.25: {'M': list(10**np.arange(9.5, 11.000, 0.500)),
    'phi': [-9.4, -9.6, -9.8],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 0.75: {'M': list(10**np.arange(9.5, 11.000, 0.500)),
    'phi': [-9.0, -9.1, -9.3],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 1.25: {'M': list(10**np.arange(10.0, 11.500, 0.500)),
    'phi': [-8.8, -9.0, -9.2],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 1.75: {'M': [3.1622777E+10, 1.0000000E+11],
    'phi': [-8.7, -9.1],
    'err': [(0.3, 0.3), (0.3, 0.3)]
   },
 2.25: {'M': [3.1622777E+10, 1.0000000E+11],
    'phi': [-8.5, -8.85],
    'err': [(0.3, 0.3), (0.3, 0.3)]
   },
}


units = {'ssfr': '1.'}

data = {}
data['ssfr'] = {}

for group in ['ssfr']:
    
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
