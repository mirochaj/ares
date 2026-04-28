"""
Feulner G. et al., 2005, ApJ, 633 L9-L12
http://arxiv.org/abs/astro-ph/0509197

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 5, for I (Initial Mass Function) corrections.

"""

import numpy as np

info = \
{
 'reference':'Feulner G. et al., 2005, ApJ, 633 L9-L12',
 'data': 'Behroozi, Table 5', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [0.5, 1.5, 1.0, 2.0, 3.0, 4.2]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 0.5: {'M': list(10**np.arange(9.0, 12.000, 1.000)),
    'phi': [-8.6, -9.2, -10.1],
    'err': [(0.4, 0.4), (0.4, 0.4), (0.4, 0.4)]
   },
 1.5: {'M': list(10**np.arange(9.0, 12.000, 1.000)),
    'phi': [-8.6, -9.1, -10.0],
    'err': [(0.4, 0.4), (0.4, 0.4), (0.4, 0.4)]
   },
 1.0: {'M': list(10**np.arange(9.0, 12.000, 1.000)),
    'phi': [-8.6, -9.1, -10.0],
    'err': [(0.4, 0.4), (0.4, 0.4), (0.4, 0.4)]
   },
 2.0: {'M': list(10**np.arange(9.0, 12.000, 1.000)),
    'phi': [-8.4, -8.8, -9.5],
    'err': [(0.4, 0.4), (0.4, 0.4), (0.4, 0.4)]
   },
 3.0: {'M': list(10**np.arange(9.0, 12.000, 1.000)),
    'phi': [-8.4, -8.8, -9.2],
    'err': [(0.4, 0.4), (0.4, 0.4), (0.4, 0.4)]
   },
 4.2: {'M': list(10**np.arange(9.0, 12.000, 1.000)),
    'phi': [-8.2, -8.8, -9.2],
    'err': [(0.4, 0.4), (0.4, 0.4), (0.4, 0.4)]
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
