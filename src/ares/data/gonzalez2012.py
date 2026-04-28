"""
Gonzalez, V., et al., 2012, arXiv:1208.4362
http://arxiv.org/abs/1208.4362

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 5, for I (Initial Mass Function) corrections.

"""

import numpy as np

info = \
{
 'reference':'Gonzalez, V., et al., 2012, arXiv:1208.4362',
 'data': 'Behroozi, Table 5', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [4.0, 5.0, 6.0]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 4.0: {'M': [2.9481602E+09, 5.8823529E+08],
    'phi': [-8.49, -8.33],
    'err': [(0.3, 0.3), (0.3, 0.3)]
   },
 5.0: {'M': [2.9481602E+09, 5.8823529E+08],
    'phi': [-8.48, -8.21],
    'err': [(0.3, 0.3), (0.3, 0.3)]
   },
 6.0: {'M': [2.9481602E+09, 5.8823529E+08],
    'phi': [-7.9, -7.91],
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
