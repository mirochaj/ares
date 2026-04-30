"""
Daddi, E., et al. 2007, ApJ, 670, 156
https://arxiv.org/abs/0705.2831v2

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 5, for I (Initial Mass Function) corrections.

"""

import numpy as np

info = \
{
 'reference':'Daddi, E., et al. 2007, ApJ, 670, 156',
 'data': 'Behroozi, Table 5', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [2.0]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 2.0: {'M': [5.8823529E+09, 1.7647059E+10, 5.8823529E+10],
    'phi': [-8.60205999132796, -8.65, -8.69897000433602],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
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
