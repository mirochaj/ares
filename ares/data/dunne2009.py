"""
Dunne, L., et al. 2009, MNRAS, 394, 3
http://arxiv.org/abs/0808.3139v2

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 4, for I (Initial Mass Function) corrections.

"""

import numpy as np

info = \
{
 'reference':'Dunne, L., et al. 2009, MNRAS, 394, 3',
 'data': 'Behroozi, Table 4', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [0.5, 0.95, 1.4, 1.85]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 0.5: {'M': [9.3229011E+08, 2.3418069E+09, 5.8823529E+09, 1.4775803E+10, 2.9481602E+10, 5.8823529E+10],
    'phi': [-9.52287874528034, -9.52287874528034, -9.60205999132796, -9.69897000433602, -9.76955107862172, -9.82390874094432],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 0.95: {'M': [3.7115138E+09, 5.8823529E+09, 1.4775803E+10, 2.9481602E+10, 7.4054436E+10],
    'phi': [-9.0, -9.09691001300805, -9.15490195998574, -9.22184874961636, -9.15490195998574],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 1.4: {'M': [4.6725190E+09, 9.3229011E+09, 1.4775803E+10, 2.9481602E+10, 7.4054436E+10],
    'phi': [-8.69897000433602, -8.74472749489669, -8.79588001734407, -8.82390874094432, -8.85387196432176],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 1.85: {'M': [9.3229011E+09, 1.4775803E+10, 3.3078901E+10, 7.4054436E+10],
    'phi': [-8.39794000867204, -8.45593195564972, -8.52287874528034, -8.52287874528034],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
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
