"""
Kajisawa, M. et al., M. 2010, ApJ, 723, 129
http://arxiv.org/abs/1009.0002v1

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 5, for I (Initial Mass Function) corrections.

"""

import numpy as np

info = \
{
 'reference':'Kajisawa, M. et al., M. 2010, ApJ, 723, 129',
 'data': 'Behroozi, Table 5', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [0.75, 1.25, 2.0, 3.0]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 0.75: {'M': [1.7647059E+08, 5.7543994E+08, 1.7647059E+09, 5.8823529E+09, 1.7647059E+10, 5.8823529E+10],
    'phi': [-9.0, -9.15490195998574, -9.30102999566398, -9.39794000867204, -9.39794000867204, -9.69897000433602],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 1.25: {'M': [5.8823529E+08, 1.7647059E+09, 5.8823529E+09, 1.7647059E+10, 5.8823529E+10],
    'phi': [-8.88605664769316, -8.88605664769316, -8.88605664769316, -9.0, -9.30102999566398],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 2.0: {'M': [1.7647059E+09, 5.8823529E+09, 1.7647059E+10, 5.8823529E+10],
    'phi': [-8.52287874528034, -8.52287874528034, -8.52287874528034, -9.0],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 3.0: {'M': [1.7647059E+10],
    'phi': [-8.48148606012211],
    'err': [(0.3, 0.3)]
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
