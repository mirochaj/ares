"""
Noeske, K. G., et al. 2007, ApJ, 660, L43
http://arxiv.org/abs/astro-ph/0703056v2

For ssfr, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), Table 5, for I (Initial Mass Function) corrections.

"""

import numpy as np

info = \
{
 'reference':'Noeske, K. G., et al. 2007, ApJ, 660, L43',
 'data': 'Behroozi, Table 5', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

redshifts = [0.5, 1.0]
wavelength = 1600.

ULIM = -1e10

fits = {}

# Table 1
tmp_data = {}
tmp_data['ssfr'] = \
{
 0.5: {'M': [6.3095734E+09, 1.0000000E+10, 1.5848932E+10, 2.5118864E+10, 3.9810717E+10, 6.3095734E+10, 1.0000000E+11, 1.5848932E+11, 2.5118864E+11],
    'phi': [-9.32258931209275, -9.28912138231483, -9.51407920780546, -9.79372614683001, -9.8173840679684, -9.8764969330579, -10.2203313727929,
           -10.3870866433571, -10.5174185464455],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
   },
 1.0: {'M': [1.2589254E+10, 1.9952623E+10, 3.1622777E+10, 5.0118723E+10, 7.9432823E+10, 1.2589254E+11, 1.9952623E+11, 3.1622777E+11],
    'phi': [-8.92007496792566, -9.05708605318108, -9.22584215073632, -9.32957944145149, -9.59738771540091, -9.77518272454617, -9.74281773821207,
           -10.0030841831815],
    'err': [(0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3), (0.3, 0.3)]
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
