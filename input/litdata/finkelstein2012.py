"""
Finkelstein et al., 2012, ApJ, 756, 164

Table 5.
"""

import numpy as np

info = \
{
 'reference': 'Finkelstein et al., 2012, ApJ, 756, 164',
 'data': 'Table 5', 
 'label': 'Finkelstein+ (2012)',
}

redshifts = [4, 5, 6, 7, 8]
wavelength = 1500.

ULIM = -1e10

# Table 5
tmp_data = {}
tmp_data['beta'] = \
{
 4: {'beta': [-2.22, -2.03, -1.88], 
     'Ms': [7.5, 8.5, 9.5], 
     'err': [(0.13, 0.05), (0.02, 0.04), (0.02, 0.02)]},
 5: {'beta': [-2.4, -2.15, -1.79], 
     'Ms': [7.5, 8.5, 9.5], 
     'err': [(0.22, 0.1), (0.06, 0.04), (0.09, 0.03)]},
 6: {'beta': [-2.59, -2.20, -1.78], 
     'Ms': [7.5, 8.5, 9.5], 
     'err': [(0.23, 0.1), (0.05, 0.16), (0.19, 0.1)]},   
 7: {'beta': [-2.68, -2.42, -1.76], 
     'Ms': [7.5, 8.5, 9.5], 
     'err': [(0.15, 0.24), (0.31, 0.11), (0.23, 0.33)]},  
 8: {'beta': [-2.5, -2.35, -1.6], 
     'Ms': [7.5, 8.5, 9.5], 
     'err': [(1.26, 0.43), (0.46, 0.16), (0.32, 0.54)]},        
}

data = {}
data['beta'] = {}
for key in tmp_data['beta']:
    data['beta'][key] = {}
    data['beta'][key]['Ms'] = np.array(tmp_data['beta'][key]['Ms']) 
    data['beta'][key]['beta'] = np.array(tmp_data['beta'][key]['beta'])
    data['beta'][key]['err'] = np.array(tmp_data['beta'][key]['err'])

data['slope_wrt_mass'] = \
{
 4: {'slope': 0.17, 'err': 0.03},
 5: {'slope': 0.30, 'err': 0.06},
 6: {'slope': 0.40, 'err': 0.1},
 7: {'slope': 0.46, 'err': 0.1},
 8: {'slope': 0.45, 'err': 0.37},
}






