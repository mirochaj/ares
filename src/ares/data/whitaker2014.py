"""

whitaker2014.py

Author: Jordan Mirocha
Affiliation: Caltech
Created on: Tue Jul 21 18:00:48 2026

Description:

"""

import numpy as np

info = \
{
 'reference':'Whitaker, K. E., et al. 2014, ApJ, 795, 104',
 'data': 'Table 2 (median stacks)', 
 'imf': ('chabrier, 2003', (0.1, 100.)),
}

cosmo = \
{
 'hubble_0': 0.7,
 'omega_m_0': 0.3,
 'omega_l_0': 0.7,
}
units = {'sfr': 'log10(sfr)', 'mass': 'log10(mass)'}
redshifts = [0.75, 1.25, 1.75, 2.25]
wavelength = 1600.

ULIM = -1e10

tmp_data = {}
tmp_data['sfr'] = \
{
 0.75: {'mass': [8.4, 8.7, 8.9, 9.1, 9.3, 9.5, 9.7, 9.9, 
                 10.1, 10.3, 10.5, 10.7, 10.9, 11.1],
    'sfr': [-0.60, -0.38, -0.20, -0.05, 0.23, 0.45, 0.65,
            0.78, 0.99, 1.06, 1.09, 1.22, 1.21, 1.27],
    'err': [0.5, 0.11, 0.05, 0.03, 0.02, 0.02, 0.02, 0.03,
            0.07, 0.05, 0.07, 0.07, 0.08, 0.10]
   },
 1.25: {'mass': [8.8, 9.1, 9.3, 9.5, 9.7, 9.9, 10.1, 10.3, 
                 10.5, 10.7, 10.9, 11.1, 11.3],
     'sfr': [-0.03, 0.17, 0.38, 0.64, 0.81, 1.02, 1.18, 1.35,
             1.47, 1.58, 1.69, 1.74, 1.81],
     'err': [0.13, 0.08, 0.05, 0.02, 0.02, 0.03, 0.04, 0.04, 
             0.05, 0.05, 0.08, 0.13, 0.11]
   },
 1.75: {'mass': [9.2, 9.4, 9.7, 9.9, 10.1, 10.3, 10.5, 10.7, 
                 10.9, 11.1, 11.3, 11.5],
     'sfr': [0.48, 0.70, 0.94, 1.15, 1.38, 1.54, 1.70, 1.83, 1.90,
             2.02, 2.19, 2.25],
     'err': [0.07, 0.03, 0.02, 0.03, 0.03, 0.04, 0.10, 0.07, 0.11,
             0.10, 0.09, 0.18]
   },
 2.25: {'mass': [9.3, 9.6, 9.8, 10.0, 10.3, 10.5, 10.7, 10.9, 
                 11.1, 11.3, 11.5],
     'sfr': [0.82, 1.05, 1.26, 1.46, 1.64, 1.86, 1.95, 2.07, 
             2.20, 2.32, 2.39],
     'err': [0.06, 0.03, 0.03, 0.03, 0.03, 0.05, 0.08, 0.06, 
             0.10, 0.12, 0.17]
   },
}

units = {'mass': 'log10(mass)', 'sfr': 'log10(sfr)'}

data = {}
data['sfr'] = {}

for group in ['sfr']:
    
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
        data[group][key]['mass'] = np.ma.array(subdata[key]['mass'], mask=mask) 
        data[group][key]['sfr'] = np.ma.array(subdata[key]['sfr'], mask=mask) 
        data[group][key]['err'] = tmp_data[group][key]['err']
