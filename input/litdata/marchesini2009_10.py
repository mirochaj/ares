"""
(z=1.3-4)
Marchesini, D., van Dokkum, P. G., Forster Schreiber, N. M., Franx, M., Labbe, I., & Wuyts, S. 2009, ApJ, 701, 1765

(z=3-4)
Marchesini, D., et al. 2010, ApJ, 725, 1277

For smf_tot, values are corrected as seen in Behroozi et al. 2013 (http://arxiv.org/abs/1207.6105), for I (Initial Mass Function), D (Dust model) corrections.
"""

import numpy as np

info = \
{
 'reference':'',
 'data': '',
 'imf': ('chabrier', (0.1, 100.)), #didn't update this
}

redshifts = [1.65, 2.5, 3.5]
wavelength = 1600. #mulit-wavelength

ULIM = -1e10 #or this

fits = {}

# Table 3
tmp_data = {}
tmp_data['smf_tot'] = \
{
 1.65: {'M': [2.8157883E+11, 1.4914201E+11, 7.8995072E+10, 4.1840802E+10, 2.2161543E+10, 1.1738159E+10, 6.2172739E+09, 3.2930627E+09],
    'phi': [-5.00278782734556, -3.83678782734556, -3.17478782734555, -2.90578782734556, -2.73178782734555, -2.78078782734556, -2.71278782734556,
           -2.59378782734556],
    'err': [(0.283, 0.3), (0.168, 0.168), (0.1, 0.1), (0.11, 0.11), (0.1, 0.1), (0.1, 0.1), (0.245, 0.245), (0.289, 0.292)]
   },
 2.5: {'M': [2.8157883E+11, 1.5233511E+11, 8.2413812E+10, 4.4586153E+10, 2.4121261E+10, 1.3049684E+10, 7.0599236E+09],
    'phi': [-5.34678782734556, -3.94478782734555, -3.66378782734556, -3.32078782734556, -3.13178782734556, -3.19278782734556, -2.75778782734556],
    'err': [(0.333, 0.356), (0.113, 0.113), (0.103, 0.103), (0.118, 0.118), (0.215, 0.215), (0.291, 0.298), (0.298, 0.302)]
   },
 3.5: {'M': [2.6989821E+11, 1.9642646E+11, 1.6233045E+11, 8.7821328E+10, 4.7511638E+10, 2.5703958E+10, 1.3905929E+10],
    'phi': [-5.24578782734556, -4.80578782734556, -4.38078782734556, -3.95978782734556, -3.92778782734556, -4.03178782734556, -3.17778782734556],
    'err': [(0.39, 0.409), (0.24, 0.244), (0.322, 0.329), (0.217, 0.221), (0.424, 0.431), (0.453, 0.47), (0.496, 0.524)]
   },
}


tmp_data['smf_sf'] = \
{

}


tmp_data['smf_q'] = \
{

}


units = {'smf_tot': 'log10', 'smf_sf': 'log10', 'smf': 'log10', 'smf_q': 'log10'}

data = {}
data['smf_tot'] = {}
data['smf_sf'] = {}
data['smf_q'] = {}
for group in ['smf_tot', 'smf_sf', 'smf_q']:

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

#default is the star-forming galaxies data only
data['smf'] = data['smf_tot']
