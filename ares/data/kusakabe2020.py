"""

kusakabe2020.py

Tables 3 and 4, last chunk.
Figures 6 and 8

"""

import numpy as np

info = \
{
 'reference': 'Kusakabe et al., 2020, A&A submitted',
 'data': 'Figures 6 and 8',  
 'label': 'Kusakabe+ (2020)',
}


redshifts = 3.3, 4.1, 4.7, 5.6

# EW > 55 vs. redshift in -20.25 <= M1500 <= -18.75
faint = (-20.25, -18.75)
_x_55 = 0.06, 0.1, 0.18, 0.13
_e_55 = (0.04, 0.03), (0.09, 0.04), (0.13, 0.07), (0.12, 0.05)

_z_25 = 3.3, 4.1, 4.7, 5.6
_x_25 = 0.13, 0.25, 0.32, 0.13
_e_25 = (0.07, 0.05), (0.14, 0.09), (0.22, 0.11), (0.13, 0.05)

_mags = (-20.75, -19.50, -18.50)
_x_50 = [0.12, 0.04, 0.17]
_e_50 = (0.23, 0.06), (0.09, 0.02), (0.12, 0.05)


# vs. M1500 at z=4.1
data_vs_M = \
{
 'mags': _mags, 'x_LAE': np.array(_x_50), 'err': np.array(_e_50), 'EW': 50,
     'z': 4.1,
}

data_vs_z = \
{
 'EW25': {'x_LAE': np.array(_x_25), 'err': np.array(_e_25)},
 'EW55': {'x_LAE': np.array(_x_55), 'err': np.array(_e_55)},
}

