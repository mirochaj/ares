import numpy as np

info = \
{
 'reference': 'Bouwens et al., 2014',
 'data': 'Table 2', 
 'fits': 'Table 4', 
}

redshifts = [4, 5, 6, 7, 8]
wavelength = None
units = {'beta': 1.}


_data = \
{
 4: {'M': np.arange(-21.75, -15.25, 0.5),
     'beta': [-1.54, -1.61, -1.7, -1.8, -1.81, -1.9, -1.97, -1.99, -2.09, 
         -2.09, -2.23, -2.15, -2.15],
     'err': [0.07, 0.04, 0.03, 0.02, 0.03, 0.02, 0.06, 0.06, 0.08, 0.07,
         0.1, 0.12, 0.12],
     'sys': [0.06] * 13,
 },
 5 : {'M': np.array(list(np.arange(-21.75, -16.75, 0.5)) + [-16.5]),
     'beta': [-1.36, -1.62, -1.74, -1.85, -1.82, -2.01, -2.12, -2.16, -2.09, 
         -2.27, -2.16],
     'err': [0.48, 0.11, 0.05, 0.05, 0.04, 0.07, 0.1, 0.09, 0.1, 0.14, 0.17],
     'sys': [0.06] * 11,
 },
 6 : {'M': np.array(list(np.arange(-21.75, -17.25, 0.5)) + [-17.]),
     'beta': [-1.55, -1.58, -1.74, -1.9, -1.9, -2.22, -2.26, -2.19, -2.4, -2.24],
     'err': [0.17, 0.1, 0.1, 0.09, 0.13, 0.18, 0.14, 0.22, 0.3, 0.2],
     'sys': [0.08] * 10,
 },
 7 : {'M': np.array([-21.25, -19.95, -18.65, -17.35]),
     'beta': [-1.75, -1.89, -2.3, -2.42],
     'err': [0.18, 0.13, 0.18, 0.28],
     'sys': [0.13] * 4,
 },
 8 : {'M': np.array([-19.95, -18.65]),
     'beta': [-2.3, -1.41],
     'err': [0.01, 0.6],
     'sys': [0.27, 0.27],
 },   
}

# Keep track of filters used to compute slope
filter_names = \
{
 'F435W': r'$B_{435}$',
 'F606W': r'$V_{606}$',
 'F775W': r'$i_{775}$',
 'F814W': r'$I_{814}$',
 'F850LP': r'$z_{850}$',
 'F098M': r'$Y_{098}$',
 'F105W': r'$Y_{105}$',
 'F125W': r'$J_{125}$',
 'F140W': r'$JH_{140}$',
 'F160W': r'$H_{160}$',
}

# Actual filters used depending on fields and redshift
_filt_ers_z4 = ('F775W', 'F814W', 'F850LP', 'F098M', 'F105W',
 'F125W', 'F160W')
_filt_ers_z5 = ('F850LP', 'F098M', 'F105W', 'F125W', 'F160W')
_filt_ers_z6 = ('F098M', 'F105W', 'F125W', 'F160W')
_filt_ers_z7 = ('F098M', 'F125W', 'F160W')

# For XDF, HUDF09-1, HUDR09-2, z=4
_filt_xdf_z4 = ('F775W', 'F814W', 'F850LP', 'F125W')
# For XDF, HUDF09-1, HUDR09-2, z=5
_filt_xdf_z5 = ('F850LP', 'F105W', 'F160W')
# For XDF, HUDF09-1, HUDR09-2, z=6
_filt_xdf_z6 = ('F105W', 'F160W')
_filt_xdf_z7 = ('F125W', 'F160W')
_filt_xdf_z7 = ('F125W', 'F160W')

filt_deep = {4: _filt_xdf_z4,  5: _filt_xdf_z5,  6: _filt_xdf_z6, 7: _filt_xdf_z7,
    8: ('F140W', 'F160W')}
filt_shallow = {4: _filt_ers_z4, 5: _filt_ers_z5, 6: _filt_ers_z6, 7: _filt_ers_z7,
    8: ('F140W', 'F160W')}

data = {}
data['beta'] = {}
for key in _data:
    data['beta'][key] = {}
    for element in _data[key]:
        data['beta'][key][element] = np.array(_data[key][element])