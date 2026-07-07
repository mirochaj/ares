"""

hopkins2007.py


"""

import os
import numpy as np
from . import ARES

link = 'https://ui.adsabs.harvard.edu/abs/2007ApJ...654..731H/abstract'

_data_raw = np.loadtxt(f"{ARES}/hopkins_lfs/bol_lf_point_dump.dat", unpack=True, 
    usecols=[0,1,2,3,4,5], comments=';;//')

redshifts = list(np.sort(np.unique(_data_raw[0])))


data = {z:{} for z in redshifts}

# Loop through file and load data
_z = redshifts[0]

_L = []
_phi = []
_err = []
_band = []

band_ids = {0:'optical', 1:'soft x-ray', 2:'hard x-ray', 3:'ir', 4:'emission lines'}
# 
for i in range(_data_raw.shape[1]):

    _L.append(_data_raw[1,i])
    _phi.append(_data_raw[2,i])
    _err.append(_data_raw[3,i])
    _band.append(_data_raw[4,i])

    if i + 1 == _data_raw.shape[1] - 1:
        data[_z]['L'] = np.array(_L)
        data[_z]['phi'] = np.array(_phi)
        data[_z]['err'] = np.array(_err)
        data[_z]['band'] = np.array(_band)
        break
    
    # Start new datasets
    if _data_raw[0,i+1] != _z:
        data[_z]['L'] = np.array(_L)
        data[_z]['phi'] = np.array(_phi)
        data[_z]['err'] = np.array(_err)
        data[_z]['band'] = np.array(_band)

        _L = []
        _phi = []
        _err = []
        _band = []
        _z = _data_raw[0,i+1]


