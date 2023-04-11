"""
Inoue, A.K., 2011, MNRAS, 415, 2920

Supplementary online data that now seems to be missing...?
"""

import os
import numpy as np
from ares.data import ARES

path = ARES + '/inoue2011/'

fn_lines = '{}/LineList.txt'.format(path)
fn_data = '{}/LineRatio_nodust.txt'.format(path)

metallicities = np.array([2e-2, 8e-3, 4e-3, 4e-4, 1e-5, 1e-7, 0.])

line_waves = {}
line_ids = {}
line_info = []
with open(fn_lines, 'r') as f:
    for i, line in enumerate(f):

        line_spl = line.split()
        line_id = int(line_spl[0])

        if line_spl[1].strip().lower() == 'totl':
            line_str = line_str #+ line_spl[1].strip()
            wave = int(line_spl[2][0:-1])
        elif len(line_spl) == 4:
            line_str = line_spl[1] + ' ' + line_spl[2]
            wave = int(line_spl[3][0:-1])
        else:
            raise IndexError('help')

        if line_str in line_waves:
            line_waves[line_str].append(wave)
            line_ids[line_str].append(line_id)
        else:
            line_waves[line_str] = [wave]
            line_ids[line_str] = [line_id]

        line_info.append((line_id, wave))

line_info = np.array(line_info)

line_data = {}
with open(fn_data, 'r') as f:
    data = np.loadtxt(f)

def _read(Z, Ztol=1e-4):
    """
    Return line intensities relative to H-beta for given metallicity.
    """
    k = np.argmin(np.abs(Z - metallicities))
    if abs(metallicities[k] - Z) < Ztol:
        col = 2 * (k + 1)
        return data[:,col:col+2], line_info

    else:
        # Interpolate

        if metallicities[k] > Z:
            k += 1

        col1 = 2 * (k + 1)
        col2 = col1 + 1

        dat1 = data[:,col1:col1+2]
        dat2 = data[:,col2:col2+2]

        return 0.5 * (dat1 + dat2), line_info


def _load(Z, Ztol=1e-4):
    """
    Returns wavelengths and
    """
    data, info = _read(Z, Ztol=Ztol)

    waves = info[:,1]
    mean = data[:,0]
    std = data[:,1]

    return waves, mean, std

#lam, ew_wrt_hbeta = _load(0.02)
#
#wave = np.arange(lam.min(), lam.max()+1, 1.)
#lum = np.zeros_like(wave)
#for i in range(len(lam)):
#    j = np.argmin(np.abs(lam[i] - wave))
#    print(i, lam[i], j, ew_wrt_hbeta[i])
#    lum[j] = ew_wrt_hbeta[i]
#
#pl.plot(wave, lum)
#pl.ylim(1e-3, 10)
