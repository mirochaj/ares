"""

shen2020.py

Author: Sylvia Chow
Ported to ares/data by JM

Getting from the raw Shen+ 2020 code+dataset to here is a bit of a process.

Everything we've done after cloning https://bitbucket.org/ShenXuejian/quasarlf
is listed here for posterity.
-----------------------------------------------------------------------------

1. Compile subroutines and edit homepath.

This will fail for a few of reasons out of the box. You must first:

- Navigate to pubtools/clib and compile (as detailed in how_to_compile.txt)
- Repeat the same command in pubtools/clib/specialuse for convolve_ao.c
- Navigate to pubtools/config.py and modify homepath on line 9 for your system. Make sure there's a / at the end of the string.
- You should now be able to run load_observations.py, which requires redshift as a command-line argument, e.g.:

> ipython load_observations.py 0.1

I've used ipython here since the plot is not saved -- this will make sure you see it before it disappears!

2. Modify source code to enable part 3 below.

As is, the key script load_observations.py requires redshift as a command line 
argument. We want to iterate over all redshifts to create a single dataset, so
the next steps are to:

- Add `redshift` as the first positional argument to get_fit_data and get_data.
- Make sure `redshift` is passed through to the get_fit_data on the fourth line 
of get_data.
- Comment out line 13 (redshift=float(sys.argv[1]))
- Comment out all the non-method code in load_observations.py (below function definitions
starting with matplotlib import) so it doesn't crash on import when `redshift` is found
in sys.path

3. Create files more immediately useful for us.

The data are loaded according to redshift and `dataid`, which encodes the band:

dataid=-1: B band
dataid=-2: Mid IR
dataid=-3: hard X-ray
dataid=-4: soft X-ray
dataid=-5: UV 1450

The following Python snippet will create a dictionary organized like ARES prefers
with one element per redshift, itself a dictionary containing the LF info. First, 
navigate to quasarlf/pubtools, then open a Python session and execute the following:

##
import pickle
import numpy as np
from load_observations import get_data

dataids = -np.arange(1, 6, 1)
redshifts = [0.1,0.2,0.5,1,2,3,4,5,6,7]
h07band_from_dataids = {-1:0, -2:3, -3:2, -4:1, -5: 5} # 4 is emission lines in h07
data = {z:{} for z in redshifts}
for z in redshifts:

    L = []
    phi = []
    err = []
    band = []

    for dataid in dataids:
        
        _Lbol, _phi, _err, _xerr = get_data(z, dataid)

        if len(_Lbol) == 0:
            continue

        L.extend(list(_Lbol))
        phi.extend(list(_phi))
        err.extend(list(_err))
        band.extend([h07band_from_dataids[dataid]]*len(_Lbol))
        
    # Save
    data[z]['L'] = np.array(L)
    data[z]['phi'] = np.array(phi)
    data[z]['err'] = np.array(err)
    data[z]['band'] = np.array(band)


with open('shen2020_qso_lfs.pkl', 'wb') as f:
    pickle.dump(data, f)

##

This file shen2020_qso_lfs.pkl is what is downloaded via `ares download shen2020_lfs`.

-----------------------------------------------------------------------------



"""

import pickle
from . import ARES

redshifts = [0.1,0.2,0.5,1,2,3,4,5,6,7]
with open(f"{ARES}/shen_lfs/shen2020_qso_lfs.pkl", "rb") as f:
    data = pickle.load(f)

bands = {0:'optical', 1:'soft x-ray', 2:'hard x-ray', 3:'ir', 4:'emission lines', 5: 'uv'}
