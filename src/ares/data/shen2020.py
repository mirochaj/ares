"""

shen2020.py

Author: Sylvia Chow
Ported to ares/data by JM

Getting from the raw Shen+ 2020 code+dataset to here is a bit of a process.

Everything we've done after cloning https://bitbucket.org/ShenXuejian/quasarlf
is listed here for posterity.
-----------------------------------------------------------------------------












-----------------------------------------------------------------------------



"""

import pickle
from . import ARES

redshifts = [0.1,0.2,0.5,1,2,3,4,5,6,7]
with open(f"{ARES}/shen_lfs/shen2020_qso_lfs.pkl", "rb") as f:
    data = pickle.load(f)