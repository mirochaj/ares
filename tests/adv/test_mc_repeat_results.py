"""

test_parallel_grid_results.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jul 11 13:01:19 PDT 2017

Description: 

"""

import ares
import numpy as np

files = ['test_mc_1', 'test_mc_2', 'test_mc_3']

for i, prefix in enumerate(files):
    anl = ares.analysis.ModelSet(prefix)

    z_C_now = anl.ExtractData('z_C')['z_C'].data
    if i > 0:
        z_C_pre = z_C_now
        z_C_now = anl.ExtractData('z_C')['z_C'].data
    else:
        continue
        
    if i <= 1:
        assert np.allclose(z_C_pre, z_C_now), \
            "Detected difference between %s*.pkl and previous!" % prefix
    else:
        assert np.allclose(z_C_pre, z_C_now), \
            "%s*.pkl should be different!" % prefix
        



