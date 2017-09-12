"""

test_parallel_grid_results.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Jul 11 13:01:19 PDT 2017

Description: 

"""

import ares
import numpy as np

files = ['order_2_2', 'order_4_4', 'order_2_4', 'order_4_2', 'order_4']

for i, prefix in enumerate(files):
    anl = ares.analysis.ModelSet(prefix)

    
    z_C_now = anl.ExtractData('z_C')['z_C'].data
    if i > 0:
        z_C_pre = z_C_now
        z_C_now = anl.ExtractData('z_C')['z_C'].data
    else:
        continue
        
    assert np.allclose(z_C_pre, z_C_now), \
        "Detected difference between {!s}*.pkl and previous!".format(prefix)
        



