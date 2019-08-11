"""

test_speed_gs_basic.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Thu  8 Aug 2019 18:30:02 EDT

Description: 

"""

import sys
import ares

N = int(sys.argv[1])

for i in range(N):
    print("Running iteration {}/{}".format(i+1, N))
    sim = ares.simulations.Global21cm(verbose=False, progress_bar=False)
    sim.run()
    
    
