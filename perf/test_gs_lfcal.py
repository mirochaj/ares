"""

test_speed_gs_lfcal.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Thu  8 Aug 2019 18:30:02 EDT

Description: 

"""

import sys
import ares

N = int(sys.argv[1])
pars = ares.util.ParameterBundle('mirocha2017:base')
pars['verbose'] = False
pars['progress_bar'] = False

for i in range(N):
    print("Running iteration {}/{}".format(i+1, N))
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    
