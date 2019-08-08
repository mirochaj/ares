"""

test_sam_histories.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Thu  8 Aug 2019 18:38:06 EDT

Description: 

"""

import sys
import ares

N = int(sys.argv[1])
pars = ares.util.ParameterBundle('in_prep:base').pars_by_pop(0, 1)
pars['verbose'] = False
pars['progress_bar'] = False

for i in range(N):
    print("Running iteration {}/{}".format(i+1, N))
    pop = ares.populations.GalaxyPopulation(**pars)
    hist = pop.histories
    
    
    


