"""

test_sources_sps.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Wed 29 Apr 2020 17:54:05 EDT

Description: Test basic functionality of SynthesisModel class.

"""

import ares
import numpy as np

def test():
    
    src = ares.sources.SynthesisModel(source_sed='eldridge2009', 
        source_sed_degrade=100, source_Z=0.02)
        
    Ebar = src.AveragePhotonEnergy(13.6, 1e2)
    assert 13.6 <= Ebar <= 1e2
    
    nu = src.frequencies
    ehat = src.emissivity_per_sfr
    
    beta = src.get_beta()
    assert -3 <= np.mean(beta) <= 2
    
if __name__ == '__main__':
    test()
