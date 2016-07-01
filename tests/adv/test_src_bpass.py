"""

test_src_bpass.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jun 22 16:57:19 PDT 2016

Description: 

"""

import ares
import numpy as np

# Test metallicity interpolation, energy interpolation at fixed tsf
def test():
    # Test metallicity interpolation
    pop1 = ares.populations.SynthesisModel(pop_sed='eldridge2009', pop_Z=0.02)
    pop2 = ares.populations.SynthesisModel(pop_sed='eldridge2009', pop_Z=0.03)
    pop3 = ares.populations.SynthesisModel(pop_sed='eldridge2009', pop_Z=0.04)
    
    for i, E in enumerate([1,5,10,20]):
        fnu = np.array([pop1.Spectrum(E), pop3.Spectrum(E)])
        mi, ma = min(fnu), max(fnu)
        assert mi <= pop2.Spectrum(E) <= ma, \
            'Error in spectrum/metallicity interpolation!'

if __name__ == '__main__':
    test()
    