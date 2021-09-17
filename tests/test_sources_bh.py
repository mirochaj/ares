"""

test_sed_mcd.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu May  2 10:46:44 2013

Description: Plot a simple multi-color disk accretion spectrum.

"""

import ares
import numpy as np

def test():
    rmax = 1e2
    mass = 10.
    fsc = 0.1
    alpha = -1.5
    Emin = 1e2
    Emax = 1e4

    simpl = \
    {
     'source_type': 'bh',
     'source_mass': mass,
     'source_rmax': rmax,
     'source_sed': 'simpl',
     'source_Emin': Emin,
     'source_Emax': Emax,
     'source_EminNorm': Emin,
     'source_EmaxNorm': Emax,
     'source_alpha': alpha,
     'source_fsc': fsc,
     'source_logN': 22.,
    }

    mcd = \
    {
     'source_type': 'bh',
     'source_sed': 'mcd',
     'source_mass': mass,
     'source_rmax': rmax,
     'source_Emin': Emin,
     'source_Emax': Emax,
     'source_EminNorm': Emin,
     'source_EmaxNorm': Emax,
    }

    agn = \
    {
     'source_type': 'bh',
     'source_sed': 'sazonov2004',
     'source_Emin': Emin,
     'source_Emax': Emax,
     'source_EminNorm': Emin,
     'source_EmaxNorm': Emax,
    }

    bh_mcd = ares.sources.BlackHole(init_tabs=False, **mcd)
    bh_sim = ares.sources.BlackHole(init_tabs=False, **simpl)
    bh_s04 = ares.sources.BlackHole(init_tabs=False, **agn)

    Earr = np.logspace(2, 4, 100)

    for src in [bh_mcd, bh_sim, bh_s04]:
        sed = bh_mcd.Spectrum(Earr)


if __name__ == '__main__':
    test()
