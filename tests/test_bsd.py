"""

test_bsd.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Oct 14 13:51:17 PDT 2016

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import erfinv, erf

def test():
    pars = {'problem_type': 101}

    pars['fstar'] = 0.1
    pars['fesc'] = 0.2
    pars['Tmin'] = 2e4
    pars['Nlw'] = 9690.
    pars['cX'] = 2.6e39
    pars['fX'] = 0.2
    pars['fXh'] = 0.2
    pars['Nion'] = 4e3
    
    # Above yield zeta = 40

    pars['include_acorr'] = True
    pars['include_xcorr'] = False
    #pars['include_xcorr_wrt'] = 'ion', 'contrast'

    pars['pop_lya_fl{0}'] = True
    pars['pop_temp_fl{1}'] = True
    pars['pop_ion_fl{2}'] = True
    pars['include_bias'] = True
    pars['powspec_redshifts'] = [6, 8, 10, 15]

    # 
    pars['include_lya_fl'] = False
    pars['include_ion_fl'] = True
    pars['include_temp_fl'] = False
    pars['include_density_fl'] = True
    pars['powspec_force_Qi_gs'] = True

    # This seems to matter a lot!
    pars['powspec_rescale_Qion'] = False

    sim = ares.simulations.PowerSpectrum21cm(**pars)
    sim.run()
    
    ax = None
    colors = 'k', 'b', 'g', 'c', 'm', 'r', 'y', 'orange'
    for i, z in enumerate(pars['powspec_redshifts']):
        iz = np.argmin(np.abs(z - sim.history['z']))
        ax = sim.BubbleSizeDistribution(z=z, ax=ax, 
            label=r'$z={}, Q={}$'.format(z, round(sim.history['Qi'][iz], 2)),
            color=colors[i])

    ax.legend(loc='upper left', frameon=True, fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 1)

    for i in range(1, 2):
        pl.figure(i)
        pl.savefig('{!s}_{}.png'.format(__file__[0:__file__.rfind('.')], i))     
        pl.close()
        
if __name__ == '__main__':
    test()
