"""

test_uvlf.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Mon Mar 26 14:18:11 PDT 2018

Description: Recreate Figure 15 in Bouwens+ 2015

"""


import ares
import numpy as np
import matplotlib.pyplot as pl

gpop = ares.analysis.GalaxyPopulation()

redshifts = [3.8, 4.9, 5.9, 6.9, 7.9]

pars = \
{
 'pop_sfr_model': 'uvlf',
 
 # Stellar pop + fesc
 'pop_sed': 'eldridge2009',
 'pop_binaries': False,
 'pop_Z': 0.02,
 'pop_Emin': 10.19,
 'pop_Emax': 24.6,
 'pop_rad_yield': 'from_sed', # EminNorm and EmaxNorm arbitrary now
                                 # should make this automatic
 'pop_uvlf': 'pq',
 'pq_func': 'schechter_evol',
 'pq_func_var': 'MUV',
 'pq_func_var2': 'z',
               
 # Bouwens+ 2015 Table 6 for z=5.9
 #'pq_func_par0[0]': 0.39e-3,
 #'pq_func_par1[0]': -21.1,
 #'pq_func_par2[0]': -1.90,    
 #
 # phi_star
 'pq_func_par0': np.log10(0.47e-3),
 'pq_func_par4': -0.27,
 
 # z-pivot
 'pq_func_par3': 6.,
 
 # Mstar
 'pq_func_par1': -20.95,
 'pq_func_par5': 0.01,
 
 # alpha
 'pq_func_par2': -1.87,
 'pq_func_par6': -0.1,
  
}

def test():

    mags = np.arange(-24, -10, 0.1)    
    
    # Schechter fit from B15
    pop_sch = ares.populations.GalaxyPopulation(**pars)
    
    # DPL SFE fit from my paper
    m17 = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0,1)
    
    # Test suite doesn't download BPASS models, so supply L1600 by hand.
    m17['pop_sed'] = None
    m17['pop_L1600_per_sfr'] = 1.019e28
    
    # Make the population
    pop_dpl = ares.populations.GalaxyPopulation(**m17)
    
    ax1 = None
    ax2 = None
    colors = 'b', 'g', 'gray', 'k', 'r'
    for i, z in enumerate(redshifts):
     
        # Plot the data
        ax1 = gpop.Plot(z=z, sources='bouwens2015', ax=ax1, color=colors[i],
            mec=colors[i], label=r'$z\sim {:d}$'.format(int(round(z, 0))))
        
        # Plot the Bouwens Schechter fit
        ax1.semilogy(mags, pop_sch.LuminosityFunction(z, mags), color=colors[i], 
            ls='-', lw=1)
    
        # My 2017 paper only looked at z > 6
        if z < 5:
            continue
    
        ax2 = gpop.Plot(z=z, sources='bouwens2015', ax=ax2, color=colors[i], fig=2,
            mec=colors[i], label=r'$z\sim {:d}$'.format(int(round(z, 0))))
            
        # Plot the physical model fit
        ax2.semilogy(mags, pop_dpl.LuminosityFunction(z, mags), color=colors[i], 
            ls='-', lw=1)
    
    
    # Add z = 10 models and data from Oesch+ 2014
    ax1.semilogy(mags, pop_sch.UVLF_M(MUV=mags, z=10.), color='m', 
        ls=':', alpha=1, lw=1)
    ax2.semilogy(*pop_dpl.phi_of_M(z=10), color='m', ls=':', alpha=0.5)
        
    ax1 = gpop.Plot(z=10., sources='oesch2014', ax=ax1, color='m',
        mec='m', label=r'$z\sim10$') 
    ax2 = gpop.Plot(z=10., sources='oesch2014', ax=ax2, color='m',
        mec='m', label=r'$z\sim10$')       
    
    # Make nice
    ax1.legend(loc='lower right', fontsize=12, numpoints=1)
    ax2.legend(loc='lower right', fontsize=12, numpoints=1)
    
    ax1.set_title('Bouwens+ 2015 Fig. 15 (reproduced)', fontsize=12)
    ax2.set_title(r'Models from Mirocha+ 2017 (cal at $z \sim 6$ only)', 
        fontsize=12)
    
    ax1.set_xlim(-24, -10)
    ax1.set_ylim(1e-7, 1)
    ax2.set_xlim(-24, -10)
    ax2.set_ylim(1e-7, 1)
    
    pl.show()
    
    pl.figure(1)
    pl.savefig('{!s}_1.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()
    pl.figure(2)
    pl.savefig('{!s}_2.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()    
        
    assert True
    
if __name__ == '__main__':
    test()
    