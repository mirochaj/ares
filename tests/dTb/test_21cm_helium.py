"""

test_21cm_helium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import ares
import matplotlib.pyplot as pl

pl.rcParams['legend.fontsize'] = 12

src1 = \
{
 'source_type': 'star',
 'source_temperature': 1e4,
 'fstar': 1e-1,
 'Nion': 4e3,
 'Nlw': 9690.,
 'is_ion_src_cgm': True,
 'is_ion_src_igm': False,
 'is_heat_src_igm': False,
 'norm_by': 'lw',
 'approx_lya': True,
}

sed1 = {}

src2 = \
{
 'source_type': 'bh',
 'fstar': 1e-1,
 'fX': 1.,
 'norm_by': 'xray',
 'is_lya_src': False,
 'is_ion_src_cgm': False,
 'is_ion_src_igm': True,
 'approx_xray': False,
 'load_tau': True,
 'redshift_bins': 400,
}

sed2 = \
{
 'spectrum_type': 'pl',
 'spectrum_alpha': -1.5,
 'spectrum_logN': 20.,
 'spectrum_Emin': 2e2,
 'spectrum_Emax': 3e4,
}

pars = \
{
 'include_He': False,
 'approx_He': False,
 'source_kwargs': [src1, src2],
 'spectrum_kwargs': [sed1, sed2],
}

ax1, ax2, ax3, ax4 = None, None, None, None

sims = []
approx = [True, False]
colors = ['k', 'b', 'r']
for i, approx in enumerate(range(-1, 2)):
    
    if approx == -1:
        label = 'H-only'    
        pars.update({'include_He': False})
    else:
        pars.update({'include_He': True, 'approx_He': approx})
        if approx == 0:
            label = 'H+He (self-consistent)'
        
        elif approx == 1:
            label = 'H+He (approx)'
                  
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    anl = ares.analysis.Global21cm(sim)
    
    ax1 = anl.GlobalSignature(ax=ax1, fig=1, color=colors[i],
        label=label)
    ax2 = anl.IonizationHistory(ax=ax2, fig=2, color=colors[i], 
        show_legend=False, show_xi=False, show_xibar=False)
    ax3 = anl.TemperatureHistory(ax=ax3, fig=3, color=colors[i], 
        show_legend=False)
    
    if approx == 0:
        ax4 = anl.IonizationHistory(fig=4, element='he', color=colors[i])
        ax4 = anl.IonizationHistory(ax=ax4, show_xi=False, show_xibar=False,
            color='k', ls=':', label=r'$x_{\mathrm{HII}}$')
        ax4.set_title('(in bulk IGM)')
        ax4.set_ylim(1e-5, 1.5)
        
    sims.append(sim)

pl.show()


