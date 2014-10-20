"""

test_21cm_cxrb.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Nov 11 13:02:00 MST 2013

Description: 

"""

import ares

src1 = \
{
 'source_type': 'star',
 'source_temperature': 1e4,
 'fstar': 1e-1,
 'Nion': 4e3,
 'Nlw': 9690.,
 'is_heat_src_igm': False,
 'is_ion_src_cgm': True,
 'is_ion_src_igm': False,
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
 'spectrum_Emin': 1e2,
 'spectrum_Emax': 3e4,
 'spectrum_EminNorm': 1e2,
 'spectrum_EmaxNorm': 3e4,
}

pars = \
{
 'Z': [1,2],
 'abundances': [1.,0.08],
 'approx_helium': False,
 'initial_ionization': [1.2e-3, 1e-8],
 'source_kwargs': [src1, src2],
 'spectrum_kwargs': [sed1, sed2],
 'EoR_xavg': 1e-2,
 'EoR_dlogx': 1e-3,
 'secondary_ionization': 3,
}

ax1, ax2, ax3, ax4 = None, None, None, None

sims = []
approx = [False, False]
colors = ['k', 'r']
for i, Z in enumerate([[1],[1,2]]):
    
    if i == 0:
        label = 'H-only'
    else:
        label = 'H+He'
    
    pars.update({'Z': Z, 'approx_helium': approx[i]})
    pars.update({'abundances': [1.0, 0.08]})
    sim = ares.simulations.Global21cm(final_redshift=3., track_extrema=0, 
        **pars)
    sim.run()
    
    anl = ares.analysis.Global21cm(sim)
    
    ax1 = anl.GlobalSignature(ax=ax1, fig=1, color=colors[i],
        label=label)
    ax2 = anl.IonizationHistory(ax=ax2, fig=2, color=colors[i], 
        show_legend=not i, show_xi=not i, show_xibar=not i)
    ax3 = anl.TemperatureHistory(ax=ax3, fig=3, color=colors[i], show_legend=not i)
    ax4 = anl.OpticalDepthHistory(ax=ax4, fig=4, color=colors[i])
    
    if i == 1:
        ax5 = anl.IonizationHistory(fig=5, element='he', color=colors[i])
        ax5 = anl.IonizationHistory(ax=ax5, show_xi=False, show_xibar=False,
            color='k', ls=':', label=r'$x_{\mathrm{HII}}$')
        ax5.set_title('(in bulk IGM)')
        ax5.set_ylim(1e-4, 1.5)
        
        
    sims.append(sim)



