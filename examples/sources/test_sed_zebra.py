"""

test_sed_zebra.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Jan 22 17:15:49 MST 2014

Description:

"""

import rt1d
import numpy as np
import matplotlib.pyplot as pl

mass = 10.
rmax = 1e3
f_scatter = [0.1, 0.5, 1.0]
gamma = [-1.5]
Emin = 1e-1

zebra = \
{
    'source_type': 'bh', 
    'source_temperature': 3e4,
    'source_mass': mass,
    'spectrum_type': 'zebra',
    'spectrum_Emin': Emin,
    'spectrum_Emax': 5e4,
    'spectrum_alpha': -0.5,
    'spectrum_fsc': 1.0,
    'spectrum_logN': -np.inf,
}

bbpars = \
{
    'source_type': 'star', 
    'source_temperature': 3e4,
    'spectrum_type': 'bb',
    'spectrum_Emin': Emin,
    'spectrum_Emax': 1e2,
}

bbsrc = rt1d.sources.RadiationSource(init_tabs=False, **bbpars)

bb = rt1d.analyze.Source(bbsrc)
ax = bb.PlotSpectrum(color='k')
    
ls = ['-', '--', ':']
colors = ['b', 'g', 'r', 'm']
for i, fsc in enumerate(f_scatter):
    zebra.update({'spectrum_fsc': fsc})
    for j, alpha in enumerate(gamma):
        zebra.update({'spectrum_alpha': alpha})
        
        bh_zebra = rt1d.sources.RadiationSource(init_tabs=False, **zebra)
        bh2 = rt1d.analyze.Source(bh_zebra)
        
        if j == 0:
            label = r'$f_{\mathrm{sc}} = %g$' % fsc
        else:
            label = None
            
        ax = bh2.PlotSpectrum(color=colors[i], ls=ls[j], label=label,
            bins=250)
        pl.draw()
        
ax.legend(loc='upper right')
ax.loglog([10.2] * 2, [1e-3, 1.5], color='k', ls=':')
ax.annotate(r'$\mathrm{Ly}-\alpha$', (10.2, 1e-1), rotation=90, ha='right',
    va='bottom')
ax.loglog([13.6] * 2, [1e-3, 1.5], color='k', ls='--')
ax.annotate(r'$13.6 \ \mathrm{eV}$', (13.6, 1e-1), rotation=90, ha='right',
    va='bottom')
ax.set_ylim(1e-3, 1.5)
ax.set_xlim(1., 1e2)
ax.fill_betweenx([1e-8, 1e-3], 5e2, 8e3, alpha=0.5, color='gray')
pl.draw()


