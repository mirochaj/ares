"""

test_physics_nebula.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sun  7 Jun 2020 16:31:42 EDT

Description: 

"""

import sys
import ares
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
from ares.physics.Constants import h_p, c, erg_per_ev

def test(spsmodel='eldridge2009'):    

    # Setup pure continuum source
    pars_con = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_con.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_con['pop_Z'] = 1e-3
    pars_con['pop_sed'] = spsmodel
    pars_con['pop_nebular'] = 0
    pop_con = ares.populations.GalaxyPopulation(**pars_con)
    dwdn = pop_con.src.dwdn

    pars_ares = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
    pars_ares.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_ares['pop_Z'] = 1e-3
    pars_ares['pop_sed'] = spsmodel
    pars_ares['pop_nebular'] = 2
    pop_ares = ares.populations.GalaxyPopulation(**pars_ares)

    # Setup source with BPASS-generated (CLOUDY) nebular emission
    pars_sps = pars_ares.copy()
    pars_sps.update(ares.util.ParameterBundle('testing:galaxies'))
    pars_sps['pop_nebular'] = 1
    pars_sps['pop_fesc'] = 0.
    pars_sps['pop_nebula_Tgas'] = 2e4
    pop_sps = ares.populations.GalaxyPopulation(**pars_sps)

    code = 'bpass' if pars_ares['pop_sed'] == 'eldridge2009' else 's99'

    fig = pl.figure(tight_layout=False, figsize=(8, 8), num=1)
    gs = gridspec.GridSpec(2, 1, hspace=0.2, wspace=0.2, figure=fig,
        height_ratios=(2, 1))

    ax_spec = fig.add_subplot(gs[0,0])
    ax_err = fig.add_subplot(gs[1,0])

    colors = 'k', 'b', 'c', 'm', 'r'
    for k, t in enumerate([1, 5, 10, 20, 50]):
        i = np.argmin(np.abs(pop_ares.src.times - t))
    
        err = np.abs(pop_ares.src.data[:,i] - pop_sps.src.data[:,i]) / pop_sps.src.data[:,i]
        ax_err.semilogx(pop_sps.src.wavelengths, err, color=colors[k],
            label=r'$t = {}$ Myr'.format(t))
                
        if t > 1:
            continue

        # Plot BPASS continuum vs. BPASS nebular solution
        ax_spec.loglog(pop_con.src.wavelengths, pop_con.src.data[:,i] * dwdn, color='k', 
            alpha=1, lw=1, label=r'{} continuum'.format(code))
        ax_spec.loglog(pop_ares.src.wavelengths, pop_ares.src.data[:,i] * dwdn, color='b', 
            alpha=0.5, label='{} continuum + ares nebula'.format(code))
        ax_spec.loglog(pop_sps.src.wavelengths, pop_sps.src.data[:,i] * dwdn, color='r', 
            alpha=0.5, label='{} continuum + {} nebula'.format(code, code))
        ax_spec.annotate(r'$t = {}$ Myr'.format(t), (0.05, 0.95),
            xycoords='axes fraction')
        
    ax_err.set_xlabel(r'Wavelength $[\AA]$')
    ax_spec.set_ylim(1e24, 1e28)
    ax_spec.set_xlim(1e2, 1e5)
    ax_err.set_xlim(1e2, 1e5)
    ax_err.set_ylim(0, 1)
    ax_spec.set_ylabel(r'$L_{\nu} \ [\mathrm{erg} \ \mathrm{s}^{-1} \mathrm{Hz}^{-1}]$')
    ax_err.set_ylabel(r'rel. error')
    ax_err.legend(loc='upper left', fontsize=10)
    ax_spec.legend(loc='upper right', fontsize=10)

    if pars_ares['pop_sed'] == 'eldridge2009':
        pl.savefig('ares_v_bpass.png')
    elif pars_ares['pop_sed'] == 'leitherer1999':
        pl.savefig('ares_v_starburst99.png')
    else:
        raise IOError('Unrecognized model!')
    
    
if __name__ == '__main__':
    test()
    
    