"""

test_spec_synth.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 20 May 2019 15:13:21 EDT

Description: 

"""

import time
import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import s_per_myr

def test(show_bpass=False, oversample_age=30., dt_coarse=10):

    toy = ares.sources.SynthesisModelToy(source_dlam=10., source_Emin=1., 
        source_Emax=54.4, source_toysps_beta=-3.5,
        source_ssp=True, source_aging=True)
        
    # Just checking    
    E = toy.energies   
    dE = toy.dE
    dndE = toy.dndE 
    f = toy.frequencies
    
    pars = ares.util.ParameterBundle('mirocha2020:univ')
    pars['pop_sed'] = 'sps-toy'
    pars['pop_dust_yield'] = 0
    pars['pop_dlam'] = 1.
    pars['pop_thin_hist'] = 0
    pars['pop_scatter_mar'] = 0
    pars['pop_Tmin'] = None # So we don't have to read in HMF table for Mmin
    pars['pop_Mmin'] = 1e8
    pars['pop_synth_minimal'] = False
    pars['tau_clumpy'] = None
    pars['pop_sed_degrade'] = None
    
    # Prevent use of hmf table
    tarr = np.arange(50, 1000, 1.)[-1::-1]
    zarr = toy.cosm.z_of_t(tarr * s_per_myr)
    pars['pop_histories'] = {'t': tarr, 'z': zarr, 
        'MAR': np.ones((10, tarr.size)), 'nh': np.ones((10, tarr.size)),
        'Mh': 1e10 * np.ones((10, tarr.size))}
    
    pop1 = ares.populations.GalaxyPopulation(**pars)
    
    if show_bpass:
        src = ares.sources.SynthesisModel(source_sed='eldridge2009', 
            source_ssp=True)
    
    fig1, ax1 = pl.subplots(1, 1, num=1)
    fig2, ax2 = pl.subplots(1, 1, num=2)
    fig3, ax3 = pl.subplots(1, 1, num=3)
    
    colors = 'k', 'b', 'c'
    
    # Plot time evolution first
    ax1.set_xlabel(r'$t / \mathrm{Myr}$')
    ax1.set_ylabel(r'$L_{\nu}$')
    ax1.set_ylim(1e25, 1e34)
    
    ax2.set_ylim(1e28, 1e35)    
    ax2.set_xlabel(r'$\lambda / \AA$')
    ax1.set_ylabel(r'$L_{\nu}$')
    
    ##
    # Experiment with toy models.
    # Plot L(t) for three different wavelengths.
    # Plot full spectrum at a few different times
    ##
    tarr = np.logspace(0, 4, 100)
    
    for i, wave in enumerate([900., 1600., 2300.]):
            
        # Plot parameteric model solution
        #ax1.loglog(tarr, L(tarr, wave=wave), color=colors[i], ls='--')
    
        y2 = toy.data[np.argmin(np.abs(toy.wavelengths - wave)),:]
        ax1.loglog(toy.times, y2, color=colors[i], ls=':', lw=3, alpha=0.3)
    
        # Plot BPASS solution
        if not show_bpass:
            continue
            
        y1 = src.data[np.argmin(np.abs(src.wavelengths - wave)),:]
        ax1.loglog(src.times, y1, color=colors[i], ls='-',
            label=r'$\lambda = {} \AA$'.format(wave))
        
    ##
    # Plot spectra
    ##
    for i, _t in enumerate([1, 10, 100]):
        
        y2 = toy.data[:,np.argmin(np.abs(toy.times - _t))]
        ax2.loglog(toy.wavelengths, y2, color=colors[i], ls='--')
        
        # Plot BPASS solution
        if not show_bpass:
            continue
        
        y1 = src.data[:,np.argmin(np.abs(src.times - _t))]
        ax2.loglog(src.wavelengths, y1, color=colors[i], ls='-', alpha=0.2)
        
    ##
    # Make sure the spectra we put in are the spectra we get out.
    # e.g., do we recover UV slope of -2 if that's what we put in?
    ##
    
    beta = pop1.Beta(6., rest_wave=(1600., 2300.))
    mags = pop1.Magnitude(6.)
    
    ok = beta != -99999
    
    assert np.allclose(beta[ok==1], -2), \
        "Not recovering beta=-2! Mean={}".format(beta[ok==1].mean())
    
    ##
    # Test adaptive time-stepping in spectral synthesis.
    ##
    
    tarr1 = np.arange(0, 1000, 1.)
    tarr2 = np.arange(0, 1000, dt_coarse)
    sfh1 = np.ones_like(tarr1)
    sfh2 = np.ones_like(tarr2)
    
    ss = ares.static.SpectralSynthesis()
    ss.src = toy
    
    ss2 = ares.static.SpectralSynthesis()
    ss2.src = toy
    ss2.oversampling_enabled = False
    ss2.oversampling_below = oversample_age
        
    t1 = time.time()
    L1 = ss.Luminosity(sfh=sfh1, tarr=tarr1, load=False)
    t2 = time.time()
    
    print('dt=1', t2 - t1)
    
    t1 = time.time()
    L2 = ss.Luminosity(sfh=sfh2, tarr=tarr2, load=False)
    t2 = time.time()
    print('dt={}, oversampling ON:'.format(dt_coarse), t2 - t1)
    
    t1 = time.time()
    L3 = ss2.Luminosity(sfh=sfh2, tarr=tarr2, load=False)
    t2 = time.time()
    print('dt=10, oversampling OFF:', t2 - t1)
    
    ax3.semilogx(tarr1, L1, color='k')
    ax3.semilogx(tarr2[L2 > 0], L2[L2 > 0], color='b', lw=3, ls='--')
    ax3.semilogx(tarr2[L3 > 0], L3[L3 > 0], color='r', lw=2, ls=':')
    ax3.set_xlabel(r'$t / \mathrm{Myr}$')
    ax3.set_ylabel(r'$L_{\nu}$')
    
    def staircase(x, dx=10):
        
        N = x.size
        y = np.zeros_like(x)
        M = dx
        
        assert N % dx == 0
        
        ct = 0
        for i, xx in enumerate(x):
                    
            y[M*ct:M*(ct+1)] = ct + 1
            
            if i % M == 0:
                ct += 1
                        
        y[M*ct:M*(ct+1)] = ct + 1
        
        return y    
    
    ##
    # Test with 'staircase' SFH.
    ##
    fig4 = pl.figure(4)
    ax4a = fig4.add_subplot(211)
    ax4b = fig4.add_subplot(212)
    
    sfh1 = staircase(tarr1, dx=100)
    sfh2 = staircase(tarr2, dx=100//dt_coarse)
    ax4a.scatter(tarr1, sfh1, edgecolors='k', marker='.', s=1)
    ax4a.scatter(tarr2, sfh2, facecolors='none', edgecolors='b')
    ax4a.set_xlim(0, 300)
    ax4a.set_ylabel(r'SFR')
    ax4a.set_ylim(0, 5)
    
    L1 = ss.Luminosity(sfh=sfh1, tarr=tarr1)
    L2 = ss.Luminosity(sfh=sfh2, tarr=tarr2)
    L3 = ss2.Luminosity(sfh=sfh2, tarr=tarr2)
    
    ax4b.plot(tarr1[L1>0], L1[L1>0], color='k')
    ax4b.plot(tarr2[L2>0], L2[L2>0], color='b', ls='--', lw=3)
    ax4b.plot(tarr2[L3>0], L3[L3>0], color='r', ls=':', lw=2)
    ax4b.set_xlim(0, 300)
    ax4b.set_xlabel(r'$t / \mathrm{Myr}$')
    ax4b.set_ylabel(r'$L_{\nu}$')
    ax4b.set_ylim(0, 0.5e29)
    
    
    # Check validity of over-sampling for non-constant SFH
    # Just take mean error over long time as the solutions will differ
    # slightly (10%ish) at sharp discontinuities in SFH.
    err = []
    for j, _t in enumerate(tarr2):
        if _t == 0:
            continue
        
        i = np.argmin(np.abs(_t - tarr1))
    
        err.append(np.abs(L2[j] - L1[i]) / L1[i])
    
    
    assert np.mean(err) < 0.01
    
    
    for i in range(1, 5):
        pl.figure(i)
        pl.savefig('{0!s}_{1}.png'.format(__file__[0:__file__.rfind('.')], i))     
    
    pl.close('all')
    
    ##
    # Test batch mode
    ##
    
    sfh2 = np.array([sfh2] * 10)
    
    L2b = ss.Luminosity(sfh=sfh2, tarr=tarr2)
    L3b = ss2.Luminosity(sfh=sfh2, tarr=tarr2)
    
    assert np.all(L2b[0] == L2)
    assert np.all(L3b[0] == L3)
                
        #print("Mean error in L(t) with oversampling at t<{} Myr: {}".format(oversample_age,
        #    np.mean(err)))
            
if __name__ == '__main__':
    test()

    
    
    


    