"""

test_ps_density.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul 26 10:37:16 PDT 2017

Description: 

"""

import ares
import time
import numpy as np
import matplotlib.pyplot as pl

def test():

    # Input: just one redshift to be quick.
    redshifts = [6.]
    
    pop = ares.populations.GalaxyPopulation(hmf_load_ps=True)
    
    karr = np.exp(np.arange(-5, 5, 0.2))
    
    colors = 'b', 'g', 'm'
    for i, z in enumerate(redshifts):
        t1 = time.time()
        one_h = np.array(map(lambda kk: pop.halos.PS_OneHalo(z, kk), karr))
        t2 = time.time()
        
        t3 = time.time()
        two_h = np.array(map(lambda kk: pop.halos.PS_TwoHalo(z, kk), karr))
        t4 = time.time()
        
        print "One halo: %.2f sec" % (t2 - t1)
        print "Two halo: %.2f sec" % (t4 - t3)
    
        delta1 = one_h * karr**3 / (2. * np.pi**2)
        delta2 = two_h * karr**3 / (2. * np.pi**2)
        pl.loglog(karr, delta1, color=colors[i], ls=':', lw=3)
        pl.loglog(karr, delta2, color=colors[i], ls=':', lw=3)
        pl.loglog(karr, (delta1+delta2), color=colors[i], ls='-', lw=1,
            label='halo model')
        
        pop.halos.MF.update(z=z)
    
        k = pop.halos.MF.k
        powspec = k**3 * pop.halos.MF.power / (2. * np.pi**2)
        pl.loglog(k, powspec, ls='--', alpha=0.5, color=colors[i],
            label='linear')
        
    pl.legend(loc='upper left', fontsize=12)
    pl.xlim(1e-2, 1e2)
    pl.ylim(1e-3, 1e4)
    pl.xlabel(r'$k \ [\mathrm{cMpc}^{-1}]$')
    pl.ylabel(r'$\Delta^2_{\delta \delta}(k)$')
    
    #ps_dd = one_h + two_h
    #xi_dd = np.fft.ifft(ps_dd)

    pl.savefig('{!s}.png'.format(__file__[0:__file__.rfind('.')]))
    pl.close()
    
    assert True
    
if __name__ == '__main__':
    test()
    


#pl.figure(2)
#pl.loglog(2. * np.pi / karr, xi_dd)

#pl.figure(2)
#
#unfw = lambda kk: pop.halos.u_nfw(kk, 1e11, z=z)
#pl.loglog(k, map(unfw, k))

