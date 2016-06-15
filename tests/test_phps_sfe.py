"""

test_phps_fstar.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jun 15 15:47:03 PDT 2016

Description: 

"""

import ares
import numpy as np
import matplotlib.pyplot as pl

pars_pl = \
{
'pop_fstar': 'php',
'php_func': 'pl',
'php_func_var': 'mass',
'php_func_par0': 1e-1,
'php_func_par1': 1e11,
'php_func_par2': 0.6,
'php_func_par3': 0.,
'php_ceil': 0.1,
}

pars_pl_w_zdep = \
{
'php_faux': 'pl',
'php_faux_var': '1+z',
'php_faux_meth': 'multiply',
'php_faux_par0': 1.,
'php_faux_par1': 7.,
'php_faux_par2': 1.,
}

pars_dpl = \
{
'pop_fstar': 'php',
'php_func': 'dpl_arbnorm',
'php_func_var': 'mass',
'php_func_par0': 1e-1,
'php_func_par1': 1e11,
'php_func_par2': 0.6,
'php_func_par3': 0.,
'php_func_par4': 1e14,      # Normalization mass
}

pars_pwpl = \
{
'pop_fstar': 'php',
'php_func': 'pwpl',
'php_func_var': 'mass',
'php_func_par0': 1e-1,
'php_func_par1': 0.6,
'php_func_par2': 1e-1,
'php_func_par3': 0.,
'php_func_par4': 1e11,
}

def test():
    
    Mh = np.logspace(7, 15, 200)
    
    ls = '-', '--', ':', '-.'
    lw = 2, 2, 4
    labels = ['pl_w_ceil', 'dpl', 'pwpl']
    for i, pars in enumerate([pars_pl, pars_dpl, pars_pwpl]):
        pop = ares.populations.GalaxyCohort(**pars)
        
        fnow = pop.SFE(6., Mh).copy()
        
        pl.loglog(Mh, fnow, ls=ls[i], label=labels[i], lw=lw[i])
            
        if i > 0:
            assert np.allclose(fnow[Mh <= 1e8], fprev[Mh <= 1e8], rtol=5e-2)
            assert np.allclose(fnow[Mh >= 1e14], fprev[Mh >= 1e14], rtol=5e-2)
    
        fprev = fnow.copy()
    
    pl.xlabel(r'$M_h / M_{\odot}$')
    pl.ylabel(r'$f_{\ast}$')
    pl.ylim(1e-4, 0.2)
    pl.legend(loc='lower right', fontsize=14)
    
    p = pars_pl.copy()
    p.update(pars_pl_w_zdep)
    pop = ares.populations.GalaxyCohort(**p)
    for i, z in enumerate([10, 15, 20]):
        pl.loglog(Mh, pop.SFE(z, Mh), color='m', ls=ls[i])
    
    pl.savefig('%s.png' % (__file__.rstrip('.py')))     
    
    
if __name__ == '__main__':
    test()
    