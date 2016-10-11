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

pars_dpl_Mofz = \
{
'pop_fstar': 'php',
'php_func': 'dpl',
'php_func_var': 'mass',
'php_func_par0': 1e-1,
'php_func_par1': 'pl',
'php_func_par1_par0': 1e11,
'php_func_par1_par1': 6.,
'php_func_par1_par2': -1.,
'php_func_par2': 0.6,
'php_func_par3': -0.5,
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

pars_ramp = \
{
'pop_fstar': 'php',
'php_func': 'logramp',
'php_func_var': 'mass',
'php_func_par0': 1e-3,
'php_func_par1': 9,
'php_func_par2': 1e-1,
'php_func_par3': 11,
}

def test():
    
    Mh = np.logspace(7, 15, 200)
    
    ls = '-', '--', ':', '-.'
    lw = 2, 2, 4, 4
    labels = ['pl_w_ceil', 'dpl', 'pwpl', 'ramp']
    for i, pars in enumerate([pars_pl, pars_dpl, pars_pwpl, pars_ramp]):
        pop = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func', **pars)
        
        fnow = pop.SFE(6., Mh).copy()
        
        pl.loglog(Mh, fnow, ls=ls[i], label=labels[i], lw=lw[i])
            
        if (i > 0) and (labels[i] != 'ramp'):
            assert np.allclose(fnow[Mh <= 1e8], fprev[Mh <= 1e8], rtol=5e-2)
            assert np.allclose(fnow[Mh >= 1e14], fprev[Mh >= 1e14], rtol=5e-2)
    
        fprev = fnow.copy()
    
    pl.xlabel(r'$M_h / M_{\odot}$')
    pl.ylabel(r'$f_{\ast}$')
    pl.ylim(1e-4, 0.2)
    pl.legend(loc='lower right', fontsize=14)
    
    pl.savefig('%s_1.png' % (__file__.rstrip('.py')))     
    pl.close()
    
    p1 = pars_pl.copy()
    p1.update(pars_pl_w_zdep)
    pop1 = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func', 
        **p1)
    pop2 = ares.populations.GalaxyPopulation(pop_sfr_model='sfe-func', 
        **pars_dpl_Mofz)
    
    colors = ['k', 'b']
    ls = ['-', '--', ':']
    labels = [r'$f_{\ast}(z)$', r'$M_p(z)$']
    for j, pop in enumerate([pop1, pop2]):
        for i, z in enumerate([6, 15, 30]):
            pl.loglog(Mh, pop.SFE(z, Mh), color=colors[j], ls=ls[i],
                label=labels[j] if i == 0 else None)
    
    pl.xlabel(r'$M_h / M_{\odot}$')
    pl.ylabel(r'$f_{\ast}$')
    pl.ylim(1e-4, 0.2)
    pl.legend(loc='upper left', fontsize=14)
    
    pl.savefig('%s_2.png' % (__file__.rstrip('.py')))     
    
if __name__ == '__main__':
    test()
    