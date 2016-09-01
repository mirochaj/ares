"""

test_galaxy_sfe_mlf.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri May 20 15:38:19 PDT 2016

Description: 

"""

import ares

def test():

    # First, a population where we model the star formation efficiency as a PL
    pars_sfe = \
    {
    'pop_sfr_model': 'sfe-func',
    'pop_fstar': 'php',
    'pop_mlf': None,
    
    # Put MAR in by hand?
    
    
    'php_func': 'pl',
    'php_func_var': 'mass',
    'php_func_par0': 1e-1,
    'php_func_par1': 1e11,
    'php_func_par2': 0.,
    }
    
    # Second, a population where we model the mass loading factor, and then 
    # figure out the SFE. We should get the same answer!
    pars_mlf = \
    {
    'pop_sfr_model': 'sfe-func',
    'pop_fstar': None,
    'pop_mlf': 'php',
    
    'php_func': 'pl',
    'php_func_var': 'mass',
    'php_func_par0': (1. / 1e-1) - 1.,  # Compute MLF from {SFE = 1 / (1  + MLF)}
    'php_func_par1': 1e11,
    'php_func_par2': 0.,
    }
        
    pop_sfe = ares.populations.GalaxyPopulation(**pars_sfe)
    pop_mlf = ares.populations.GalaxyPopulation(**pars_mlf)
    
    assert pop_sfe.fstar(10., 1e10) == pop_mlf.fstar(10., 1e10) == 1e-1, \
        "Mass evolution not working properly in both SFE and MLF approaches."
    
    # Third, a population where we parameterize the SFR function. Just take it
    # from one of the GalaxyCohort instances above.
    pars_sfr = \
    {
    'pop_fstar': None,
    'pop_mlf': None,
    'pop_sfr': 'php',
    
    'php_func': 'pl',
    'php_func_var': 'mass',
    'php_func_par0': (1. / 1e-1) - 1.,  # Compute MLF from {SFE = 1 / (1  + MLF)}
    'php_func_par1': 1e11,
    'php_func_par2': 0.,
    }    
        
         
    """
    Next: make sure redshift dependence works.
    """ 
    
    pars_sfe = \
    {
    'pop_sfr_model': 'sfe-func',
    'pop_fstar': 'php',
    
    'php_func': 'pl',
    'php_func_var': 'mass',
    'php_func_par0': 1e-1,
    'php_func_par1': 1e11,
    'php_func_par2': 0.,
    
    # The php_faux will get applied to the php_func with the same pop ID
    # and php ID.
    'php_faux': 'pl',
    'php_faux_var': 'redshift',
    'php_faux_meth': 'multiply',
    'php_faux_par0': 1.,
    'php_faux_par1': 10.,
    'php_faux_par2': 0.5,
    }  
    
    z1 = 10.
    z2 = 15.
    
    pop_sfe = ares.populations.GalaxyPopulation(**pars_sfe)
    
    correction = (z2 / z1)**pars_sfe['php_faux_par2']
    
    assert pop_sfe.fstar(z1, 1e10) * correction == pop_sfe.fstar(z2, 1e10), \
        "Redshift evolution not working properly for SFE."

    """
    Last test: introduce redshift dependent parameter.
    """
    
    pars_nest = \
    {
    'pop_fstar': 'php',
    'pop_mlf': None,
    
    'php_func': 'dpl',
    'php_func_var': 'mass',
    'php_func_par0': 'prefix',
    'php_func_par1': 1e11,
    'php_func_par2': 0.6,
    'php_func_par3': 0.6,
    
    'prefix_func': 'pl',
    'prefix_func_var': 'pl',
    'prefix_func_par0': 1.,
    'prefix_func_par1': 8.,
    'prefix_func_par2': -1.,
    }
    
if __name__ == '__main__':
    test()    
    



