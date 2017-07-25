"""

test_pq_gen.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Thu Dec 15 14:31:54 PST 2016

Description: 

"""

import ares
import numpy as np

def test():
    # Make sure we can parameterize a bunch of things

    pars = ares.util.ParameterBundle('pop:sfe-dpl')
    
    pop = ares.populations.GalaxyPopulation(**pars)
    
    base_kwargs = \
    {
     'pq_func': 'dpl',
     'pq_func_var': 'Mh',
     'pq_func_par0': 0.1,
     'pq_func_par1': 3e11,
     'pq_func_par2': 0.6,
     'pq_func_par3': -0.6,
    }
    
    val = []
    parameterizable_things = \
        ['fstar', 'fshock', 'fpoll', 'fstall', 'rad_yield', 'fduty', 'fesc_LW']
    
    for par in parameterizable_things:
        
        pars = base_kwargs.copy()
        pars['pop_%s' % par] = 'pq'
        
        pop = ares.populations.GalaxyPopulation(**pars)
        
        func = pop.__getattr__(par)
        val.append(func(z=6, Mh=1e12))

    print val
    assert np.unique(val).size == 1, "Error in building ParameterizedQuantity!"
    
if __name__ == '__main__':
    test()
