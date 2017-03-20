"""

SetDefaultPriorValues.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sun Mar 19 11:52:39 PDT 2017

Description: 

"""

import numpy as np

# Format is ((lo, hi), is_log)
default_priors = \
{

 'pop_rad_yield': ((37, 42), True),
 'pop_Z': ((-3, -1.4), True),
 'pop_fesc_LW': ((0.6, 1), False),
 'pop_fesc': ((-3, 0.), True),
 'pop_Tmin': ((2.5, 5.5), True),

}

##
# For ParameterizedQuantity 
##
defaults = \
{

 ('dpl', 'Mh'): [((-5., 0.), True), ((9., 14.), True), 
                 ((-2., 2.), False), ((-2., 2.), False)],
 ('pl', 'Mh'):  [((-5., 0.), True), ((9., 14.), True), 
                 ((-3., 3.), False)],                
 ('pl', '1+z'): [((-5., 0.), True), ((9., 14.), True), 
                 ((-3., 3.), False)],
 ('quad', '1+z'): [((-5., 0.), True), ((-4., 1.), True), 
                  ((-4., 1.), True), ((3., 12.), False)],
 ('dpl_arbnorm', 'Mh'): [((-5., 0.), True), ((9., 14.), True), 
                 ((-2., 2.), False), ((-2., 2.), False)],
                
 ('log_tanh_abs', 'Mh'):  [((0., 1.), False), ((0., 1.), False), 
                 ((9., 14.), True), ((0.01, 5.))],             
}


def default_prior(func, var, num, pfunc, pvar, pnum):
    result = defaults[(func, var)][num]
    if pfunc is None:
        pass
    else:
        # Generally, this means a PQ has evolving components. So,
        # because the zeroth parameter is typically the normalization, it 
        # should use the same prior as the zeroth parameter in the case
        # without evolution.   
        if num == 0:
            result = defaults[(pfunc, pvar)][pnum]
        else:
            pass
            
        
    return result    
        
        
        
        