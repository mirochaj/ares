"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sat Jul 16 10:41:50 PDT 2016

Description: 

"""

from .GalaxyCohort import GalaxyCohort
from .GalaxyEnsemble import GalaxyEnsemble
from .GalaxyAggregate import GalaxyAggregate
from .Parameterized import ParametricPopulation

def GalaxyPopulation(**kwargs):
    """
    Return the appropriate Galaxy* instance depending on the value of 
    `pop_sfr_model`.
    """
    
    if 'pop_sfr_model' not in kwargs:
        model = 'fcoll'
    else:
        model = kwargs['pop_sfr_model']
    
    if model in ['sfe-func']:
        return GalaxyCohort(**kwargs)
    elif model in ['fcoll', 'sfrd-func']:
        return GalaxyAggregate(**kwargs)
    elif model in ['rates']:
        return ParametricPopulation(**kwargs)
    else:
        raise ValueError('Unrecognized sfrd_model %s' % model)


