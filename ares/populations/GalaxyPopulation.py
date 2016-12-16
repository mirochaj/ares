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
from .Parameterized import ParametricPopulation, parameter_options

def GalaxyPopulation(**kwargs):
    """
    Return the appropriate Galaxy* instance depending on if any quantities
    are being parameterized by hand.
    """
    
    Npq = 0
    pqs = []
    for kwarg in kwargs:
        if type(kwargs[kwarg]) != str:
            continue
        if kwargs[kwarg][0:2] == 'pq':
            Npq += 1
            pqs.append(kwarg)
    
    if Npq == 0:
        model = 'fcoll'
    elif (Npq == 1) and pqs[0] == 'pop_sfrd':
        model = 'sfrd-func'
    else:
        if set(pqs).intersection(parameter_options):
            model = 'rates'
        else:   
            model = 'sfe-func'
    
    if model in ['sfe-func']:
        return GalaxyCohort(**kwargs)
    elif model in ['fcoll', 'sfrd-func']:
        return GalaxyAggregate(**kwargs)
    elif model in ['rates']:
        return ParametricPopulation(**kwargs)
    else:
        raise ValueError('Unrecognized sfrd_model %s' % model)


