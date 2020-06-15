"""

GalaxyPopulation.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sat Jul 16 10:41:50 PDT 2016

Description: 

"""

from .Toy import Toy
from ..util import ParameterFile
from .GalaxyCohort import GalaxyCohort
from .GalaxyEnsemble import GalaxyEnsemble
from .GalaxyAggregate import GalaxyAggregate
from .ClusterPopulation import ClusterPopulation
from .BlackHoleAggregate import BlackHoleAggregate
from .GalaxyHOD import GalaxyHOD
from ..util.SetDefaultParameterValues import PopulationParameters
from .Parameterized import ParametricPopulation, parametric_options
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

default_model = PopulationParameters()['pop_sfr_model']

def GalaxyPopulation(**kwargs):
    """
    Return the appropriate Galaxy* instance depending on if any quantities
    are being parameterized by hand.
    
    kwargs should NOT be ParameterFile instance. Still trying to remind
    myself why that is.
    
    """

    ##
    # First. Identify all ParameterizedQuantity parameters and 
    # if the user has directly supplied ionization/heating rates.
    ##
    
    Npq = 0
    Nparam = 0
    pqs = []
    for kwarg in kwargs:

        if isinstance(kwargs[kwarg], basestring):
            if kwargs[kwarg][0:2] == 'pq':
                Npq += 1
                pqs.append(kwarg)
        elif (kwarg in parametric_options) and (kwargs[kwarg]) is not None:
            Nparam += 1

    # If parametric, return right away   
    if Nparam > 0:
        assert Npq == 0
        return ParametricPopulation(**kwargs)
    
    # Allow pop_sfr_model to trump presence of PQs
    if 'pop_sfr_model' in kwargs:
        model = kwargs['pop_sfr_model']
    else:
        
        if Npq == 0:
            model = default_model  
        elif (Npq == 1) and pqs[0] == 'pop_sfrd':
            model = 'sfrd-func'
        else:
            if set(pqs).intersection(parametric_options):
                model = 'rates'
            else:   
                model = 'sfe-func'
                
    if model in ['sfe-func', 'sfr-func', 'mlf-func', 'sfe-tab', 'sfr-tab', 
        'uvlf', '21cmfast', 'smhm-func']:
        return GalaxyCohort(**kwargs)
    elif model in ['fcoll', 'sfrd-func', 'sfrd-tab', 'sfrd-class']:
        return GalaxyAggregate(**kwargs)
    elif model in ['frd-func']:
        return ClusterPopulation(**kwargs)
    elif model in ['ensemble']:
        return GalaxyEnsemble(**kwargs)
    elif model in ['rates']:
        return ParametricPopulation(**kwargs)
    elif model in ['bhmd']:
        return BlackHoleAggregate(**kwargs)
    elif model in ['toy']:
        return Toy(**kwargs)    
    elif model in ['hod']:
        return GalaxyHOD(**kwargs)
    else:
        raise ValueError('Unrecognized sfrd_model {!s}'.format(model))


