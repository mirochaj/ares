"""

FitBundle.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sun Mar 19 11:42:25 PDT 2017

Description: Setup priors etc. in some semi-automated way.

"""

from ..util.ParameterFile import par_info
from ..util.SetDefaultPriorValues import *

try:
    from distpy import DistributionSet, UniformDistribution
except ImportError:
    pass

class FitHelper(object):
    """
    Make fitting high-dimensional models simpler.
    
    Tries to initialize the PriorSet
    """
    
    def __init__(self, **kwargs):
        self.base_kwargs = kwargs
        
    @property
    def prior_set(self):
        if not hasattr(self, '_prior_set'):
            self._prior_set = DistributionSet()
        return self._prior_set
    
    @property
    def is_log(self):
        if not hasattr(self, '_is_log'):
            self._is_log = []
        return self._is_log
   
    @property
    def parameters(self):
        if not hasattr(self, '_parameters'):
            self._parameters = []
        return self._parameters
   
    def run(self, pars):
        """
        Loop over parameters and depending on what they are, try
        to determine appropriate prior range.
        """
    
        for par in pars:
            
            self.parameters.append(par)
            
            if par in default_priors:
                prior, _is_log = default_priors[prior]
                distribution = UniformDistribution(*prior)
                self.prior_set.add_distribution(distribution, par)
                self.is_log.append(_is_log)
                continue
                
            ##
            # Harder case: ParameterizedQuantity
            ##
            prefix, popid, pqid = par_info(par)
            
            if pqid is None:
                raise ValueError('help')
            
            assert popid is None
                                                
            # Need to know function and independent variable
            parnum = int(par[11])
            func = self.base_kwargs['pq_func[{}]'.format(pqid)]
            var  = self.base_kwargs['pq_func_var[{}]'.format(pqid)]
            
            # See if this is nested
            parent = None
            for kw in self.base_kwargs:
                if self.base_kwargs[kw] == 'pq[{}]'.format(pqid):
                    parent = kw
                    break
                        
            # What is the PQID of the parent?
            if parent[0:2] != 'pq':
                pparnum = pfunc = pvar = None
            else:
                pparnum = int(parent[11])
                pprefix, ppopid, ppqid = par_info(parent)
                pfunc = self.base_kwargs['pq_func[{}]'.format(ppqid)]
                pvar  = self.base_kwargs['pq_func_var[{}]'.format(ppqid)]
            
            
            prior, _is_log = \
                default_prior(func, var, parnum, pfunc, pvar, pparnum)
                        
            self.is_log.append(_is_log)
            self.prior_set.add_distribution(UniformDistribution(*prior), par)
            
            
            
            
            
