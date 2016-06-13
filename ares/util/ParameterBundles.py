"""

ParameterBundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jun 10 11:00:05 PDT 2016

Description: 

"""

import numpy as np
from .ParameterFile import pop_id_num
from .SetDefaultParameterValues import SetAllDefaults
from .PrintInfo import header, footer, separator, line

#from .ProblemTypes import ProblemType

defaults = SetAllDefaults()

gs_options = ['4par', '2pop', 'lf']

def _add_pop_tag(par, num):
    """
    Add a population ID tag to each parameter.
    """
    
    prefix, idnum = pop_id_num(par)
    
    if idnum is not None:
        return '%s{%i}' % (prefix, num)
    else:
        return '%s{%i}' % (par, num)

class ParameterBundle(dict):
    def __init__(self, base=None, **kwargs):
        self.base = base
        self.kwargs = kwargs
        
        if base is not None:
            self._initialize(base, **kwargs)
        
    def _initialize(self, base, **kwargs):
        
        tmp = self.keys()
        for key in tmp:
            del self[key]
        
        for key in kwargs[base]:
            self[key] = kwargs[base][key] 
               
    def __getattr__(self, name):
        if name not in self.keys():
            pass    
        return self[name]
        
    def __add__(self, other):
        tmp = self.copy()
        
        # Make sure to not overwrite anything here!
        
        tmp.update(other)
        return tmp
    
    @property    
    def info(self):
        """ Print out info about this bundle. """
        
        header('Bundle Options')
        for key in self.kwargs.keys():
            if key == self.base:
                print line('*%s*' % self.base)
            else:
                print line(key)
            
        separator()
        print line('Run \'reinitialize\' with one of the above as argument to change.')
        footer()

_pop_sfe = \
{
 'pop_model': 'sfe',
 'pop_fstar': 'php',
 'pop_MAR': 'hmf',
 'php_Mfun': 'dpl',
 'php_Mfun_par0': 0.1,
 'php_Mfun_par1': 1e12,
 'php_Mfun_par2': 0.67,
 'php_Mfun_par3': 0.5,
}

_Population = \
{
 'fcoll': dict(pop_Tmin=1e4),
 'sfe': _pop_sfe.copy(),
 'mlf': {},
 'sfr': {},
 'ml': {},
}

_uvsed_toy = dict(pop_yield=4000, pop_yield_units='photons/b',
    pop_Emin=10.2, pop_Emax=24.6, pop_EminNorm=13.6, pop_EmaxNorm=24.6)
_uvsed_bpass = dict(pop_sed='eldridge2009', pop_binaries=False, pop_Z=0.02,
    pop_Emin=10.2, pop_Emax=24.6, pop_EminNorm=13.6, pop_EmaxNorm=24.6)
_xrsed = dict(pop_yield=2.6e39, pop_yield_units='erg/s/sfr',
    pop_Emin=2e2, pop_Emax=3e4, pop_EminNorm=5e2, pop_EmaxNorm=8e3,
    pop_sed='mcd', pop_logN=-np.inf, pop_solve_rte=False,
    pop_tau_Nz=1e3, pop_approx_tau='neutral')

_Spectrum = \
{
 'uv': _uvsed_toy,
 'xray': _xrsed,
 'bpass': _uvsed_bpass,
}

_Simulation = \
{
 'gs': [],
 'gp': [],
 'mpm': [],
 'mgb': [],
 'rt1d': [],
 'all': [],
}

_Physics = \
{
 'simple': {},
 'adv': {'include_He': True, 'secondary_ionization': 3,
    'approx_Salpha': 3},
}

class Spectrum(ParameterBundle):
    """
    Create a set of parameters needed for Spectra. In general, this means
    choosing (i) the basic underlying model and (ii) the yields.
    """
    
    def __init__(self, base=None):
        ParameterBundle.__init__(self, base, **_Spectrum)
        
    def reinitialize(self, base):
        ParameterBundle.__init__(self, base, **_Spectrum)    

class Population(ParameterBundle):
    """
    Create a set of parameters needed for Populations. In general, this means
    choosing (i) the basic underlying model and (ii) the yields.
    """
    
    def __init__(self, base=None):
        ParameterBundle.__init__(self, base, **_Population)
        
    def reinitialize(self, base):
        ParameterBundle.__init__(self, base, **_Population)    
        
    @property
    def num(self):
        if not hasattr(self, '_num'):
            self._num = None
        return self._num

    @num.setter
    def num(self, value):
        assert value % 1 == 0
        self._value = value
        
        for key in self.keys():
            self[_add_pop_tag(key, value)] = self.pop(key)
  

class Physics(ParameterBundle):
    def __init__(self):
        pass
        

class Simulation(ParameterBundle):
    def __init__(self, base=None):
        ParameterBundle.__init__(self, base, **_Simulation)
        
        
