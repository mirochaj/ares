"""

ParameterBundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jun 10 11:00:05 PDT 2016

Description: 

"""

import numpy as np
from .ReadData import read_lit
from .ParameterFile import pop_id_num
from .ProblemTypes import ProblemType
from .PrintInfo import header, footer, separator, line

def _add_pop_tag(par, num):
    """
    Add a population ID tag to each parameter.
    """
    
    prefix, idnum = pop_id_num(par)
    
    if idnum is not None:
        return '%s{%i}' % (prefix, num)
    else:
        return '%s{%i}' % (par, num)

_pop_fcoll = \
{
 'pop_sfr_model': 'fcoll',
 'pop_Tmin': 1e4,
 'pop_Tmax': None,
}

_pop_user_sfrd = \
{

 'pop_sfr_model': 'sfrd-func',

 'pop_sfrd': 'php[0]',
 'php_func[0]': 'dpl',
 'php_func_var[0]': 'redshift',
 'php_func_par0[0]': 1e-6,
 'php_func_par1[0]': 15.,
 'php_func_par2[0]': -5.,
 'php_func_par3[0]': -8.,
 
}

_sed_toy = \
{
 'pop_sed_model': False,
 'pop_Nion': 4e3,
 'pop_Nlw': 9690,
 'pop_fX': 1.0,
 'pop_fesc': 0.1,
}

_pop_sfe = \
{
 'pop_sfr_model': 'sfe-dpl',
 'pop_fstar': 'php',
 'php_func': 'dpl',
 'php_func_par0': 0.1,
 'php_func_par1': 3e11,
 'php_func_par2': 0.6,
 'php_func_par3': -0.6,

 # Redshift dependent parameters here
}

_pop_sfe_ext = \
{
 'php_faux': 'plexp',
 'php_faux_var': 'mass',
 'php_faux_meth': 'add',
 'php_faux_par0': 0.005,
 'php_faux_par1': 1e9,
 'php_faux_par2': 0.01,
 'php_faux_par3': 1e10,
}

_pop_mlf = \
{
 'pop_sfr_model': 'mlf',
 'pop_fstar': None,
 'pop_mlf': 'php',
 'pop_MAR': 'hmf',
 
 'php_func': 'dpl',
 'php_func_par0': 0.1,
 'php_func_par1': 1e12,
 'php_func_par2': 0.67,
 'php_func_par3': 0.5,
}

_sed_uv = \
{
 # Emits LW and LyC
 "pop_lya_src": True,
 "pop_ion_src_cgm": True,
 "pop_ion_src_igm": False,
 "pop_heat_src_igm": False,
 
 "pop_fesc": 0.1,
 "pop_fesc_LW": 1.0,
 
 'pop_sed': 'pl',
 'pop_alpha': 1.0,
 "pop_Emin": 10.2,
 "pop_Emax": 24.6,
 "pop_EminNorm": 13.6,
 "pop_EmaxNorm": 24.6,        
 "pop_yield": 4e3, 
 "pop_yield_units": 'photons/baryon',
}

_sed_lw = _sed_uv.copy()
_sed_lw['pop_ion_src_cgm'] = False

_sed_lyc = _sed_uv.copy()
_sed_lyc['pop_lya_src'] = False

_pop_synth = \
{
 # Stellar pop + fesc
 'pop_sed': 'eldridge2009',
 'pop_binaries': False,
 'pop_Z': 0.02,
 'pop_Emin': 1,
 'pop_Emax': 1e2,
 'pop_yield': 'from_sed',
}

_sed_xr = \
{

 # Emits X-rays
 "pop_lya_src": False,
 "pop_ion_src_cgm": False,
 "pop_ion_src_igm": True,
 "pop_heat_src_cgm": False,
 "pop_heat_src_igm": True,
 
 "pop_sed": 'pl',
 "pop_alpha": -1.5,
 'pop_logN': -np.inf,
 
 "pop_Emin": 2e2,
 "pop_Emax": 3e4,
 "pop_EminNorm": 5e2,
 "pop_EmaxNorm": 8e3,
 
 "pop_Ex": 500.,
 "pop_yield": 2.6e39, 
 "pop_yield_units": 'erg/s/SFR',
}

_crte_xrb = \
{
 "pop_solve_rte": True, 
 "pop_tau_Nz": 400,
 "pop_approx_tau": 'neutral',
}

_crte_lwb = _crte_xrb.copy()
_crte_lwb['pop_solve_rte'] = (10.2, 13.6)
_crte_lwb['pop_approx_tau'] = True

# Some different spectral models
_uvsed_toy = dict(pop_yield=4000, pop_yield_units='photons/b',
    pop_Emin=10.2, pop_Emax=24.6, pop_EminNorm=13.6, pop_EmaxNorm=24.6)
_uvsed_bpass = dict(pop_sed='eldridge2009', pop_binaries=False, pop_Z=0.02,
    pop_Emin=10.2, pop_Emax=24.6, pop_EminNorm=13.6, pop_EmaxNorm=24.6)
_uvsed_s99 = _uvsed_bpass.copy()
_uvsed_s99['pop_sed'] = 'leitherer1999'

_mcd = _sed_xr.copy()
_mcd['pop_sed'] = 'mcd'
_pl = _mcd.copy()
_pl['pop_sed'] = 'pl'

_Bundles = \
{
 'pop': {'fcoll': _pop_fcoll, 'sfe-dpl': _pop_sfe, 'sfe-func': _pop_sfe, 
    'sfrd-func': _pop_user_sfrd, 'sfe-pl-ext': _pop_sfe_ext},
 'sed': {'uv': _sed_uv, 'lw': _sed_lw, 'lyc': _sed_lyc, 
         'xray':_sed_xr, 'pl': _pl, 'mcd': _mcd, 'toy': _sed_toy,
         'bpass': _uvsed_bpass, 's99': _uvsed_s99},
 'physics': {'xrb': _crte_xrb, 'lwb': _crte_lwb},
}

class ParameterBundle(dict):
    def __init__(self, bundle=None, id_num=None, **kwargs):
        self.bundle = bundle
        self.kwargs = kwargs
        
        # data should be a string
        if bundle is not None:
            self._initialize(bundle, **kwargs)
            if id_num is not None:
                self.num = id_num
        else:
            for key in kwargs:
                self[key] = kwargs[key]
            
    def _initialize(self, bundle, **kwargs):
        
        # Clear out
        tmp = self.keys()
        for key in tmp:
            del self[key]
        
        # Assume format: "modeltype:model", e.g., "pop:fcoll" or "sed:uv"
        pre, post = bundle.split(':')
        
        if pre in _Bundles.keys():
            kw = _Bundles[pre][post]
        # Assume format: "paperyear:modelname", e.g., "mirocha2016:dpl"
        elif pre == 'prob':
            kw = ProblemType(float(post))
        else:
            kw = read_lit(pre).__dict__[post]
        
        pars = kw.keys()

        for key in pars:
            self[key] = kw[key]    

    def __getattr__(self, name):
        if name not in self.keys():
            pass
        return self[name]
        
    def __add__(self, other):
        tmp = self.copy()
        
        # Make sure to not overwrite anything here!
        for key in other:
            if key in tmp:
                raise KeyError('%s supplied more than once!' % key)
                                
            tmp[key] = other[key]    
                
        return ParameterBundle(**tmp)
        
    def __sub__(self, other):
        tmp1 = self.copy()
    
        for key in other:    
            del tmp1[key]
    
        return ParameterBundle(**tmp1)    
    
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
    
    @property
    def Npops(self):
        """ Number of distinct populations represented. """
        return len(self.pop_ids)
    
    @property
    def pop_ids(self):
        """ List of ID numbers -- one for each population."""
        pops = []
        for key in self:
            prefix, idnum = pop_id_num(key)    
            if idnum not in pops:
                pops.append(idnum)
    
        return pops      
    
    def link_sfrd_to(self, num):
        if self.num is not None:
            self['pop_tunnel{%i}' % self.num] = num
        else:
            self['pop_tunnel'] = num
   
    @property    
    def info(self):
        """ Print out info about this bundle. """
        
        header('Bundle Info')
        for key in self.kwargs.keys():
            if key == self.bundle:
                found = True
                print line('*%s*' % self.base)
            else:
                found = False
                print line(key)
            
        if not found:
            print line('*%s*' % self.base)
            
        separator()
        print line('Run \'reinitialize\' with one of the above as argument to change.')
        footer()
    
    @property    
    def orphans(self):
        """
        Return dictionary of parameters that aren't associated with a population.
        """
        tmp = {}
        for par in self:
            prefix, idnum = pop_id_num(par)
            if idnum is None:
                tmp[par] = self[par]
                
        return tmp
    
    def pars_by_pop(self, num, strip_id=False):
        """
        Return dictionary of parameters associated with population `num`.
        """
        tmp = {}
        for par in self:
            prefix, idnum = pop_id_num(par)
            if idnum == num:
                if strip_id:
                    tmp[prefix] = self[par]
                else:    
                    tmp[par] = self[par]
                
        return tmp        


_PB = ParameterBundle
_uv_pop = _PB('pop:fcoll', id_num=0) + _PB('sed:uv', id_num=0)
_xr_pop = _PB('pop:fcoll', id_num=1) + _PB('sed:xray', id_num=1)

_gs_4par = _PB('pop:fcoll', id_num=0) + _PB('sed:lw', id_num=0) \
         + _PB('pop:fcoll', id_num=1) + _PB('sed:lyc', id_num=1) \
         + _PB('pop:fcoll', id_num=2) + _PB('sed:xray', id_num=2)

_tmp = {'gs_2pop': _uv_pop+_xr_pop, 'gs_4par': _gs_4par}

_Bundles['sim'] = _tmp
