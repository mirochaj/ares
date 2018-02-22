"""

ParameterBundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Fri Jun 10 11:00:05 PDT 2016

Description: 

"""

import numpy as np
from .ReadData import read_lit
from .ProblemTypes import ProblemType
from .ParameterFile import pop_id_num, par_info
from .PrintInfo import header, footer, separator, line

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

def _add_pop_tag(par, num):
    """
    Add a population ID tag to each parameter.
    """
    
    prefix, idnum = pop_id_num(par)
    
    if idnum is not None:
        return '{0!s}{{{1}}}'.format(prefix, num)
    else:
        return '{0!s}{{{1}}}'.format(par, num)

def _add_pq_tag(par, num):
    """
    Add a population ID tag to each parameter.
    """

    prefix, idnum = pop_id_num(par)

    if idnum is not None:
        return '{0!s}[{1}]'.format(prefix, num)
    else:
        return '{0!s}[{1}]'.format(par, num)
        


class PreBundle(dict):
    
    @property
    def typical_free_parameters(self):
        return self._typical_free_parameters
        
    @typical_free_parameters.setter
    def typical_free_parameters(self, value):
        if type(value) in [list, tuple]:
            self._typical_free_parameters = value
        elif type(value) == dict:
            self._typical_free_parameters = []
            for key in value:
                self._typical_free_parameters.append(key)
                self._common_names = {key: value}

    @property
    def common_names(self):
        return self._common_names
        
    @common_names.setter
    def common_names(self, value):
        if not hasattr(self, '_common_names'):
            self._common_names = {}
            
        assert type(value) is dict
            
        for key in value:
            if key in self._common_names:
                print(("Overwriting common name for {0!s}: {1!s}->" +\
                    "{2!s}").format(key, self._common_names[key], value[key]))
            
            self._common_names[key] = value[key]

        
_pop_fcoll = \
{
 'pop_sfr_model': 'fcoll',
 'pop_Tmin': 1e4,
 'pop_Tmax': None,
}

_pop_user_sfrd = \
{

 'pop_sfr_model': 'sfrd-func',
 'pop_sfrd': 'pq[0]',
 'pq_func[0]': 'dpl',
 'pq_func_var[0]': 'z',
 'pq_func_par0[0]': 1e-6,
 'pq_func_par1[0]': 15.,
 'pq_func_par2[0]': -5.,
 'pq_func_par3[0]': -8.,
 
}

_src_lya = \
{
 'pop_Nlw': 1e4,
 'pop_lw_src': False,
 'pop_lya_src': True,
 'pop_heat_src_igm': False,
 'pop_ion_src_cgm': False,
 'pop_ion_src_igm': False,
 'pop_sed_model': False,
}

_src_ion = \
{
 'pop_sfr_model': 'fcoll',
 'pop_Nion': 4000.,
 'pop_fesc': 0.1,
 'pop_lw_src': False,
 'pop_lya_src': False,
 'pop_heat_src_igm': False,
 'pop_ion_src_cgm': True,
 'pop_ion_src_igm': False,
 'pop_sed_model': False,
}

_src_xray = \
{
 'pop_rad_yield': 2.6e39,
 'pop_rad_yield_units': 'erg/s/sfr',
 'pop_Emin': 2e2, 
 'pop_Emax': 5e4,
 'pop_EminNorm': 5e2, 
 'pop_EmaxNorm': 8e3,
 'pop_sed': 'pl',
 'pop_alpha': -1.5,
 'pop_lw_src': False,
 'pop_lya_src': False,
 'pop_heat_src_igm': True,
 'pop_ion_src_cgm': False,
 'pop_ion_src_igm': False,
 'pop_sed_model': True,
 'pop_fXh': 0.2,
}


_sed_toy = \
{
 'pop_sed_model': False,
 'pop_Nion': 4e3,
 'pop_Nlw': 9690,
 'pop_rad_yield': 2.6e39,
 'pop_fesc': 0.1,
 'pop_lya_src': True,
 'pop_lw_src': False,
 'pop_ion_src_cgm': True,
 'pop_ion_src_igm': False,
 'pop_heat_src_igm': True,
 
}

_sed_xi = \
{
 'pop_sed_model': False,
 'pop_xi_LW': 40.,
 'pop_xi_UV': 969.,
 'pop_xi_XR': 0.1,
}

_pop_sfe = \
{
 'pop_sfr_model': 'sfe-func',
 'pop_sed': 'eldridge2009',
 'pop_Z': 0.02,
 'pop_fstar': 'pq',
 'pq_func': 'dpl',
 'pq_func_var': 'Mh',
 'pq_func_par0': 0.05,
 'pq_func_par1': 3e11,
 'pq_func_par2': 0.6,
 'pq_func_par3': -0.6,

 # Redshift dependent parameters here
}

_pop_sfe_ext = \
{
 'pq_faux': 'plexp',
 'pq_faux_var': 'mass',
 'pq_faux_meth': 'add',
 'pq_faux_par0': 0.005,
 'pq_faux_par1': 1e9,
 'pq_faux_par2': 0.01,
 'pq_faux_par3': 1e10,
}

_pop_mlf = \
{
 'pop_sfr_model': 'mlf',
 'pop_fstar': None,
 'pop_mlf': 'pq',
 'pop_MAR': 'hmf',
 
 'pq_func': 'dpl',
 'pq_func_par0': 0.1,
 'pq_func_par1': 1e12,
 'pq_func_par2': 0.67,
 'pq_func_par3': 0.5,
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
 "pop_rad_yield": 4e3, 
 "pop_rad_yield_units": 'photons/baryon',
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
 'pop_rad_yield': 'from_sed',
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
 "pop_rad_yield": 2.6e39, 
 "pop_rad_yield_units": 'erg/s/SFR',
}

_crte_xrb = \
{
 "pop_solve_rte": True, 
 "tau_redshift_bins": 400,
 "tau_approx": 'neutral',
}

_crte_lwb = _crte_xrb.copy()
_crte_lwb['pop_solve_rte'] = (10.2, 13.6)
_crte_lwb['tau_approx'] = True

# Some different spectral models
_uvsed_toy = dict(pop_rad_yield=4000, pop_rad_yield_units='photons/b',
    pop_Emin=10.2, pop_Emax=24.6, pop_EminNorm=13.6, pop_EmaxNorm=24.6)
_uvsed_bpass = dict(pop_sed='eldridge2009', pop_binaries=False, pop_Z=0.02,
    pop_Emin=10.2, pop_Emax=24.6, pop_EminNorm=13.6, pop_EmaxNorm=24.6)
_uvsed_s99 = _uvsed_bpass.copy()
_uvsed_s99['pop_sed'] = 'leitherer1999'

_mcd = _sed_xr.copy()
_mcd['pop_sed'] = 'mcd'
_pl = _mcd.copy()
_pl['pop_sed'] = 'pl'

_simple_dc1 = {'dustcorr_method': 'meurer1999', 'dustcorr_beta': -2.}
_simple_dc2 = {'dustcorr_method': 'meurer1999', 'dustcorr_beta': 'bouwens2014'}
_evolve_dc = \
{
'dustcorr_method': ['meurer1999', 'pettini1998', 'capak2015'],
'dustcorr_beta': 'bouwens2014',
'dustcorr_ztrans': [0, 4, 5],
}

_Bundles = \
{
 'pop': {'fcoll': _pop_fcoll, 'sfe-dpl': _pop_sfe, 'sfe-func': _pop_sfe, 
    'sfrd-func': _pop_user_sfrd, 'sfe-pl-ext': _pop_sfe_ext},
 'sed': {'uv': _sed_uv, 'lw': _sed_lw, 'lyc': _sed_lyc, 
         'xray':_sed_xr, 'pl': _pl, 'mcd': _mcd, 'toy': _sed_toy,
         'bpass': _uvsed_bpass, 's99': _uvsed_s99, 'xi': _sed_xi},
 'src': {'toy-lya': _src_lya, 'toy-xray': _src_xray, 'toy-ion': _src_ion},
 'physics': {'xrb': _crte_xrb, 'lwb': _crte_lwb},
 'dust': {'simple': _simple_dc1, 'var_beta': _simple_dc2, 
    'evolving': _evolve_dc, 'none': {},
    }
}

class ParameterBundle(dict):
    def __init__(self, bundle=None, id_num=None, bset=None, **kwargs):
        self.bundle = bundle
        self.kwargs = kwargs
        
        if bset is None:
            self.bset = _Bundles
        else:
            self.bset = bset
        
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
        
        if pre in self.bset.keys():
            kw = self.bset[pre][post]
        elif pre == 'prob':
            kw = ProblemType(float(post))
        else:
            mod = read_lit(pre) 
            kw = mod.__dict__[post]
            
            # Save where we found it for future reference / sanity checking.
            if hasattr(mod, 'path'):
                self.path = mod.path
        
        pars = kw.keys()

        for key in pars:
            self[key] = kw[key]    

    def __getattr__(self, name):
        if name not in self.keys():
            pass
        try:
            return self[name]
        except KeyError as e:
            # this is needed for hasattr to work as expected in python 3!
            raise AttributeError('{!s}'.format(e.args))

    def __add__(self, other):
        tmp = self.copy()

        # If any keys overlap, overwrite first instance with second.
        # Just alert the user that this is happening.
        for key in other:
            if key in tmp and rank == 0:
                msg1 = "UPDATE: Setting {0} -> {1}".format(key.ljust(20), 
                    str(other[key]).ljust(12))
                msg2 = "previously {0} = {1}".format(str(key).ljust(20), tmp[key])
                print('{0} [{1}]'.format(msg1, msg2))

            tmp[key] = other[key]

        return ParameterBundle(**tmp)

    def __sub__(self, other):
        tmp1 = self.copy()
    
        for key in other:    
            del tmp1[key]
    
        return ParameterBundle(**tmp1)
        
    def copy(self):
        return ParameterBundle(**self)
    
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
                
    def tag_pq_id(self, par, num):
        """
        Find ParameterizedQuantity parameters and tag with `num`.
        """
                
        if self[par] == 'pq':
            current_tag = None
        else:
            m = re.search(r"\[([0-9])\]", par)

            assert m is not None, "No ID found for par={!s}".format(par)
            
            current_tag = int(m.group(1))
        
        # Figure out what all the parameters are currently
        pars = self.pars_by_pq(current_tag, strip_id=False)
        
        # Delete 'em, rename 'em
        for key in pars:
            del self[key]
        
        self[par] = _add_pq_tag('pq', num)
        for key in pars:
            self[_add_pq_tag(key, num)] = pars[key]
            
    @property
    def pqid(self):
        if not hasattr(self, '_pqid'):
            self._pqid = None
        return self._pqid
    
    @pqid.setter
    def pqid(self, value):
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
            self['pop_tunnel{{{}}}'.format(self.num)] = num
        else:
            self['pop_tunnel'] = num

    @property    
    def info(self):
        """ Print out info about this bundle. """

        header('Bundle Info')
        for key in self.kwargs.keys():
            if key == self.bundle:
                found = True
                print(line('*{!s}*'.format(self.base)))
            else:
                found = False
                print(line(key))
            
        if not found:
            print(line('*{!s}*'.format(self.base)))
        
        separator()
        print(line('Run \'reinitialize\' with one of the above as argument to change.'))
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
        
        This will take any parameters with ID numbers, and any parameters
        with the `hmf_` prefix, since populations need to know about that 
        stuff. Also, dustcorr parameters.
        """
        tmp = {}
        for par in self:
            prefix, idnum = pop_id_num(par)
            if (idnum == num) or prefix.startswith('hmf_') \
                or prefix.startswith('dustcorr') or prefix.startswith('sam_'):
                if strip_id:
                    tmp[prefix] = self[par]
                else:    
                    tmp[par] = self[par]
                
        return ParameterBundle(**tmp)
        
    def get_base_kwargs(self):
        tmp = {}
        for par in self:
            prefix, idnum = pop_id_num(par)
            
            if idnum is None: 
                tmp[par] = self[par]
                
        return ParameterBundle(**tmp)
        
    @property
    def pqs(self):
        if not hasattr(self, '_pqs'):
            self.pqids
            
        return self._pqs
        
    @property
    def pqids(self):
        if not hasattr(self, '_pqids'):
            pqs = []
            pqids = []
            for par in self:                
                if self[par] == 'pq':
                    pqid = 'None'
                else:
                    prefix, popid, pqid = par_info(par)
                
                if pqid is None:
                    continue

                if pqid in pqids:
                    continue

                pqs.append(par)
                
                if pqid is 'None':
                    pqids.append(None)
                else:
                    pqids.append(pqid)            
            
            self._pqs = pqs
            self._pqids = pqids
            
        return self._pqids    
        
    def pars_by_pq(self, num=None, strip_id=False):
        
        if num == None and (self.pqids == [None]):
            untagged = True
        else:
            untagged = False
        
        i = self.pqids.index(num)
        tmp = {self.pqs[i]: self[self.pqs[i]]}
        for par in self:   
            
            if untagged and (par[0:2] == 'pq'):
                tmp[par] = self[par]
                continue
                     
            prefix, popid, pqid = par_info(par)
            
            if pqid is None:
                continue
            
            if pqid == num:
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
         
         
# Build a template four-parameter model
_lw = _PB('pop:fcoll') + _PB('src:toy-lya')
_lw.num = 0
_xr = _PB('src:toy-xray')
_xr.num = 1
_xr.link_sfrd_to = 0
_uv = _PB('src:toy-ion')
_uv.num = 2
_uv.link_sfrd_to = 0
         
_gs_4par = _lw + _xr + _uv         

_tanh_sim = {'problem_type': 100, 'tanh_model': True,
    'output_frequencies': np.arange(30., 201.)}

_param_sim = {'problem_type': 100, 'parametric_model': True,
    'output_frequencies': np.arange(30., 201.)}

_tmp = {'2pop': _uv_pop+_xr_pop, '4par': _gs_4par,
    'tanh': _tanh_sim, 'param': _param_sim}

_Bundles['gs'] = _tmp
