from mirocha2017 import base as _base
from mirocha2018 import low, med, high
from ares.physics.Constants import E_LyA
from ares.util import ParameterBundle as PB

base = PB(**_base).pars_by_pop(0, 1)

_nirb_updates = {}
_nirb_updates['pop_Emin'] = 0.42 # as low as BPASS goes
_nirb_updates['pop_Emax'] = 25.
_nirb_updates['pop_solve_rte'] = (0.42, 13.61)
_nirb_updates['pop_fesc'] = 0.1
_nirb_updates['tau_redshift_bins'] = 1000 # probably overkill

base.update(_nirb_updates)

_generic_lya = \
{
 'pop_reproc': True,  # Just means use (1 - fesc) rather than fesc
 'pop_frep': 0.6667,  # In this case, frep = flya
 
 'pop_sed': 'delta',
 'pop_Emin': 0.1,
 'pop_Emax': E_LyA,
 'pop_EminNorm': 9.9,
 'pop_EmaxNorm': 10.5,
 
 # Solution method
 "lya_nmax": 8,
 'pop_solve_rte': True,
 'tau_approx': True,
 
 # Help out the integrator by telling it this is a sharply peaked function!
 'pop_sed_sharp_at': E_LyA,
}

def add_lya(pop_pars):
    pass

new_pop_lya1 = \
{
    'pop_sfr_model{2}': 'link:sfrd:0',
    'pop_fesc': 'pop_fesc{0}',  # Make sure this pop has same fesc as PopII
    # THIS IS NEW! Makes sure we take emission from PopII stellar emission.
    'pop_rad_yield{2}': 'link:src.rad_yield:0:13.6-400', #'link:yield_per_sfr:0',

    
}