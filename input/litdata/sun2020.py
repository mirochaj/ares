from mirocha2017 import base as _base
from mirocha2018 import low as _low, med as _med, high as _high
from ares.util import ParameterBundle as PB
from ares.physics.Constants import E_LyA, E_LL

_base = PB(**_base).pars_by_pop(0, 1)

_nirb_updates = {}
_nirb_updates['pop_Emin'] = 0.42 # as low as BPASS goes
_nirb_updates['pop_Emax'] = E_LL
_nirb_updates['pop_solve_rte'] = (0.42, 13.61)
_nirb_updates['pop_fesc'] = 0.1
_nirb_updates['tau_redshift_bins'] = 1000 # probably overkill
_nirb_updates['tau_approx'] = False
_nirb_updates['tau_clumpy'] = 'madau1995'

_base.update(_nirb_updates)

_generic_lya = \
{
 'pop_sfr_model': 'link:sfrd:0',
 'pop_fesc': 'pop_fesc',  # Make sure this pop has same fesc as PopII
 # THIS IS NEW! Makes sure we take emission from PopII stellar emission.
 'pop_rad_yield': 'link:src.rad_yield:0:13.6-400',    
    
 'pop_reproc': True,  # This will get replaced by `add_lya` below.
 'pop_frep': 0.6667,  # This will get replaced by `add_lya` below.
 'pop_fesc': 0.0,     # This will get replaced by `add_lya` below.
 
 'pop_sed': 'delta',
 'pop_Emin': 0.1,
 'pop_Emax': E_LyA,
 'pop_EminNorm': 9.9,
 'pop_EmaxNorm': 10.5,
 
 # Solution method
 "lya_nmax": 8,
 'pop_solve_rte': True,
 
 # Help out the integrator by telling it this is a sharply peaked function!
 'pop_sed_sharp_at': E_LyA,
}

def add_lya(pop1):
            
    if pop1.num is None:
        pop1.num = 0
        
    pop2 = PB(**_generic_lya)
    pop2.num = pop1.num + 1
    
    pop2['pop_sfr_model{{{}}}'.format(pop2.num)] = \
        'link:sfrd:{}'.format(pop1.num)
    pop2['pop_rad_yield{{{}}}'.format(pop2.num)] = \
        'link:src.rad_yield:{}:13.6-400'.format(pop1.num)
    pop2['pop_fesc{{{}}}'.format(pop2.num)] = \
        'pop_fesc{{{}}}'.format(pop1.num)
    
    pars = pop1 + pop2
    
    return pars

base = add_lya(_base)

_low_st = PB(**_low).pars_by_pop(2, 1)
_low_st.num = 2
_med_st = PB(**_med).pars_by_pop(2, 1)
_med_st.num = 2
_high_st = PB(**_high).pars_by_pop(2, 1)
_high_st.num = 2

low = add_lya(_low_st)
med = add_lya(_med_st)
high = add_lya(_high_st)
low['sam_dz'] = None
med['sam_dz'] = None
high['sam_dz'] = None




