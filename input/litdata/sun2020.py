from mirocha2017 import base as _base
from mirocha2018 import low as _low, med as _med, high as _high, bb as _bb
from mirocha2020 import _halo_updates
from ares.util import ParameterBundle as PB
from ares.physics.Constants import E_LyA, E_LL, lam_LyA

_base = PB(**_base).pars_by_pop(0, 1)
#_base.update(_halo_updates)

_nirb_updates = {}
_nirb_updates['pop_Emin'] = 0.41 # as low as BPASS goes
#_nirb_updates['pop_Emax'] = E_LL
_nirb_updates['pop_zdead'] = 5
_nirb_updates['final_redshift'] = 5
_nirb_updates['pop_solve_rte'] = (0.41, E_LL) # This kills heating!
_nirb_updates['pop_fesc'] = 0.1
_nirb_updates['tau_redshift_bins'] = 1000 # probably overkill
_nirb_updates['tau_approx'] = False
_nirb_updates['tau_clumpy'] = 'madau1995'

_base.update(_nirb_updates)
_base.num = 0
_base['pop_zdead{0}'] = 5.
_base['pop_nebular{0}'] = 2
_base['pop_nebular_continuum{0}'] = True
_base['pop_nebular_lines{0}'] = True
_base['pop_nebular_caseBdeparture{0}'] = 1.

base = _base

low = PB(**_low).pars_by_pop(2, 1)
low.num = 1
med = PB(**_med).pars_by_pop(2, 1)
med.num = 1
high = PB(**_high).pars_by_pop(2, 1)
high.num = 1
bb = PB(**_bb).pars_by_pop(2, 1)
bb.num = 1

low['pop_nebular{1}'] = 2
low['pop_nebular_continuum{1}'] = True
low['pop_nebular_lines{1}'] = True

_popIII_updates = {'sam_dz': None, 'feedback_LW_sfrd_popid': 1}
low.update(_popIII_updates)
for pbund in [med, high, bb]:
    pbund['pop_sed{1}'] = 'sps-toy'
    pbund['pop_toysps_method{1}'] = 'schaerer2002'
    pbund['pop_ssp{1}'] = False
    pbund['pop_model{1}'] = 'tavg_nms'
    pbund['pop_zdead{1}'] = 5.
    pbund['pop_nebular{1}'] = 2
    pbund['pop_nebular_continuum{1}'] = True
    pbund['pop_nebular_lines{1}'] = True
    # Set energy range by hand. This is picky! Be careful that Emax <= 13.6 eV
    # (long story -- will work to fix in future)

    pbund.update(_popIII_updates)

bb['pop_toysps_method{1}'] = 'bb'
bb['pop_nebular{1}'] = 2
bb['pop_nebular_continuum{1}'] = True
bb['pop_nebular_lines{1}'] = True
bb['pop_nebular_caseBdeparture{1}'] = 2.

bb['pop_mass{1}'] = 100.      # This is redundant with pop_sfr
bb['pop_lifetime{1}'] = 1e7   # This is redundant with pop_sfr
bb['pop_fesc{1}'] = 0.1
bb['pop_Emin{1}'] = 0.41
bb['pop_Emax{1}'] = 2e2
bb['pop_EminNorm{1}'] = 13.6
bb['pop_EmaxNorm{1}'] = 24.6
bb['pop_qdot{1}'] = 1e50
bb['pop_dlam{1}'] = 1.
bb['pop_lmin{1}'] = 100.
bb['pop_lmax{1}'] = 1e4
bb['pop_solve_rte{1}'] = (0.4, E_LL)
