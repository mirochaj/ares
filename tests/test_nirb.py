import ares
import numpy as np
import matplotlib.pyplot as pl
from ares.physics.Constants import c, h_p, erg_per_ev, rhodot_cgs, E_LL, E_LyA

# Vanilla PopII
popII  = ares.util.ParameterBundle('sun2020:base_nolya').pars_by_pop(0, 1)

# Vanilla PopII + simple nebular emission
popIIa = popII.copy()
popIIa['pop_nebular'] = 2
popIIa['pop_nebular_continuum'] = True
popIIa['pop_nebular_lines'] = True

popIIa['pop_Emin'] = 0.4
popIIa['pop_Emax'] = 400.
popIIa['pop_solve_rte'] = (0.4, 400.)

# Add in PopIII
popIIb = popIIa.copy()
popIIb.num = 0
popIIIa = popIIb + ares.util.ParameterBundle('sun2020:med')

popIIIa['feedback_LW'] = True
popIIIa['feedback_LW_maxiter'] = 15 # just for testing
popIIIa['feedback_LW_mean_err'] = True # don't wait until max error is small
popIIIa['feedback_LW_sfrd_rtol'] = 1e-2
popIIIa['feedback_LW_tol_zrange'] = (5.5, 40)

# Source model
popIIIa['pop_sfr{1}'] = 1e-4
popIIIa['pop_time_limit{1}'] = 250
popIIIa['pop_bind_limit{1}'] = None
# Turn off Pop III LW feedback
#popIII['pop_fesc_LW{1}'] = 0.

popIIIa['pop_zdead{0}'] = 5
popIIIa['pop_zdead{1}'] = 5

# Additions for PopIII synthesis
popIIIa['pop_sed{1}'] = 'sps-toy'
popIIIa['pop_toysps_method{1}'] = 'schaerer2002'
popIIIa['pop_ssp{1}'] = False
popIIIa['pop_model{1}'] = 'tavg_nms'
popIIIa['pop_mass{1}'] = 5.

# Set energy range by hand. This is picky! Be careful that Emax <= 13.6 eV
# (long story -- will work to fix in future)
popIIIa['pop_Emin{1}'] = 0.4
popIIIa['pop_Emax{1}'] = 400.
popIIIa['pop_dE{1}'] = 0.1
popIIIa['pop_solve_rte{1}'] = (0.4, 400.)

popIIIa['pop_nebular{1}'] = 2
popIIIa['pop_nebular_continuum{1}'] = True
popIIIa['pop_nebular_lines{1}'] = True

sim1 = ares.simulations.MetaGalacticBackground(**popIIIa)
sim1.run()

#################################################
zgeq = 5.
band12 = (1, 2)
band_e = np.arange(0.75, 5.25, 0.25)
k = np.logspace(-2, 2, 20)
arcsec = np.logspace(0, 4, 50) # arcsec
ell = np.logspace(1, 4, 50) # spherical harmonic l "ell"
band_averaged = True
#################################################

# Find band centers and construct pairs of band edges
bands = [tuple(band_e[i:i+2]) for i in range(band_e.size-1)]
bands_c = np.array([np.mean(band_e[i:i+2]) for i in range(band_e.size-1)])

## SPHEREx bands
spherex_e = np.array([0.75, 0.85, 0.96, 1.11, 1.35, 1.75, 2.33, 3.08, 4.0, 5.0])
spherex_bands = [tuple(spherex_e[i:i+2]) for i in range(spherex_e.size-1)]
spherex_c = [np.mean(spherex_e[i:i+2]) for i in range(spherex_e.size-1)]

print 'calculating Pop 2 PS...'
ps_v_l2 = sim1.pops[0].get_ps_obs(ell, wave_obs=band12, scale_units='ell', time_res=20, include_shot=1)

print 'calculating Pop 3 PS...'
ps_v_l3 = sim1.pops[1].get_ps_obs(ell, wave_obs=band12, scale_units='ell', time_res=20, include_shot=1)