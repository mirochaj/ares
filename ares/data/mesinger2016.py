import os
import numpy as np
from ares.util import ParameterBundle as PB
from ares.physics.Constants import E_LL, E_LyA

_input = os.getenv('ARES') + '/input/eos'

def load(model='faint_galaxies'):
    """
    Assumes you downloaded the raw EoS data into $ARES/input/eos, and that
    there are separate directories 'faint_galaxies', and 'bright_galaxies'
    containing all the ps_no_halos* files.
    """
    k = []
    z = []
    ps = []
    QHII = []
    dTb = []
    for fn in os.listdir('%s/%s' % (_input, model)):
        if not fn.startswith('ps_no_halos'):
            continue

        _z = float(fn[13:19])
        _dTb = float(fn[fn.index('aveTb')+5:fn.index('aveTb')+11])
        _QHII = float(fn[fn.index('nf')+2:fn.index('nf')+10])

        z.append(_z)
        dTb.append(_dTb)
        QHII.append(1. - _QHII)

        x, y, err = np.loadtxt('%s/%s/%s' % (_input,model,fn), unpack=True)

        k.append(x)
        ps.append(y)

    z = np.array(z)

    s = np.argsort(z)

    dTb = np.array(dTb)[s]
    k = np.array(k)[s]
    ps = np.array(ps)[s]
    QHII = np.array(QHII)[s]

    return {'z': z[s], 'k': k[0], 'ps_21_dl': ps, 'dTb': dTb, 'Qi': QHII}


_base = {}
_base = PB('gs:4par')
_base['pop_Tmin{0}'] = 2e4
_base['pop_fstar{0}'] = 0.1

# Ly-a
_base['pop_solve_rte{0}'] = (E_LyA, E_LL)
_base['pop_sed_model{0}'] = True
_base['pop_sed{0}'] = 'pl'
_base['pop_alpha{0}'] = 0
_base['pop_Emin{0}'] = E_LyA
_base['pop_Emax{0}'] = E_LL
_base['pop_EminNorm{0}'] = E_LyA
_base['pop_EmaxNorm{0}'] = E_LL
_base['pop_rad_yield{0}'] = 9690.
_base['pop_rad_yield_units{0}'] = 'photons/baryon'
_base['pop_lw_src{0}'] = True

# X-ray
_base['pop_solve_rte{1}'] = True
_base['pop_sed_model{1}'] = True
_base['tau_redshift_bins'] = 1000
_base['pop_sed{1}'] = 'pl'
_base['pop_alpha{1}'] = -1.5
_base['pop_Emin{1}'] = 300.
_base['pop_Emax{1}'] = 3e4
_base['pop_EminNorm{1}'] = 500
_base['pop_EmaxNorm{1}'] = 8e3
_base['pop_rad_yield{1}'] = 2.6e39 * 0.5
_base['pop_src_ion_igm{1}'] = True

# LyC
_base['pop_sed_model{2}'] = True
_base['pop_fesc{2}'] = 0.1
_base['pop_rad_yield{2}'] = 2e3
_base['pop_rad_yield_units{2}'] = 'photons/baryon'
_base['pop_Emin{2}'] = E_LL*1.01
_base['pop_Emax{2}'] = 24.6

# General
_base['tau_approx'] = 'neutral'
_base['include_He'] = True
_base['approx_He'] = True
_base['secondary_ionization'] = 3
_base['approx_Salpha'] = 3
_base['clumping_factor'] = 0.
_base['photon_counting'] = True
_base['problem_type'] = 101.3

faint_galaxies = _base.copy()
bright_galaxies = _base.copy()

bright_galaxies['pop_Tmin{0}'] = 2e5
bright_galaxies['pop_fstar{0}'] = 0.1
bright_galaxies['pop_fesc{2}'] = 1
bright_galaxies['pop_rad_yield{2}'] = 2e3
