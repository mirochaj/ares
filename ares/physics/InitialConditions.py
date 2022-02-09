"""

InitialConditions.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Mon 23 Mar 2020 09:15:53 EDT

Description:

"""

from __future__ import print_function

import os
import re
import numpy as np
from ..data import ARES

try:
    import camb
    have_camb = True
except ImportError:
    have_camb = False

_pars_CosmoRec = ['cosmorec_nz', 'cosmorec_z0', 'cosmorec_zf',
    'helium_by_mass', 'cmb_temp_0', 'omega_m_0', 'omega_b_0', 'omega_l_0',
    'omega_k_0', 'hubble_0', 'relativistic_species', 'cosmorec_recfast_fudge',
    'cosmorec_nshells_H', 'cosmorec_nS', 'cosmorec_dm_annhil', 'cosmorec_A2s1s',
    'cosmorec_nshells_He', 'cosmorec_HI_abs', 'cosmorec_spin_forb',
    'cosmorec_feedback_He', 'cosmorec_run_pde', 'cosmorec_corr_2s1s',
    'cosmorec_2phot', 'cosmorec_raman', 'cosmorec_output', 'cosmorec_fmt']

class InitialConditions(object):
    """
    This should be inherited by Cosmology.
    """
    def __init__(self, pf):
        self.pf = pf

    @property
    def prefix(self):
        return self._prefix

    @prefix.setter
    def prefix(self, value):
        self._prefix = value

    def get_inits_rec(self):
        """
        Get recombination history from file or directly from CosmoRec.
        """
        fn = '{}/input/inits/inits_{}.txt'.format(ARES, self.prefix)

        # Look for table first, then run if we don't find it.
        if os.path.exists(fn):
            z, xe, Tk = np.loadtxt(fn, unpack=True)
            if self.pf['verbose']:
                name = fn
                print("# Loaded {}.".format(fn.replace(ARES, '$ARES')))
            return {'z': z, 'xe': xe, 'Tk': Tk}
        else:
            if self.pf['verbose']:
                print("# Did not find initial conditions in file {}.".format(fn))

            assert os.path.exists('{}/CosmoRec'.format(self.pf['cosmorec_path'])), \
                "No CosmoRec executable found. Set via cosmorec_path parameter."

            if self.pf['verbose']:
                print("# Will generate from scratch using {}/CosmoRec.".format(
                    self.pf['cosmorec_path']))

            return self._run_CosmoRec()

    def get_inits_ps(self):
        """
        Get matter power spectrum from file or directly from CAMB or CLASS.

        .. note :: Must supply this to HMF, if not None. Perhaps add a parameter
            that tells ARES to use CAMB or CLASS. How about, e.g., setting
            "use_boltzmann='camb'" or something?
        """

        pass

    def _run_CosmoRec(self, save=True): # pragma: no cover
        """
        Run CosmoRec. Assumes we've got an executable waiting for us in
        directory supplied via ``cosmorec_path`` parameter in ARES.

        Will save to $ARES/input/inits. Can check in $ARES/input/inits/outputs
        for CosmoRec parameter files should any debugging be necessary. They
        will have the same naming convention, just different filename prefix
        ("cosmorec" instead of "inits").
        """
        # Some defaults copied over from CosmoRec.
        CR_pars = [self.pf[par] for par in _pars_CosmoRec]

        # Correct output dir. Just add provided path on top of $ARES
        CR_pars[-2] = '{}/{}/'.format(ARES, CR_pars[-2])

        fn_pars = 'cosmorec_{}.dat'.format(self.prefix)

        # Create parameter file for reference
        to_outputs = CR_pars[-2]

        if not os.path.exists(to_outputs):
            os.mkdir(to_outputs)

        with open(to_outputs + '/' + fn_pars, 'w') as f:
            for element in CR_pars:
                print(element, file=f)

        # Run the thing
        str_to_exec = '{}/CosmoRec {}/{} >> cr.log'.format(
            self.pf['cosmorec_path'], to_outputs, fn_pars)
        os.system(str_to_exec)

        for fn in os.listdir(to_outputs):
            if re.search('trans', fn):
                break

        # Convert it to ares format
        data = np.loadtxt('{}/{}'.format(to_outputs, fn))

        new_data = \
        {
         'z': data[:,0][-1::-1],
         'xe': data[:,1][-1::-1],
         'Tk': data[:,2][-1::-1],
        }

        fn_out = '{}/input/inits/inits_{}.txt'.format(ARES,
            self.prefix)

        np.savetxt(fn_out, data[-1::-1,0:3], header='z; xe; Te')

        if self.pf['verbose']:
            print("# Wrote {}.".format(fn_out))

        return new_data
