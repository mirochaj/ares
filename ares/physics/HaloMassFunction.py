"""
Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description:

"""

import os
import re
import sys
import glob
import pickle
import numpy as np
from . import Cosmology
from ..data import ARES
from types import FunctionType
from ..util import ParameterFile
from scipy.misc import derivative
from scipy.optimize import fsolve
from ..util.Warnings import no_hmf
from scipy.integrate import cumtrapz, simps
from ..util.PrintInfo import print_hmf
from ..util.ProgressBar import ProgressBar
from ..util.ParameterFile import ParameterFile
from ..util.Math import central_difference, smooth
from ..util.Pickling import read_pickle_file, write_pickle_file
from ..util.SetDefaultParameterValues import CosmologyParameters
from .Constants import g_per_msun, cm_per_mpc, s_per_yr, G, cm_per_kpc, \
    m_H, k_B, s_per_myr
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, \
    interp1d, InterpolatedUnivariateSpline


try:
    from scipy.special import erfc
except ImportError:
    pass

try:
    import h5py
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

try:
    import hmf
    from hmf import MassFunction
    have_hmf = True
    hmf_vstr = hmf.__version__
    hmf_vers = float(hmf_vstr[0:hmf_vstr.index('.')+2])
except ImportError:
    have_hmf = False
    hmf_vers = 0

if have_hmf:
	if 0 <= hmf_vers <= 3.4:
		try:
		    MassFunctionWDM = hmf.wdm.MassFunctionWDM
		except ImportError:
		    pass

# Old versions of HMF
try:
    import camb
    have_pycamb = True
except ImportError:
    have_pycamb = False

    try:
        import pycamb
        have_pycamb = True
        if int(hmf.__version__.split('.')[0]) >= 3:
            print("For HMF v3 or greater, must use new 'camb' Python package.")
    except ImportError:
        have_pycamb = False

sqrt2 = np.sqrt(2.)

tiny_dndm = 1e-30
tiny_fcoll = 1e-18
tiny_dfcolldz = 1e-18

class HaloMassFunction(object):
    def __init__(self, **kwargs):
        """
        Initialize HaloDensity object.

        If an input table is supplied, set up interpolation tables over
        mass and redshift for the collapsed fraction and its derivative.
        If no table is supplied, create one using Steven Murray's halo
        mass function calculator.

        =================================
        The following kwargs are relevant
        =================================
        logMmin : float
            Minimum log-Mass value over which to tabulate mass function.
        logMmax : float
            Maximum log-Mass value over which to tabulate mass function.
        dlogM : float
            log-Mass resolution of mass function table.
        zmin : float
            Minimum redshift in mass function table.
        zmax : float
            Maximum redshift in mass function table.
        dz : float
            Redshift resolution in mass function table.
        hmf_model : str
            Halo mass function fitting function. Options are:
                PS
                ST
                Warren
                Jenkins
                Reed03
                Reed07
                Angulo
                Angulo_Bound
                Tinker
                Watson_FoF
                Watson
                Crocce
                Courtin
        hmf_table : str
            HDF5 or binary file containing fcoll table.
        hmf_analytic : bool
            If hmf_func == 'PS', will compute fcoll analytically.
            Used as a check of numerical calculation.

        Table Format
        ------------

        """
        self.pf = ParameterFile(**kwargs)

        # Read in a few parameters for convenience
        self.tab_name = self.pf["hmf_table"]
        self.hmf_func = self.pf['hmf_model']
        self.hmf_analytic = self.pf['hmf_analytic']

        # Verify that Tmax is set correctly
        #if self.pf['pop_Tmax'] is not None:
        #    if self.pf['pop_Tmin'] is not None and self.pf['pop_Mmin'] is None:
        #        assert self.pf['pop_Tmax'] > self.pf['pop_Tmin'], \
        #            "Tmax must exceed Tmin!"

        if self.pf['hmf_path'] is not None:
            _path = self.pf['hmf_path']
        else:
            _path = '{0!s}/input/hmf'.format(ARES)

        # Look for tables in input directory

        if ARES is not None and self.pf['hmf_load'] and (self.tab_name is None):
            prefix = self.tab_prefix_hmf(True)
            fn = '{0!s}/{1!s}'.format(_path, prefix)

            # First, look for a perfect match
            if os.path.exists('{0!s}.{1!s}'.format(fn,\
                self.pf['preferred_format'])):
                self.tab_name = '{0!s}.{1!s}'.format(fn, self.pf['preferred_format'])
            # Next, look for same table different format
            elif os.path.exists('{!s}.hdf5'.format(fn)):
                self.tab_name = '{!s}.hdf5'.format(fn)
            else:
                # Leave resolution blank, but enforce ranges
                prefix = self.tab_prefix_hmf()
                candidates =\
                    glob.glob('{0!s}/input/hmf/{1!s}*'.format(ARES, prefix))
                print(candidates)

                if len(candidates) == 1:
                    self.tab_name = candidates[0]
                else:

                    # What parameter file says we need.
                    logMmax = self.pf['hmf_logMmax']
                    logMmin = self.pf['hmf_logMmin']
                    logMsize = (logMmax - logMmin) / self.pf['hmf_dlogM']
                    # Get an extra bin so any derivatives will still be
                    # sane at the boundary.
                    zmax = self.pf['hmf_zmax']
                    zmin = self.pf['hmf_zmin']
                    zsize = (zmax - zmin) / self.pf['hmf_dz'] + 1

                    self.tab_name = None
                    for candidate in candidates:

                        if 'hist' in candidate:
                            continue

                        results = list(map(int, re.findall(r'\d+', candidate)))

                        if self.hmf_func == 'Tinker10':
                            ist = 1
                        else:
                            ist = 0

                        if 'hdf5' in candidate:
                            ien = -1
                        else:
                            ien = None

                        _Nm, _logMmin, _logMmax, _Nz, _zmin, _zmax = \
                            results[ist:ien]

                        if (_logMmin > logMmin) or (_logMmax < logMmax):
                            continue

                        if (_zmin > zmin) or (_zmax < zmax):
                            continue

                        self.tab_name = candidate


        # Override switch: compute Press-Schechter function analytically
        if self.hmf_func == 'PS' and self.hmf_analytic:
            self.tab_name = None

        # Either create table from scratch or load one if we found a match
        if self.tab_name is None:
            if have_hmf and have_pycamb:
                pass
            else:
                no_hmf(self)
                sys.exit()
        #else:
        #    self.load_hmf()

        self._is_loaded = False

        if self.pf['hmf_dfcolldz_smooth']:
            assert self.pf['hmf_dfcolldz_smooth'] % 2 != 0, \
                'hmf_dfcolldz_smooth must be odd!'

    @property
    def Mmax_ceil(self):
        if not hasattr(self, '_Mmax_ceil'):
            self._Mmax_ceil= 1e18
        return self._Mmax_ceil

    @property
    def logMmax_ceil(self):
        if not hasattr(self, '_logMmax'):
            self._logMmax_ceil = np.log10(self.Mmax_ceil)
        return self._logMmax_ceil

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(pf=self.pf, **self.pf)
        return self._cosm


    def __getattr__(self, name):

        if (name[0] == '_'):
            raise AttributeError('Should get caught by `hasattr` (#1).')

        if name not in self.__dict__.keys():
            if self.pf['hmf_load']:
                self._load_hmf()
            else:
                # Can generate on the fly!
                if name == 'tab_MAR':
                    self.TabulateMAR()
                else:
                    self.TabulateHMF(save_MAR=False)

        # If we loaded the HMF and still don't see this attribute, then
        # either (1) something is wrong with the HMF tables we have or
        # (2) this is an attribute that lives elsewhere.
        if name not in self.__dict__.keys():
            if name.startswith('tab'):
                s = "May need to run 'python remote.py fresh hmf' or check hmf_* parameters."
                raise KeyError("HMF table element `{}` not found. {}".format(name, s))
            else:
                raise AttributeError('Should get caught by `hasattr` (#2).')

        return self.__dict__[name]

    def _load_hmf_wdm(self): # pragma: no cover

        m_X = self.pf['hmf_wdm_mass']

        if self.pf['hmf_wdm_interp']:
            wdm_file_hmfs = []
            import glob
            for wdm_file in glob.glob('{!s}/input/hmf/*'.format(ARES)):
                if self.pf['hmf_window'] in wdm_file and self.pf['hmf_model'] in wdm_file and \
                	'_wdm_' in wdm_file:
                    wdm_file_hmfs.append(wdm_file)

            wdm_m_X_from_hmf_files = [int(hmf_file[hmf_file.find('_wdm') + 5 : hmf_file.find(\
                '.')]) for hmf_file in wdm_file_hmfs]
            wdm_m_X_from_hmf_files.sort()
            #print(wdm_m_X_from_hmf_files)

            closest_mass = min(wdm_m_X_from_hmf_files, key=lambda x: abs(x - m_X))
            closest_mass_index = wdm_m_X_from_hmf_files.index(closest_mass)

            if closest_mass > m_X:
                m_X_r = closest_mass
                m_X_l = wdm_m_X_from_hmf_files[closest_mass_index - 1]
            elif closest_mass < m_X:
                m_X_l = closest_mass
                m_X_r = wdm_m_X_from_hmf_files[closest_mass_index + 1]
            else:
                m_X_l = int(m_X)
                m_X_r = m_X_l + 1
        else:
            m_X_l = int(m_X)
            m_X_r = m_X_l + 1

        _fn = self.tab_prefix_hmf(True) + '.hdf5'

        if self.pf['hmf_path'] is not None:
            _path = self.pf['hmf_path'] + '/'
        else:
            _path = "{0!s}/input/hmf/".format(ARES)

        if not os.path.exists(_path+_fn) and (not self.pf['hmf_wdm_interp']):
            raise ValueError("Couldn't find file {} and wdm_interp=False!".format(_fn))

        ##
        # Find some files to interpolate

        # Hack of _wdm_?.?? suffix
        prefix = _fn[:_fn.find('_wdm_')]

        # Look for bracketing files
        fn_l = prefix + '_wdm_{:.2f}.hdf5'.format(m_X_l)
        fn_r = prefix + '_wdm_{:.2f}.hdf5'.format(m_X_r)

        dndm = []
        ngtm = []
        mgtm = []
        tmar = []
        interp = True
        mass = [m_X_l, m_X_r]
        for i, fn in enumerate([fn_l, fn_r]):

            # OK as long as we're not relying on interpolation
            if not os.path.exists(_path + fn):
                continue

            with h5py.File(_path + fn, 'r') as f:

                tab_z = np.array(f[('tab_z')])
                tab_M = np.array(f[('tab_M')])
                tab_dndm = np.array(f[('tab_dndm')])
                tab_dndm[tab_dndm==0.0] = tiny_dndm

                #self.tab_k_lin = np.array(f[('tab_k_lin')])
                #self.tab_ps_lin = np.array(f[('tab_ps_lin')])
                #self.tab_sigma = np.array(f[('tab_sigma')])
                #self.tab_dlnsdlnm = np.array(f[('tab_dlnsdlnm')])

                tab_ngtm = np.array(f[('tab_ngtm')])
                tab_ngtm[tab_ngtm==0.0] = tiny_dndm
                tab_mgtm = np.array(f[('tab_mgtm')])
                tab_mgtm[tab_mgtm==0.0] = tiny_dndm

                if 'tab_MAR' in f:
                    if self.pf['hmf_MAR_from_CDM']:
                        fn_cdm = _path + prefix + '.hdf5'
                        cdm_file = h5py.File(fn_cdm, 'r')
                        tab_MAR = np.array(cdm_file[('tab_MAR')])
                        cdm_file.close()
                        print("# Loaded MAR from {}".format(fn_cdm))
                    else:
                        tab_MAR = np.array(f[('tab_MAR')])
                else:
                    print("# No MAR in file {}.".format(_path+fn))
                #self.tab_growth = np.array(f[('tab_growth')])

                if m_X == mass[i]:
                    interp = False
                    break

            dndm.append(tab_dndm)
            ngtm.append(tab_ngtm)
            mgtm.append(tab_mgtm)
            tmar.append(tab_MAR)

        self.tab_z = tab_z
        self.tab_M = tab_M

        if (not interp):
            self.tab_dndm = tab_dndm
            self.tab_ngtm = tab_ngtm
            self.tab_mgtm = tab_mgtm
            self._tab_MAR = tab_MAR
        else:

            assert len(dndm) == 2

            # Interpolate
            log_dndm = np.log10(dndm)
            log_ngtm = np.log10(ngtm)
            log_mgtm = np.log10(mgtm)
            log_tmar = np.log10(tmar)

            self.tab_dndm = 10**(np.diff(log_dndm, axis=0).squeeze() \
                * (m_X - m_X_l) + log_dndm[0])
            self.tab_ngtm = 10**(np.diff(log_ngtm, axis=0).squeeze() \
                * (m_X - m_X_l) + log_ngtm[0])
            self.tab_mgtm = 10**(np.diff(log_mgtm, axis=0).squeeze() \
                * (m_X - m_X_l) + log_mgtm[0])
            self._tab_MAR = 10**(np.diff(log_tmar, axis=0).squeeze() \
                * (m_X - m_X_l) + log_tmar[0])

        if interp:
        	print('# Finished interpolation in WDM mass dimension of HMF.')

    def _get_ngtm_mgtm_from_dndm(self):

        # Generates ngtm and mgtm with Murray's formulas
        ngtm = []
        mgtm = []
        for i in range(len(self.tab_z)):
            m = self.tab_M[np.logical_not(np.isnan(self.tab_M))]
            dndm = self.tab_dndm[i][np.logical_not(np.isnan(self.tab_dndm[i]))]
            dndlnm = m * dndm
            if m[-1] < m[0] * 10 ** 18 / m[3]:
                m_upper = np.arange(np.log(m[-1]), np.log(10 ** 18), np.log(m[1]) - np.log(m[0]))
                mf_func = InterpolatedUnivariateSpline(np.log(m), np.log(dndlnm), k=1)
                mf = mf_func(m_upper)

                int_upper_n = simps(np.exp(mf), dx=m_upper[2] - m_upper[1], even='first')
                int_upper_m = simps(np.exp(m_upper + mf), dx=m_upper[2] - m_upper[1], even='first')
            else:
                int_upper_n = 0
                int_upper_m = 0

            ngtm_ = np.concatenate((cumtrapz(dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1], np.zeros(1)))
            mgtm_ = np.concatenate((cumtrapz(m[::-1] * dndlnm[::-1], dx=np.log(m[1]) - np.log(m[0]))[::-1], np.zeros(1)))

            ngtm.append(ngtm_ + int_upper_n)
            mgtm.append(mgtm_ + int_upper_m)

        return np.array(ngtm), np.array(mgtm)

    def _load_hmf(self):
        """ Load table from HDF5 or binary. """

        if self._is_loaded:
            return

        if self.pf['hmf_wdm_mass'] is not None:
            return self._load_hmf_wdm()

        if self.pf['hmf_cache'] is not None:
            if len(self.pf['hmf_cache']) == 3:
                self.tab_z, self.tab_M, self.tab_dndm = self.pf['hmf_cache']
                self.tab_ngtm, self.tab_mgtm = self._get_ngtm_mgtm_from_dndm()
                # tab_MAR will be re-generated automatically if summoned,
                # as will tab_Mmin_floor.
            else:
                self.tab_z, self.tab_M, self.tab_dndm, self.tab_mgtm, \
                    self.tab_ngtm, self._tab_MAR, self.tab_Mmin_floor = \
                        self.pf['hmf_cache']
            return

        if self.pf['hmf_pca'] is not None: # pragma: no cover
            f = h5py.File(self.pf['hmf_pca'], 'r')
            self.tab_z = np.array(f[('tab_z')])
            self.tab_M = np.array(f[('tab_M')])

            tab_dndm_pca = self.pf['hmf_pca_coef0'] * np.array(f[('e_vec')])[0]
            for i in range(1, len(f[('e_vec')])):
                tab_dndm_pca += self.pf['hmf_pca_coef{}'.format(i)] * np.array(f[('e_vec')])[i]

            self.tab_dndm = 10**np.array(tab_dndm_pca)

            self.tab_ngtm, self.tab_mgtm = self._get_ngtm_mgtm_from_dndm()

            f.close()

            if (not self.pf['hmf_gen_MAR']) and (ARES is not None):
                _hmf_def_ = HaloMassFunction()

                # Interpolate to common (z, Mh) grid
                _MAR_ = RectBivariateSpline(_hmf_def_.tab_z,
                    np.log10(_hmf_def_.tab_M), np.log10(_hmf_def_.tab_MAR))

                logM = np.log10(self.tab_M)
                self.tab_MAR = np.zeros((self.tab_z.size, self.tab_M.size))

                for i, z in enumerate(self.tab_z):
                    self.tab_MAR[i,:] = 10**_MAR_(z, logM)

            elif self.pf['hmf_gen_MAR']:
                self.TabulateMAR()

        elif self.tab_name is None:
            _path = self.pf['hmf_path'] \
                if self.pf['hmf_path'] is not None \
                else'{0!s}/input/hmf'.format(ARES)

            _prefix = self.tab_prefix_hmf(True)
            _fn_ = '{0!s}/{1!s}'.format(_path, _prefix)
            raise IOError("Did not find HMF table suitable for given parameters. Was looking for {}".format(_fn_))

        elif ('.hdf5' in self.tab_name) or ('.h5' in self.tab_name):
            f = h5py.File(self.tab_name, 'r')
            self.tab_z = np.array(f[('tab_z')])
            self.tab_M = np.array(f[('tab_M')])
            self.tab_dndm = np.array(f[('tab_dndm')])

            #if self.pf['hmf_load_ps']:
            self.tab_k_lin = np.array(f[('tab_k_lin')])
            self.tab_ps_lin = np.array(f[('tab_ps_lin')])
            self.tab_sigma = np.array(f[('tab_sigma')])
            self.tab_dlnsdlnm = np.array(f[('tab_dlnsdlnm')])

            self.tab_ngtm = np.array(f[('tab_ngtm')])
            self.tab_mgtm = np.array(f[('tab_mgtm')])
            if 'tab_MAR' in f:
                self._tab_MAR = np.array(f[('tab_MAR')])
            self.tab_growth = np.array(f[('tab_growth')])

            f.close()
        else:
            raise IOError('Unrecognized format for hmf_table.')

        self._is_loaded = True

        if self.pf['verbose'] and rank == 0:
            name = self.tab_name
            print("# Loaded {}.".format(name.replace(ARES, '$ARES')))

        if self.pf['hmf_func'] is not None:
            if self.pf['verbose']:
                print("Overriding tabulated HMF in favor of user-supplied ``hmf_func``.")

            # Look for extra kwargs
            hmf_kwargs = ['hmf_extra_par{}'.format(i) for i in range(5)]
            kw = {par:self.pf[par] for par in hmf_kwargs}
            for par in CosmologyParameters():
                kw[par] = self.pf[par]

            kw['hmf_window'] = self.pf['hmf_window']

            self.tab_dndm = self.pf['hmf_func'](**kw)
            assert self.tab_dndm.shape == (self.tab_z.size, self.tab_M.size), \
                "Must return dndm in native shape (z, Mh)!"

            # Need to re-calculate mgtm and ngtm also.
            self.tab_ngtm = np.zeros_like(self.tab_dndm)
            self.tab_mgtm = np.zeros_like(self.tab_dndm)

            for i, z in enumerate(self.tab_z):
                self.tab_dndm[i,np.argwhere(np.isnan(self.tab_dndm[i]))] = 1e-70
                ngtm_0 = np.trapz(self.tab_dndm[i] * self.tab_M,
                    x=np.log(self.tab_M))
                mgtm_0 = np.trapz(self.tab_dndm[i] * self.tab_M**2,
                    x=np.log(self.tab_M))
                self.tab_ngtm[i,:] = ngtm_0 \
                    - cumtrapz(self.tab_dndm[i] * self.tab_M,
                        x=np.log(self.tab_M), initial=0.0)
                self.tab_mgtm[i,:] = mgtm_0 \
                    - cumtrapz(self.tab_dndm[i] * self.tab_M**2,
                        x=np.log(self.tab_M), initial=0.0)

            # Keep it positive please.
            self.tab_mgtm = np.maximum(self.tab_mgtm, 1e-70)
            self.tab_ngtm = np.maximum(self.tab_mgtm, 1e-70)

            # Reset fcoll
            if hasattr(self, '_tab_fcoll'):
                del self._tab_fcoll

    @property
    def pars_cosmo(self):
        return {'Om0':self.cosm.omega_m_0,
                'Ob0':self.cosm.omega_b_0,
                'H0':self.cosm.h70*100}

    @property
    def pars_growth(self):
        if not hasattr(self, '_pars_growth'):
            self._pars_growth = {'dlna': self.pf['hmf_dlna']}
        return self._pars_growth

    @property
    def pars_transfer(self):
        if not hasattr(self, '_pars_transfer'):
            _transfer_pars = \
               {'k_per_logint': self.pf['hmf_transfer_k_per_logint'],
                'kmax': np.log(self.pf['hmf_transfer_kmax'])}

            p = camb.CAMBparams()
            p.set_matter_power(**_transfer_pars)

            self._pars_transfer = {'camb_params': p}

        return self._pars_transfer

    @property
    def _MF(self):
        if not hasattr(self, '_MF_'):

            logMmin = self.pf['hmf_logMmin']
            logMmax = self.pf['hmf_logMmax']
            dlogM = self.pf['hmf_dlogM']
			#TODO FIX THIS OR REMOVE CODE
            from hmf import filters
            SharpK, TopHat = filters.SharpK, filters.TopHat
            #from hmf.filters import SharpK, TopHat
            if self.pf['hmf_window'] == 'tophat':
                # This is the default in hmf
                window = TopHat
            elif self.pf['hmf_window'].lower() == 'sharpk':
                window = SharpK
            else:
                raise ValueError("Unrecognized window function.")

            MFclass = MassFunction if self.pf['hmf_wdm_mass'] is None \
                else MassFunctionWDM
            xtras = {'wdm_mass': self.pf['hmf_wdm_mass']} \
                if self.pf['hmf_wdm_mass'] is not None else {}

            # Initialize Perturbations class
            self._MF_ = MFclass(Mmin=logMmin, Mmax=logMmax,
                dlog10m=dlogM, z=self.tab_z[0], filter_model=window,
                hmf_model=self.hmf_func, cosmo_params=self.pars_cosmo,
                growth_params=self.pars_growth, sigma_8=self.cosm.sigma8,
                n=self.cosm.primordial_index, transfer_params=self.pars_transfer,
                dlnk=self.pf['hmf_dlnk'], lnk_min=self.pf['hmf_lnk_min'],
                lnk_max=self.pf['hmf_lnk_max'], hmf_params=self.pf['hmf_params'],
                use_splined_growth=self.pf['hmf_use_splined_growth'],\
                filter_params=self.pf['filter_params'], **xtras)

        return self._MF_

    @_MF.setter
    def _MF(self, value):
        self._MF_ = value

    @property
    def tab_dndlnm(self):
        if not hasattr(self, '_tab_dndlnm'):
            self._tab_dndlnm = self.tab_M * self.tab_dndm
        return self._tab_dndlnm

    @property
    def tab_fcoll(self):
        if not hasattr(self, '_tab_fcoll'):
            self._tab_fcoll = self.tab_mgtm / self.cosm.mean_density0
        return self._tab_fcoll

    @property
    def tab_bias(self):
        if not hasattr(self, '_tab_bias'):
            self._tab_bias = np.zeros((self.tab_z.size, self.tab_M.size))

            for i, z in enumerate(self.tab_z):
                self._tab_bias[i] = self.get_bias(z)

        return self._tab_bias

    @property
    def tab_t(self):
        if not hasattr(self, '_tab_t'):
            tab_z = self.tab_z
        return self._tab_t

    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            if (self.pf['hmf_table'] is not None) or (self.pf['hmf_pca'] is not None):

                if self._is_loaded:
                    raise AttributeError('this shouldnt happen!')

                self._load_hmf()

            elif self.pf['hmf_dt'] is None:

                dz = self.pf['hmf_dz']
                zmin = max(self.pf['hmf_zmin'] - 2*dz, 0.0)
                zmax = self.pf['hmf_zmax'] + 2*dz

                Nz = int(round(((zmax - zmin) / dz) + 1, 1))
                self._tab_z = np.linspace(zmin, zmax, Nz)
            else:
                dt = self.pf['hmf_dt'] # Myr

                tmin = max(self.pf['hmf_tmin'] - 2*dt, 20.)
                tmax = self.pf['hmf_tmax'] + 2*dt

                Nt = Nz = int(round(((tmax - tmin) / dt) + 1, 1))
                self._tab_t = np.linspace(tmin, tmax, Nt)[-1::-1]
                self._tab_z = self.cosm.z_of_t(self.tab_t * s_per_myr)

        return self._tab_z

    @tab_z.setter
    def tab_z(self, value):
        self._tab_z = value

    def prep_for_cache(self):
        keys = ['tab_z', 'tab_M', 'tab_dndm', 'tab_mgtm', 'tab_ngtm',
            'tab_MAR', 'tab_Mmin_floor']
        hist = [self.__getattribute__(key) for key in keys]
        return hist

    @property
    def info(self):
        if rank == 0:
            print_hmf(self)

    def TabulateHMF(self, save_MAR=True):
        """
        Build a lookup table for the halo mass function / collapsed fraction.

        Can be run in parallel.
        """

        # Initialize the MassFunction object.
        # Will setup an array of masses
        MF = self._MF

        # Masses in hmf are really Msun / h
        if hmf_vers < 3 and self.pf['hmf_wdm_mass'] is None:
            self.tab_M = self._MF.M / self.cosm.h70
        else:
            self.tab_M = self._MF.m / self.cosm.h70

        # Main quantities of interest.
        self.tab_dndm = np.zeros([self.tab_z.size, self.tab_M.size])
        self.tab_mgtm = np.zeros_like(self.tab_dndm)
        self.tab_ngtm = np.zeros_like(self.tab_dndm)

        # Extras
        self.tab_k_lin  = self._MF.k * self.cosm.h70
        self.tab_ps_lin = np.zeros([len(self.tab_z), len(self.tab_k_lin)])
        self.tab_growth = np.zeros_like(self.tab_z)

        pb = ProgressBar(len(self.tab_z), 'hmf', use=self.pf['progress_bar'])
        pb.start()

        for i, z in enumerate(self.tab_z):

            if i > 0:
                self._MF.update(z=z)

            if i % size != rank:
                continue

            # Undo little h for all main quantities
            self.tab_dndm[i] = self._MF.dndm.copy() * self.cosm.h70**4
            self.tab_mgtm[i] = self._MF.rho_gtm.copy() * self.cosm.h70**2
            self.tab_ngtm[i] = self._MF.ngtm.copy() * self.cosm.h70**3

            self.tab_ps_lin[i] = self._MF.power.copy() / self.cosm.h70**3
            self.tab_growth[i] = self._MF.growth_factor * 1.

            pb.update(i)

        pb.finish()

        # All processors will have this.
        self.tab_sigma = self._MF._sigma_0
        self.tab_dlnsdlnm = self._MF._dlnsdlnm

        # Collect results!
        if size > 1: # pragma: no cover
            tmp2 = np.zeros_like(self.tab_dndm)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_dndm, tmp2)
            self.tab_dndm = tmp2

            tmp3 = np.zeros_like(self.tab_ngtm)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_ngtm, tmp3)
            self.tab_ngtm = tmp3

            tmp4 = np.zeros_like(self.tab_mgtm)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_mgtm, tmp4)
            self.tab_mgtm = tmp4

            tmp6 = np.zeros_like(self.tab_ps_lin)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_ps_lin, tmp6)
            self.tab_ps_lin = tmp6

            tmp7 = np.zeros_like(self.tab_growth)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_growth, tmp7)
            self.tab_growth = tmp7
        ##
        # Done!
        ##
        if not save_MAR:
            return

        self.TabulateMAR()

    def TabulateMAR(self):
        ##
        # Generate halo growth histories
        ##


        pb = ProgressBar(self.tab_z.size+self.tab_M.size, 'mar')
        pb.start()

        # First, do the cumulative number density calculation.
        # This is actually the slowest part.
        MM = np.zeros((self.tab_z.size+self.tab_M.size, self.tab_z.size))

        for j, i in enumerate(range(self.tab_z.size-1, 0, -1)):
            if i % size != rank:
                continue
            MM[i] = self._run_CND(i,0)

            pb.update(j)

        # Find trajectories for more massive halos with zform=zfirst
        for j, i in enumerate(range(1, self.tab_M.size)):
            if i % size != rank:
                continue
            MM[i+self.tab_z.size-1] = self._run_CND(self.tab_z.size-1, i)

            pb.update(j+self.tab_z.size-1)

        pb.finish()

        self.tab_traj = MM


        if size > 1: # pragma: no cover
            tmp = np.zeros_like(self.tab_traj)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_traj, tmp)
            self.tab_traj = tmp.copy()
            del tmp

        # The first dimension is halo identity, the first self.tab_z.size
        # elements are halos with M=self.tab_M[0], the next self.tab_M.size
        # elements are halos with M=self.tab_M and formation redshifts
        # equal to self.tab_z[-1]. The second dimension is a series
        # of masses at corresponding redshifts in self.tab_z.

        dtdz = self.cosm.dtdz(self.tab_z)[1:-1]
        # Step 0: Compute dMdt for each history.
        # Step 1: Interpolate dMdt onto tab_M grid.
        tab_dMdt_of_z = np.zeros((self.tab_traj.shape[0], self.tab_z.size))
        tab_dMdt_of_M = np.zeros((self.tab_traj.shape[0], self.tab_M.size))
        for i, hist in enumerate(self.tab_traj):
            # 'hist' is the trajectory of a halo

            # Remember that mass increases as z decreases, so we flip

            z, dmdz = central_difference(self.tab_z, hist)

            mofz = hist[1:-1] * 1

            # Interpolate accretion rates onto common mass grid
            dmdt = dmdz[-1::-1] * s_per_yr / -dtdz[-1::-1]
            _interp1 = interp1d(np.log(mofz[-1::-1]), np.log(dmdt),
                kind='linear', bounds_error=False, fill_value=-np.inf)
            _interp2 = interp1d(np.log(z[-1::-1]), np.log(dmdt),
                kind='linear', bounds_error=False, fill_value=-np.inf)
            dmdt_rg1 = np.exp(_interp1(np.log(self.tab_M)))
            dmdt_rg2 = np.exp(_interp2(np.log(self.tab_z)))

            tab_dMdt_of_M[i,:] = dmdt_rg1
            tab_dMdt_of_z[i,:] = dmdt_rg2

        ##
        # Convert from trajectories to (z, Mh) table.
        arr = np.zeros((self.tab_z.size, self.tab_M.size))
        for i, z in enumerate(self.tab_z):

            if i % size != rank:
                continue

            # At what redshift does an object of a particular mass have
            # this MAR?

            # At this redshift, all objects have some mass. Find it!
            M = self.tab_traj[:,i]
            Mdot = tab_dMdt_of_z[:,i]

            ok = np.logical_and(Mdot > 0, np.isfinite(M))

            if ok.sum() == 0:
                continue

            arr[i] = np.exp(np.interp(np.log(self.tab_M), np.log(M[ok==1]),
                np.log(Mdot[ok==1])))

        self._tab_MAR = arr

        if size > 1: # pragma: no cover
            tmp = np.zeros_like(self.tab_MAR)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_MAR, tmp)
            self._tab_MAR = tmp

        ##
        # OK, *now* we're done.
        ##

    @property
    def tab_MAR(self):
        if not hasattr(self, '_tab_MAR'):
            if (not self._is_loaded) and self.pf['hmf_load']:
                poke = self.tab_dndm
            else:
                self.TabulateMAR()

        return self._tab_MAR

    @tab_MAR.setter
    def tab_MAR(self, value):
        self._tab_MAR = value

    @property
    def fcoll_Tmin(self):
        if not hasattr(self, '_fcoll_Tmin'):
            self.build_1d_splines(Tmin=self.pf['pop_Tmin'], mu=self.pf['mu'])
        return self._fcoll_Tmin

    @fcoll_Tmin.setter
    def fcoll_Tmin(self, value):
        self._fcoll_Tmin = value

    def build_1d_splines(self, Tmin, mu=0.6, return_fcoll=False,
        return_fcoll_p=True, return_fcoll_pp=False):
        """
        Construct splines for fcoll and its derivatives given a (fixed)
        minimum virial temperature.
        """

        Mmin_of_z = (self.pf['pop_Mmin'] is None) or \
            type(self.pf['pop_Mmin']) is FunctionType
        Mmax_of_z = (self.pf['pop_Tmax'] is not None) or \
            type(self.pf['pop_Mmax']) is FunctionType

        self.logM_min = np.zeros_like(self.tab_z)
        self.logM_max = np.ones_like(self.tab_z) * np.inf
        self.fcoll_Tmin = np.zeros_like(self.tab_z)
        self.dndm_Mmin = np.zeros_like(self.tab_z)
        self.dndm_Mmax = np.zeros_like(self.tab_z)
        for i, z in enumerate(self.tab_z):
            if self.pf['pop_Mmin'] is None:
                self.logM_min[i] = np.log10(self.VirialMass(z, Tmin, mu=mu))
            else:
                if type(self.pf['pop_Mmin']) is FunctionType:
                    self.logM_min[i] = np.log10(self.pf['pop_Mmin'](z))
                elif type(self.pf['pop_Mmin']) == np.ndarray:
                    self.logM_min[i] = np.log10(self.pf['pop_Mmin'][i])
                else:
                    self.logM_min[i] = np.log10(self.pf['pop_Mmin'])

            if Mmax_of_z:
                self.logM_max[i] = np.log10(self.VirialMass(z, self.pf['pop_Tmax'], mu=mu))
                self.dndm_Mmax[i] = 10**np.interp(self.logM_min[i], np.log10(self.tab_M),
                    np.log10(self.tab_dndm[i,:]))

            # For boundary term
            if Mmin_of_z:
                self.dndm_Mmin[i] = 10**np.interp(self.logM_min[i],
                    np.log10(self.tab_M), np.log10(self.tab_dndm[i,:]))

            self.fcoll_Tmin[i] = self.fcoll_2d(z, self.logM_min[i])

        # Main term: rate of change in collapsed fraction in halos that were
        # already above the threshold.
        self.ztab, self.dfcolldz_tab = \
            central_difference(self.tab_z, self.fcoll_Tmin)

        # Compute boundary term(s)
        if Mmin_of_z:
            self.ztab, dMmindz = \
                central_difference(self.tab_z, 10**self.logM_min)

            bc_min = 10**self.logM_min[1:-1] * self.dndm_Mmin[1:-1] \
                * dMmindz / self.cosm.mean_density0

            self.dfcolldz_tab -= bc_min

        if Mmax_of_z:
            self.ztab, dMmaxdz = \
                central_difference(self.tab_z, 10**self.logM_max)

            bc_max = 10**self.logM_min[1:-1] * self.dndm_Mmax[1:-1] \
                * dMmaxdz / self.cosm.mean_density0

            self.dfcolldz_tab += bc_max

        # Maybe smooth things
        if self.pf['hmf_dfcolldz_smooth']:
            if int(self.pf['hmf_dfcolldz_smooth']) > 1:
                kern = self.pf['hmf_dfcolldz_smooth']
            else:
                kern = 3

            self.dfcolldz_tab = smooth(self.dfcolldz_tab, kern)

            if self.pf['hmf_dfcolldz_trunc']:
                self.dfcolldz_tab[0:kern] = np.zeros(kern)
                self.dfcolldz_tab[-kern:] = np.zeros(kern)

            # Cut off edges of array?

        # 'cuz time and redshift are different
        self.dfcolldz_tab *= -1.

        if return_fcoll:
            fcoll_spline = interp1d(self.tab_z, self.fcoll_Tmin,
                kind=self.pf['hmf_interp'], bounds_error=False,
                fill_value=0.0)
        else:
            fcoll_spline = None

        self.dfcolldz_tab[self.dfcolldz_tab < tiny_dfcolldz] = tiny_dfcolldz

        if np.any(np.isnan(self.dfcolldz_tab)):
            self.dfcolldz_tab[np.isnan(self.dfcolldz_tab)] = tiny_dfcolldz

            if self.pf['verbose']:
                print("Some NaNs detected in dfcolldz_tab. Kludging...")


        spline = interp1d(self.ztab, np.log10(self.dfcolldz_tab),
            kind=self.pf['hmf_interp'],
            bounds_error=False, fill_value=np.log10(tiny_dfcolldz))
        dfcolldz_spline = lambda z: 10**spline.__call__(z)

        return fcoll_spline, dfcolldz_spline, None

    @property
    def tab_fcoll_2d(self):
        if not hasattr(self, '_tab_fcoll_2d'):
            # Remember that mgtm and mean_density have factors of h**2
            # so we're OK here dimensionally
            self._tab_fcoll_2d = self.tab_mgtm / self.cosm.mean_density0

            # May be unnecessary these days
            #self._tab_fcoll_2d[np.isnan(self._tab_fcoll_2d)] = 0.0

        return self._tab_fcoll_2d

    @property
    def fcoll_spline_2d(self):
        if not hasattr(self, '_fcoll_spline_2d'):
            self._fcoll_spline_2d = RectBivariateSpline(self.tab_z,
                np.log10(self.tab_M), self.tab_fcoll_2d, kx=3, ky=3)
        return self._fcoll_spline_2d

    @fcoll_spline_2d.setter
    def fcoll_spline_2d(self, value):
        self._fcoll_spline_2d = value

    def Bias(self, z):
        return self.get_bias(z)

    def get_bias(self, z):
        """
        Compute the halo bias for all halos (over self.tab_M) at redshift `z`.
        """

        g = np.interp(z, self.tab_z, self.tab_growth)

        # Note also that this is also HMF's definition of nu
        delta_sc = 1.686
        nu = (delta_sc / self.tab_sigma / g)
        nu_sq = nu**2

        # Cooray & Sheth (2002) Equations 68-69
        if self.hmf_func == 'PS':
            bias = 1. + (nu_sq - 1.) / delta_sc
        elif self.hmf_func == 'ST':
            ap, qp = 0.707, 0.3

            bias = 1. \
                + (ap * nu_sq - 1.) / delta_sc \
                + (2. * qp / delta_sc) / (1. + (ap * nu_sq)**qp)
        elif self.hmf_func == 'Tinker10':
            y = np.log10(200.)
            A = 1. + 0.24 * y * np.exp(-(4. / y)**4)
            a = 0.44 * y - 0.88
            B = 0.183
            b = 1.5
            C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4. / y)**4)
            c = 2.4

            bias = 1. - A * (nu**a / (nu**a + delta_sc**a)) + B * nu**b \
                 + C * nu**c
        else:
            raise NotImplemented('No bias for non-PS non-ST MF yet!')

        return bias

    def fcoll_2d(self, z, logMmin):
        """
        Return fraction of mass in halos more massive than 10**logMmin.
        Interpolation in 2D, x = redshift = z, y = logMass.
        """

        if self.Mmax_ceil is not None:
            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, self.logMmax_ceil))
        elif self.pf['pop_Tmax'] is not None:
            logMmax = np.log10(self.VirialMass(z, self.pf['pop_Tmax'],
                mu=self.pf['mu']))

            if logMmin >= logMmax:
                return tiny_fcoll

            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, logMmax))
        else:
            return np.squeeze(self.fcoll_spline_2d(z, logMmin))

    def dfcolldz(self, z):
        """
        Return derivative of fcoll(z).
        """

        return np.squeeze(self.dfcolldz_spline(z))

    def _run_CND(self, iz, iM=0):
        """
        "Evolve" a halo through time (assuming fixed number density).
        """

        M = np.zeros(self.tab_z.size)

        m_1 = self.tab_M[iM]
        for j in range(iz, 1, -1):

            # Find the cumulative number density of objects with m >= m_1
            ngtm_1 = np.exp(np.interp(np.log(m_1), np.log(self.tab_M),
                np.log(self.tab_ngtm[j])))
            # Find n(>M) at next timestep.
            ngtm_2 = self.tab_ngtm[j-1,:]
            # Interpolate n(>M;z) onto n(>M,z'<z)
            m_2 = np.exp(np.interp(np.log(ngtm_1),
                np.log(ngtm_2[-1::-1]),
                np.log(self.tab_M[-1::-1])))

            M[j] = m_2

            m_1 = m_2

        return M

    @property
    def tab_MAR_delayed(self):
        if not hasattr(self, '_tab_MAR_delayed'):
            tdyn = self.DynamicalTime(self.tab_z)

            MAR = self.tab_MAR

        return self._tab_MAR_delayed

    def MAR_func(self, z, M, grid=True):
        return self.MAR_func_(z, M, grid=grid)

    @property
    def MAR_func_(self):
        if not hasattr(self, '_MAR_func_'):
            mask = np.isfinite(self.tab_MAR)

            tab = np.log(self.tab_MAR)
            bad = np.logical_or(np.isnan(self.tab_MAR), np.isinf(tab))
            tab[bad==1] = -50

            _MAR_func = RectBivariateSpline(self.tab_z, np.log(self.tab_M), tab)

            self._MAR_func_ = lambda z, M, grid=True: np.exp(_MAR_func(z,
                np.log(M), grid=grid)).squeeze()

        return self._MAR_func_

    def VirialTemperature(self, z, M, mu=0.6):
        return self.get_Tvir(z, M, mu=mu)

    def get_Tvir(self, z, M, mu=0.6):
        """
        Compute virial temperature corresponding to halo of given mass and
        collapse redshift.

        Equation 26 in Barkana & Loeb (2001).

        Parameters
        ----------
        M : float

        """

        return 1.98e4 * (mu / 0.6) * (M * self.cosm.h70 / 1e8)**(2. / 3.) * \
            (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) /
            self.cosm.OmegaMatter(z) / 18. / np.pi**2)**(1. / 3.) * \
            ((1. + z) / 10.)

    def VirialMass(self, z, T, mu=0.6):
        return self.get_Mvir(z, T, mu=mu)

    def get_Mvir(self, z, T, mu=0.6):
        """
        Compute virial mass corresponding to halo of given virial temperature
        and collapse redshift.

        Equation 26 in Barkana & Loeb (2001), rearranged.
        """

        return (1e8 / self.cosm.h70) * (T / 1.98e4)**1.5 * (mu / 0.6)**-1.5 \
            * (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) \
            / self.cosm.OmegaMatter(z) / 18. / np.pi**2)**-0.5 \
            * ((1. + z) / 10.)**-1.5

    def VirialRadius(self, z, M, mu=0.6):
        return self.get_Rvir(z, M, mu=mu)

    def get_Rvir(self, z, M, mu=0.6):
        """
        Compute virial radius corresponding to halo of given virial mass
        and collapse redshift.

        Equation 24 in Barkana & Loeb (2001).
        """

        return 0.784 * (M * self.cosm.h70 / 1e8)**(1. / 3.) \
            * (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) \
            / self.cosm.OmegaMatter(z) / 18. / np.pi**2)**(-1. / 3.) \
            * ((1. + z) / 10.)**-1.

    def CircularVelocity(self, z, M, mu=0.6):
        return self.get_vcirc(z, M, mu=mu)

    def get_vcirc(self, z, M, mu=0.6):
        return np.sqrt(G * M * g_per_msun / self.VirialRadius(z, M, mu) / cm_per_kpc)

    def EscapeVelocity(self, z, M, mu=0.6):
        return self.get_vesc(z, M, mu=mu)

    def get_vesc(self, z, M, mu=0.6):
        return np.sqrt(2. * G * M * g_per_msun / self.VirialRadius(z, M, mu) / cm_per_kpc)

    def MassFromVc(self, z, Vc):
        cterm = (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) \
            / self.cosm.OmegaMatter(z) / 18. / np.pi**2)
        return (1e8 / self.cosm.h70) \
            *  (Vc / 23.4)**3 / cterm**0.5 / ((1. + z) / 10.)**1.5

    def BindingEnergy(self, z, M, mu=0.6):
        return (0.5 * G * (M * g_per_msun)**2 / self.VirialRadius(z, M, mu)) \
            * self.cosm.fbaryon / cm_per_kpc

    def MassFromEb(self, z, Eb, mu=0.6):
        # Could do this analytically but I'm lazy
        func = lambda M: abs(np.log10(self.BindingEnergy(z, 10**M, mu=mu)) - np.log10(Eb))
        return 10**fsolve(func, x0=7.)[0]

    def MeanDensity(self, z, M, mu=0.6):
        """
        Mean density in g / cm^3.
        """
        V = 4. * np.pi * self.VirialRadius(z, M, mu)**3 / 3.
        return (M / V) * g_per_msun / cm_per_kpc**3

    def JeansMass(self, z, M, mu=0.6):
        rho = self.MeanDensity(z, M, mu)
        T = self.VirialTemperature(z, M, mu)
        cs = np.sqrt(k_B * T / m_H)

        l = np.sqrt(np.pi * cs**2 / G / rho)
        return 4. * np.pi * rho * (0.5 * l)**3 / 3. / g_per_msun

    def DynamicalTime(self, z, M=1e12, mu=0.6):
        return self.get_tdyn(z, M=M, mu=mu)

    def get_tdyn(self, z, M=1e12, mu=0.6):
        """
        Doesn't actually depend on mass, just need to plug something in
        so we don't crash.
        """
        return np.sqrt(self.VirialRadius(z, M, mu)**3 * cm_per_kpc**3 \
            / G / M / g_per_msun)

    @property
    def tab_Mmin_floor(self):
        if not hasattr(self, '_tab_Mmin_floor'):
            if not self._is_loaded:
                if self.pf['hmf_load']:
                    self._load_hmf()
                    if hasattr(self, '_tab_Mmin_floor'):
                        return self._tab_Mmin_floor

            self._tab_Mmin_floor = np.array([self._tegmark(z) for z in self.tab_z])
        return self._tab_Mmin_floor

    @tab_Mmin_floor.setter
    def tab_Mmin_floor(self, value):
        assert len(value) == len(self.tab_z)
        self._tab_Mmin_floor = value

    @property
    def Tegmark(self):
        if not hasattr(self, '_Tegmark'):
            def func(z):
                return np.interp(z, self.tab_z, self.tab_Mmin_floor)
            self._Tegmark = func
        return self._Tegmark

    def _tegmark(self, z):
        fH2s = lambda T: 3.5e-4 * (T / 1e3)**1.52
        fH2c = lambda T: 1.6e-4 * ((1. + z) / 20.)**-1.5 \
            * (1. + (10. * (T / 1e3)**3.5) / (60. + (T / 1e3)**4))**-1. \
            * np.exp(512. / T)

        to_min = lambda T: abs(fH2s(T) - fH2c(T))
        Tcrit = fsolve(to_min, 2e3)[0]

        M = self.VirialMass(z, Tcrit)

        return M

    def Mmin_floor(self, zarr):
        if self.pf['feedback_streaming']:
            vbc = self.pf['feedback_vel_at_rec'] * (1. + zarr) / 1100.
            # Anastasia's "optimal fit"
            Vcool = np.sqrt(3.714**2 + (4.015 * vbc)**2)
            Mmin_vbc = self.MassFromVc(zarr, Vcool)
        else:
            Mmin_vbc = np.zeros_like(zarr)

        Mmin_H2 = self.Tegmark(zarr)

        return np.maximum(Mmin_vbc, Mmin_H2)
        #return Mmin_vbc + Mmin_H2

    def tab_prefix_hmf(self, with_size=False):
        """
        What should we name this table?

        Convention:
        hmf_FIT_logM_nM_logMmin_logMmax_z_nz_

        Read:
        halo mass function using FIT form of the mass function
        using nM mass points between logMmin and logMmax
        using nz redshift points between zmin and zmax

        """

        M1, M2 = self.pf['hmf_logMmin'], self.pf['hmf_logMmax']


        if self.pf['hmf_dt'] is None:
            z1, z2 = self.pf['hmf_zmin'], self.pf['hmf_zmax']

            # Just use integer redshift bounds please.
            assert z1 % 1 == 0
            assert z2 % 1 == 0

            z1 = int(z1)
            z2 = int(z2)

            s = 'z'

            zsize = ((self.pf['hmf_zmax'] - self.pf['hmf_zmin']) \
                / self.pf['hmf_dz']) + 1

        else:
            t1, t2 = self.pf['hmf_tmin'], self.pf['hmf_tmax']

            # Just use integer redshift bounds please.
            assert t1 % 1 == 0
            assert t2 % 1 == 0

            # really times just use z for formatting below.
            t1 = z1 = int(t1)
            t2 = z2 = int(t2)

            s = 't'

            tsize = zsize = ((self.pf['hmf_tmax'] - self.pf['hmf_tmin']) \
                / self.pf['hmf_dt']) + 1


        if with_size:
            logMsize = (self.pf['hmf_logMmax'] - self.pf['hmf_logMmin']) \
                / self.pf['hmf_dlogM']


            assert logMsize % 1 == 0
            logMsize = int(logMsize)
            assert zsize % 1 == 0
            zsize = int(round(zsize, 1))

            s = 'hmf_{0!s}_{1!s}_logM_{2}_{3}-{4}_{5}_{6}_{7}-{8}'.format(\
                self.hmf_func, self.cosm.get_prefix(),
                logMsize, M1, M2, s, zsize, z1, z2)

        else:

            s = 'hmf_{0!s}_{1!s}_logM_*_{2}-{3}_{4}_*_{5}-{6}'.format(\
                self.hmf_func, self.cosm.get_prefix(), M1, M2, s, z1, z2)

        if self.pf['hmf_window'].lower() != 'tophat':
            s += '_{}'.format(self.pf['hmf_window'].lower())

        if self.pf['hmf_wdm_mass'] is not None:
        	#TODO: For Testing, the assertion is for correct nonlinear fits.
            #assert self.pf['hmf_window'].lower() == 'sharpk'
            s += '_wdm_{:.2f}'.format(self.pf['hmf_wdm_mass'])

        return s

    def SaveHMF(self, fn=None, clobber=False, destination=None, fmt='hdf5',
        save_MAR=True):
        self.save(fn=fn, clobber=clobber, destination=destination, fm=fmt,
            save_MAR=save_MAR)

    def save(self, fn=None, clobber=False, destination=None, fmt='hdf5',
        save_MAR=True):
        """
        Save mass function table to HDF5 or binary (via pickle).

        Parameters
        ----------
        fn : str (optional)
            Name of file to save results to. If None, will use
            self.tab_prefix_hmf and value of format parameter to make one up.
        clobber : bool
            Overwrite pre-existing files of the same name?
        destination : str
            Path to directory (other than CWD) to save table.
        format : str
            Format of output. Can be 'hdf5' or 'pkl'

        """

        try:
            import hmf
            hmf_v = hmf.__version__
        except AttributeError:
            hmf_v = 'unknown'

        if destination is None:
            destination = '.'

        # Determine filename
        if fn is None:
            fn = '{0!s}/{1!s}.{2!s}'.format(destination,\
                self.tab_prefix_hmf(True), fmt)
        else:
            if fmt not in fn:
                print("Suffix of provided filename does not match chosen format.")
                print("Will go with format indicated by filename suffix.")

        if os.path.exists(fn):
            if clobber and rank == 0:
                os.remove(fn)
            else:
                raise IOError(('File {!s} exists! Set clobber=True or ' +\
                    'remove manually.').format(fn))

        # Do this first! (Otherwise parallel runs will be garbage)
        self.TabulateHMF(save_MAR)

        if rank > 0:
            return

        if fmt == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('tab_z', data=self.tab_z)
            f.create_dataset('tab_M', data=self.tab_M)
            f.create_dataset('tab_dndm', data=self.tab_dndm)
            f.create_dataset('tab_ngtm', data=self.tab_ngtm)
            f.create_dataset('tab_mgtm', data=self.tab_mgtm)

            if save_MAR:
                f.create_dataset('tab_MAR', data=self.tab_MAR)

            f.create_dataset('tab_Mmin_floor', data=self.tab_Mmin_floor)
            f.create_dataset('tab_growth', data=self.tab_growth)
            f.create_dataset('tab_sigma', data=self.tab_sigma)
            f.create_dataset('tab_dlnsdlnm', data=self.tab_dlnsdlnm)
            f.create_dataset('tab_k_lin', data=self.tab_k_lin)
            f.create_dataset('tab_ps_lin', data=self.tab_ps_lin)

            f.create_dataset('hmf-version', data=hmf_v)

            # Save cosmology
            grp = f.create_group('cosmology')
            grp.attrs.update(cosmology_name=self.pf['cosmology_name'],
                cosmology_id=self.pf['cosmology_id'])

            grp.create_dataset('omega_m_0', data=self.cosm.omega_m_0)
            grp.create_dataset('omega_l_0', data=self.cosm.omega_l_0)
            grp.create_dataset('sigma_8', data=self.cosm.sigma_8)
            grp.create_dataset('h70', data=self.cosm.h70)
            grp.create_dataset('omega_b_0', data=self.cosm.omega_b_0)
            grp.create_dataset('omega_cdm_0', data=self.cosm.omega_cdm_0)
            grp.create_dataset('helium_by_mass', data=self.cosm.Y)
            grp.create_dataset('cmb_temp_0', data=self.cosm.cmb_temp_0)
            grp.create_dataset('primordial_index', data=self.cosm.primordial_index)

            f.close()

        # Otherwise, pickle it!
        else:
            with open(fn, 'wb') as f:
                pickle.dump(self.tab_z, f)
                pickle.dump(self.tab_M, f)
                pickle.dump(self.tab_dndm, f)
                pickle.dump(self.tab_ngtm, f)
                pickle.dump(self.tab_mgtm, f)

                if save_MAR:
                    pickle.dump(self.tab_MAR, f)

                pickle.dump(self.tab_Mmin_floor, f)
                pickle.dump(self.tab_ps_lin, f)
                pickle.dump(self.tab_sigma, f)
                pickle.dump(self.tab_dlnsdlnm, f)
                pickle.dump(self.tab_k_lin, f)
                pickle.dump(self.tab_growth, f)

                # Should save cosmology
                pickle.dump({'hmf-version': hmf_v}, f)

        print('# Wrote {!s}.'.format(fn))

        return
