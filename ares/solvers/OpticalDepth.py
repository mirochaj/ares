"""

OpticalDepth.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Feb 21 11:26:50 MST 2015

Description:

"""

import inspect
import numpy as np
from ..data import ARES
import os, re, types, sys
from ..util.Pickling import read_pickle_file, write_pickle_file
from scipy.integrate import quad
from ..physics import Cosmology, Hydrogen
from scipy.interpolate import interp1d as interp1d_scipy
from ..util.Misc import num_freq_bins
from ..physics.Constants import c, h_p, erg_per_ev, lam_LyA, lam_LL
from ..util.Math import interp1d
from ..util.Warnings import no_tau_table
from ..util import ProgressBar, ParameterFile
from ..physics.CrossSections import PhotoIonizationCrossSection, \
    ApproximatePhotoIonizationCrossSection
from ..util.Warnings import tau_tab_z_mismatch, tau_tab_E_mismatch

try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False

try:
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
except ImportError:
    size = 1
    rank = 0

# Put this stuff in utils
defkwargs = \
{
 'zf':None,
 'xray_flux':None,
 'xray_emissivity': None,
 'lw_flux':None,
 'lw_emissivity': None,
 'tau':None,
 'return_rc': False,
 'energy_units':False,
 'xavg': 0.0,
 'zxavg':0.0,
}

barn = 1e-24
Mbarn = 1e-18

class OpticalDepth(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

        # Include helium opacities approximately?
        self.approx_He = self.pf['include_He'] and self.pf['approx_He']

        # Include helium opacities self-consistently?
        self.self_consistent_He = self.pf['include_He'] \
            and (not self.pf['approx_He'])

        if self.pf['approx_sigma']:
            self.sigma = ApproximatePhotoIonizationCrossSection
        else:
            self.sigma = PhotoIonizationCrossSection

        self._set_integrator()

    def _set_integrator(self):
        self.integrator = self.pf["unsampled_integrator"]
        self.sampled_integrator = self.pf["sampled_integrator"]
        self.rtol = self.pf["integrator_rtol"]
        self.atol = self.pf["integrator_atol"]
        self.divmax = int(self.pf["integrator_divmax"])

    @property
    def ionization_history(self):
        if not hasattr(self, '_ionization_history'):
            self._ionization_history = lambda z: 0.0
        return self._ionization_history

    @ionization_history.setter
    def ionization_history(self, value):
        if inspect.ismethod(interp1d):
            self._ionization_history = value
        elif isinstance(value, interp1d_scipy):
            self._ionization_history = value
        elif type(value) is not types.FunctionType:
            self._ionization_history = lambda z: value
        else:
            self._ionization_history = value

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(pf=self.pf, **self.pf)
        return self._cosm

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr

    def OpticalDepth(self):
        return self.DiffuseOpticalDepth()

    def DiffuseOpticalDepth(self, z1, z2, E, **kwargs):
        """
        Compute the optical depth between two redshifts.

        If no keyword arguments are supplied, assumes the IGM is neutral.

        Parameters
        ----------
        z1 : float
            observer redshift
        z2 : float
            emission redshift
        E : float
            observed photon energy (eV)

        Notes
        -----
        If keyword argument 'xavg' is supplied, it must be a function of
        redshift.

        Returns
        -------
        Optical depth between z1 and z2 at observed energy E.

        """

        kw = self._fix_kwargs(functionify=True, **kwargs)

        # Compute normalization factor to help numerical integrator
        #norm = self.cosm.hubble_0 / c / Mbarn

        # Temporary function to compute emission energy of observed photon
        Erest = lambda z: self.RestFrameEnergy(z1, E, z)

        # Always have hydrogen
        sHI = lambda z: self.sigma(Erest(z), species=0)

        # Figure out number densities and cross sections of everything
        if self.approx_He:
            nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
            nHeI = lambda z: nHI(z) * self.cosm.y
            sHeI = lambda z: self.sigma(Erest(z), species=1)
            nHeII = lambda z: 0.0
            sHeII = lambda z: 0.0
        elif self.self_consistent_He:
            if type(kw['xavg']) is not list:
                raise TypeError('hey! fix me')

            nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
            nHeI = lambda z: self.cosm.nHe(z) \
                * (1. - kw['xavg'](z) - kw['xavg'](z))
            sHeI = lambda z: self.sigma(Erest(z), species=1)
            nHeII = lambda z: self.cosm.nHe(z) * kw['xavg'](z)
            sHeII = lambda z: self.sigma(Erest(z), species=2)
        else:
            nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
            nHeI = sHeI = nHeII = sHeII = lambda z: 0.0

        tau_integrand = lambda z: self.cosm.dldz(z) \
            * (nHI(z) * sHI(z) + nHeI(z) * sHeI(z) + nHeII(z) * sHeII(z))

        # Integrate using adaptive Gaussian quadrature
        tau = quad(tau_integrand, z1, z2, epsrel=self.rtol,
            epsabs=self.atol, limit=self.divmax)[0] #/ norm

        return tau

    def _fix_kwargs(self, functionify=False, **kwargs):

        kw = defkwargs.copy()
        kw.update(kwargs)

        if functionify and (type(kw['xavg']) is not types.FunctionType):
            tmp = kw['xavg']
            kw['xavg'] = lambda z: tmp

        if kw['zf'] is None:
            kw['zf'] = self.pf['final_redshift']

        #if not self.pf['source_solve_rte']:
        #    pass
        #elif (kw['Emax'] is None) and self.background.solve_rte[popid] and \
        #    np.any(self.background.bands_by_pop[popid] > pop.pf['pop_Emin_xray']):
        #    kw['Emax'] = self.background.energies[popid][-1]

        return kw

    def TabulateOpticalDepth(self):
        """
        Compute optical depth as a function of (redshift, photon energy).

        Parameters
        ----------
        xavg : function
            Mean ionized fraction as a function of redshift.

        Notes
        -----
        Assumes logarithmic grid in variable x = 1 + z. Corresponding
        grid in photon energy determined in _init_xrb.

        Returns
        -------
        Optical depth table.

        """

        xavg = self.ionization_history

        if not hasattr(self, 'L'):
            self._set_xrb(use_tab=False)

        # Create array for each processor
        tau_proc = np.zeros([self.L, self.N])

        pb = ProgressBar(self.L * self.N, 'tau')
        pb.start()

        # Loop over redshift, photon energy
        for l in range(self.L):

            for n in range(self.N):
                m = l * self.N + n + 1

                if m % size != rank:
                    continue

                # Compute optical depth
                if l == (self.L - 1):
                    tau_proc[l,n] = 0.0
                else:
                    tau_proc[l,n] = self.DiffuseOpticalDepth(self.z[l],
                        self.z[l+1], self.E[n], xavg=xavg)

                pb.update(m)

        pb.finish()

        # Communicate results
        if size > 1:
            tau = np.zeros_like(tau_proc)
            nothing = MPI.COMM_WORLD.Allreduce(tau_proc, tau)
        else:
            tau = tau_proc

        self.tau = tau

        return tau

    def RestFrameEnergy(self, z, E, zp):
        """
        Return energy of a photon observed at (z, E) and emitted at zp.
        """

        return E * (1. + zp) / (1. + z)

    def ObserverFrameEnergy(self, z, Ep, zp):
        """
        What is the energy of a photon observed at redshift z and emitted
        at redshift zp and energy Ep?
        """

        return Ep * (1. + z) / (1. + zp)

    def _set_xrb(self, use_tab=True):
        """
        From parameter file, initialize grids in redshift and frequency.

        Parameters
        ----------
        Only depends on contents of self.pf.

        Notes
        -----
        If tau_Nz != None, will setup logarithmic grid in new parameter
        x = 1 + z. Then, given that R = x_{j+1} / x_j = const. for j < J, we can
        create a logarithmic photon frequency / energy grid. This technique
        is outlined in Haardt & Madau (1996) Appendix C.

        References
        ----------
        Haardt, F. & Madau, P. 1996, ApJ, 461, 20

        """

        if self.pf['tau_redshift_bins'] is None and self.pf['tau_table'] is None:

            # Set bounds in frequency/energy space
            self.E0 = self.pf['tau_Emin']
            self.E1 = self.pf['tau_Emax']

            return

        self.tabname = None

        # Use Haardt & Madau (1996) Appendix C technique for z, nu grids
        if not ((self.pf['tau_redshift_bins'] is not None or \
            self.pf['tau_table'] is not None)):
            #  and (not self.pf['approx_xrb'])?

            raise NotImplemented('whats going on here')

        if use_tab and (self.pf['tau_table'] is not None or self.pf['pop_solve_rte']):

            found = False
            if self.pf['pop_solve_rte']:

                # First, look in CWD or $ARES (if it exists)
                self.tabname = self.find_tau(self.pf['tau_prefix'])

                if self.tabname is not None:
                    found = True

            # tau_table will override any tables found automatically
            if self.pf['tau_table'] is not None:
                self.tabname = self.pf['tau_table']
            elif found:
                pass
            else:
                # Raise an error if we haven't found anything
                no_tau_table(self)
                sys.exit(1)

            # If we made it this far, we found a table that may be suitable
            z, E, tau = self.load(self.tabname)

            zmax_ok = (self.z.max() >= self.pf['first_light_redshift']) or \
                np.allclose(self.z.max(), self.pf['first_light_redshift'])

            zmin_ok = (self.z.min() <= self.pf['final_redshift']) or \
                np.allclose(self.z.min(), self.pf['final_redshift'])

            Emin_ok = (self.E0 <= self.pf['tau_Emin']) or \
                np.allclose(self.E0, self.pf['tau_Emin'])

            # Results insensitive to Emax (so long as its relatively large)
            # so be lenient with this condition (100 eV or 1% difference
            # between parameter file and lookup table)
            Emax_ok = np.allclose(self.E1, self.pf['tau_Emax'],
                atol=100., rtol=1e-2)

            # Check redshift bounds
            if not (zmax_ok and zmin_ok):
                if not zmax_ok:
                    tau_tab_z_mismatch(self, zmin_ok, zmax_ok, self.z)
                    sys.exit(1)
                else:
                    if self.pf['verbose']:
                        tau_tab_z_mismatch(self, zmin_ok, zmax_ok)

            if not (Emax_ok and Emin_ok):
                if self.pf['verbose']:
                    tau_tab_E_mismatch(self, Emin_ok, Emax_ok)

                if self.E1 < self.pf['tau_Emax']:
                    sys.exit(1)

            dlogx = np.diff(self.logx)
            if not np.all(np.abs(dlogx - np.roll(dlogx, -1)) <= tiny_dlogx):
                raise ValueError(wrong_tab_type)

        else:

            # Set bounds in frequency/energy space
            self.E0 = self.pf['tau_Emin']
            self.E1 = self.pf['tau_Emax']

            # Set up log-grid in parameter x = 1 + z
            self.x = np.logspace(np.log10(1+self.pf['final_redshift']),
                np.log10(1+self.pf['first_light_redshift']),
                int(self.pf['tau_redshift_bins']))

            self.z = self.x - 1.
            self.logx = np.log10(self.x)
            self.logz = np.log10(self.z)

            # Constant ratio between elements in x-grid
            self.R = self.x[1] / self.x[0]
            self.logR = np.log10(self.R)

            # Create mapping to frequency space
            self.N = num_freq_bins(self.x.size,
                zi=self.pf['first_light_redshift'], zf=self.pf['final_redshift'],
                Emin=self.E0, Emax=self.E1)

            # Create energy arrays
            if self.pf['tau_Emin_pin']:
                self.E = self.E0 * self.R**np.arange(self.N)
            else:
                self.E = np.flip(self.E1 * self.R**-np.arange(self.N), 0)

        # Frequency grid must be index-1-based.
        self.nn = np.arange(1, self.N+1)

        # R-squared and x-squared (crop up in CXRB calculation)
        self.Rsq = self.R**2
        self.xsq = self.x**2

        # Set attributes for z-dimensions of optical depth grid
        self.L = self.M = len(self.x)
        self.ll = self.mm = np.arange(self.L)

        self.logE = np.log10(self.E)

        self.n0 = min(self.nn)
        self.dE = np.diff(self.E)
        self.dlogE = np.diff(self.logE)

        # Pre-compute cross-sections
        self.sigma_E = np.array([np.array([self.sigma(E, i) for E in self.E]) \
            for i in range(3)])
        self.log_sigma_E = np.log10(self.sigma_E)

    def load(self, fn):
        """
        Read optical depth table.
        """

        #if (rank == 0) and self.pf['verbose']:
        #    print("Loading {!s}...".format(fn))

        if type(fn) is dict:
            self.E0 = fn['E'].min()
            self.E1 = fn['E'].max()
            self.E = fn['E']
            self.z = fn['z']
            self.x = self.z + 1
            self.N = self.E.size

            self.R = self.x[1] / self.x[0]

            self.tau = fn['tau']

        elif re.search('hdf5', fn):

            f = h5py.File(self.tabname, 'r')

            self.E0 = min(f[('photon_energy')])
            self.E1 = max(f[('photon_energy')])
            self.E = np.array(f[('photon_energy')])
            self.z = np.array(f[('redshift')])
            self.x = self.z + 1
            self.N = self.E.size

            self.R = self.x[1] / self.x[0]

            self.tau = self._tau = np.array(f[('tau')])
            f.close()

        elif re.search('pkl', fn):
            data = read_pickle_file(fn, nloads=1, verbose=False)

            self.E0 = data['E'].min()
            self.E1 = data['E'].max()
            self.E = data['E']
            self.z = data['z']
            self.x = self.z + 1
            self.N = self.E.size

            self.R = self.x[1] / self.x[0]

            self.tau = self._tau = data['tau']

        else:
            f = open(self.tabname, 'r')
            hdr = f.readline().split()[1:]

            tmp = []
            for element in hdr:
                tmp.append(float(element[element.rfind('=')+1:]))

            zmin, zmax, self.E0, self.E1 = tmp

            self.tau = self._tau = np.loadtxt(self.tabname)
            self.N = self.tau.shape[1]

            self.x = np.logspace(np.log10(1+zmin), np.log10(1.+zmax),
                int(self.tau.shape[0]))

            self.z = self.x - 1.
            self.E = np.logspace(np.log10(self.E0), np.log10(self.E1), self.N)

        # Correct for inconsistencies between parameter file and table
        if self.pf['tau_Emin'] > self.E0:
            Ediff = self.E - self.pf['tau_Emin']
            i_E0 = np.argmin(np.abs(Ediff))
            if Ediff[i_E0] < 0:
                i_E0 += 1

            self.tau[:,0:i_E0] = np.inf

        if self.pf['tau_Emax'] < self.E1:
            Ediff = self.E - self.pf['tau_Emax']
            i_E0 = np.argmin(np.abs(Ediff))
            if Ediff[i_E0] < 0:
                i_E0 += 1

            self.tau[:,i_E0+1:] = np.inf

        self.logx = np.log10(self.x)
        self.logz = np.log10(self.z)

        if self.pf['verbose']:
            print("# Loaded {}.".format(fn))

        return self.z, self.E, self.tau

    def tau_name(self, prefix=None, suffix='pkl'):
        """
        Return name of table based on its properties.
        """

        # Return right away if we supplied a table by hand
        if self.pf['tau_table'] is not None:
            return self.pf['tau_table'], None

        if suffix != self.pf['preferred_format']:
            suffix = self.pf['preferred_format']

        HorHe = 'He' if self.pf['include_He'] else 'H'

        zf = self.pf['final_redshift']
        zi = self.pf['first_light_redshift']

        L, N = self.tau_shape()

        E0 = self.pf['tau_Emin']
        E1 = self.pf['tau_Emax']

        #if self.ionization_history is not None:
        #    fn = lambda z1, z2, E1, E2: \
        #        ('optical_depth_{0!s}_{1}x{2}_z_{3}-{4}_logE_{5:.2g}-' +\
        #        '{6:.2g}.{7!s}').format(HorHe, L, N, z1, z2, E1, E2, suffix)
        #else:
        fn = lambda z1, z2, E1, E2: \
            'optical_depth_{0!s}_{1!s}_{2}x{3}_z_{4:.0f}-{5:.0f}_logE_{6:.2g}-{7:.2g}.{8!s}'.format(self.cosm.get_prefix(), HorHe, L, N, z1, z2,
            E1, E2, suffix)

        return fn(zf, zi, np.log10(E0), np.log10(E1)), fn

    def find_tau(self, prefix=None):
        """
        Find an optical depth table.
        """

        fn, fn_func = self.tau_name()

        #if rank == 0 and self.pf['verbose']:
        #    print(("Looking for optical depth table equivalent to " +\
        #        "{!s}...").format(fn))

        if prefix is None:
            ares_dir = ARES
            if not ares_dir:
                print("No ARES environment variable.")
                return None

            if self.pf['tau_path'] is None:
                input_dirs = ['{!s}/input/optical_depth'.format(ARES)]
            else:
                input_dirs = [self.pf['tau_path']]

        else:
            if isinstance(prefix, basestring):
                input_dirs = [prefix]
            else:
                input_dirs = prefix

        guess = '{0!s}/{1!s}'.format(input_dirs[0], fn)
        if os.path.exists(guess):
            return guess

        # Find exactly what table should be
        zmin, zmax, Nz, lEmin, lEmax, chem, pre, post = self._parse_tab(fn)

        ok_matches = []
        perfect_matches = []

        # Loop through input directories
        for input_dir in input_dirs:

            # Loop over files in input_dir, look for best match
            for fn1 in os.listdir(input_dir):

                if re.search('hdf5', fn1) and (not have_h5py):
                    continue

                tab_name = '{0!s}/{1!s}'.format(input_dir, fn1)

                try:
                    zmin_f, zmax_f, Nz_f, lEmin_f, lEmax_f, chem_f, p1, p2 = \
                        self._parse_tab(fn1)
                except:
                    continue

                # Dealbreakers
                if zmax_f < zmax:
                    continue
                if chem_f != chem:
                    continue
                if Nz_f < Nz:
                    continue

                # Continue with possible matches
                for fmt in ['pkl', 'hdf5']:

                    if fn1 == fn and fmt == self.pf['preferred_format']:
                        perfect_matches.append(tab_name)
                        continue

                    if c and fmt == self.pf['preferred_format']:
                        perfect_matches.append(tab_name)
                        continue

                    # If number of redshift bins and energy range right...
                    if re.search(pre, fn1) and re.search(post, fn1):
                        if re.search(fmt, fn1) and fmt == self.pf['preferred_format']:
                            perfect_matches.append(tab_name)
                        else:
                            ok_matches.append(tab_name)

                    # If number of redshift bins is right...
                    elif re.search(pre, fn1):

                        if re.search(fmt, fn1) and fmt == self.pf['preferred_format']:
                            perfect_matches.append(tab_name)
                        else:
                            ok_matches.append(tab_name)

        if perfect_matches:
            return perfect_matches[0]
        elif ok_matches:
            return ok_matches[0]
        else:
            return None

    def _parse_tab(self, fn):
        """

        """
        tmp1, tmp2 = fn.split('_z_')
        pre = tmp1[0:tmp1.rfind('x')]
        red, tmp3 = fn.split('_logE_')
        post = '_logE_' + tmp3.replace('.hdf5', '')

        # Find exactly what table should be
        zmin, zmax = list(map(float, red[red.rfind('z')+2:].partition('-')[0::2]))
        logEmin, logEmax = list(map(float, tmp3[tmp3.rfind('E')+1:tmp3.rfind('.')].partition('-')[0::2]))

        Nz = pre[pre.rfind('_')+1:]

        # Hack off Nz string and optical_depth_
        chem = pre.strip(Nz)[14:-1]#.strip('optical_depth_')

        return zmin, zmax, int(Nz), logEmin, logEmax, chem, pre, post

    def _fetch_tau(self, pop, zpf, Epf):
        """
        Look for optical depth tables. Supply corrected energy and redshift
        arrays if there is a mistmatch between those generated from information
        in the parameter file and those found in the optical depth table.

        .. note:: This will only be called from UniformBackground, and on
            populations which are using the generator framework.

        Parameters
        ----------
        popid : int
            ID # for population of interest.
        zpf : np.ndarray
            What the redshifts should be according to the parameter file.
        Epf : np.ndarray
            What the energies should be according to the parameter file.

        Returns
        -------
        Energies and redshifts, potentially revised from Epf and zpf.

        """

        # First, look in CWD or $ARES (if it exists)
        if self.pf['tau_table'] is None:
            self.tabname = self.find_tau(self.pf['tau_prefix'])
        else:
            self.tabname = self.pf['tau_table']

        if self.tabname is None:
            return zpf, Epf, None

        if not os.path.exists(self.tabname):
            raise IOError("Optical depth table {} does not exist!".format(self.tabname))

        # If we made it this far, we found a table that may be suitable
        ztab, Etab, tau = self.load(self.tabname)

        # Return right away if there's no potential for conflict
        if (zpf is None) and (Epf is None):
            return ztab, Etab, tau

        # Figure out if the tables need fixing
        zmax_ok = (round(ztab.max(), 2) >= self.pf['first_light_redshift'])

        zmin_ok = \
            (ztab.min() <= zpf.min()) or \
            np.allclose(ztab.min(), zpf.min())

        Emin_ok = \
            (Etab.min() <= Epf.min()) or \
            np.allclose(Etab.min(), Epf.min())

        # Results insensitive to Emax (so long as its relatively large)
        # so be lenient with this condition (100 eV or 1% difference
        # between parameter file and lookup table)
        Emax_ok = np.allclose(Etab.max(), Epf.max(), atol=100., rtol=1e-2)

        # Check redshift bounds
        if not (zmax_ok and zmin_ok):
            if not zmax_ok:
                tau_tab_z_mismatch(self, zmin_ok, zmax_ok, ztab)
                sys.exit(1)
            else:
                if self.pf['verbose']:
                    tau_tab_z_mismatch(self, zmin_ok, zmax_ok, ztab)

        if not (Emax_ok and Emin_ok):
            if self.pf['verbose']:
                tau_tab_E_mismatch(pop, self.tabname, Emin_ok, Emax_ok, Etab)

            if Etab.max() < Epf.max():
                sys.exit(1)

        # Correct for inconsistencies between parameter file and table
        # By effectively masking out those elements with tau -> inf
        if Epf.min() > Etab.min():
            Ediff = Etab - Epf.min()
            i_E0 = np.argmin(np.abs(Ediff))
            if Ediff[i_E0] < 0:
                i_E0 += 1

            i_E0 -= 1

            #tau[:,0:i_E0+1] = np.inf
        else:
            i_E0 = 0

        if Epf.max() < Etab.max():
            Ediff = Etab - Epf.max()
            i_E1 = np.argmin(np.abs(Ediff))
            if Ediff[i_E1] < 0:
                i_E1 += 1

            #tau[:,i_E1+1:] = np.inf
        else:
            i_E1 = None

        self.z_fetched = ztab
        self.E_fetched = Etab[i_E0:i_E1]
        self.tau_fetched = tau[:,i_E0:i_E1]

        # We're done!
        return ztab, Etab[i_E0:i_E1], tau[:,i_E0:i_E1]

    def tau_shape(self):
        """
        Determine dimensions of optical depth table.

        Unfortunately, this is a bit redundant with the procedure in
        self._init_xrb, but that's the way it goes.
        """

        # Set up log-grid in parameter x = 1 + z
        x = np.logspace(np.log10(1+self.pf['final_redshift']),
            np.log10(1+self.pf['first_light_redshift']),
            int(self.pf['tau_redshift_bins']))
        z = x - 1.
        logx = np.log10(x)
        logz = np.log10(z)

        # Constant ratio between elements in x-grid
        R = x[1] / x[0]
        logR = np.log10(R)

        E0 = self.pf['tau_Emin']

        # Create mapping to frequency space
        E = 1. * E0
        n = 1
        while E < self.pf['tau_Emax']:
            E = E0 * R**(n - 1)
            n += 1

        # Set attributes for dimensions of optical depth grid
        L = len(x)

        # Frequency grid must be index 1-based.
        N = num_freq_bins(L, zi=self.pf['first_light_redshift'],
            zf=self.pf['final_redshift'], Emin=E0,
            Emax=self.pf['tau_Emax'])
        N -= 1

        return L, N

    def save(self, fn=None, prefix=None, suffix='pkl', clobber=False):
        """
        Write optical depth table to disk.

        Parameters
        ----------
        fn : str
            Full filename (including suffix). Will override prefix and suffix
            parameters.

        """
        if rank != 0:
            return

        if fn is None:
            if prefix is None:
                fn, func = self.tau_name(prefix=None, suffix=suffix)
            else:
                fn = prefix + '.' + suffix

        else:
            suffix = fn[fn.rfind('.')+1:]

        if os.path.exists(fn) and (not clobber):
            raise IOError(('{!s} exists! Set clobber=True to ' +\
                'overwrite.').format(fn))

        if suffix == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('tau', data=self.tau)
            f.create_dataset('redshift', data=self.z)
            f.create_dataset('photon_energy', data=self.E)
            f.close()

        elif suffix == 'pkl':
            write_pickle_file({'tau': self.tau, 'z': self.z, 'E': self.E}, fn,\
                ndumps=1, open_mode='w', safe_mode=False, verbose=False)

        else:
            print('Unrecognized suffix \'{!s}\'. Using np.savetxt...'.format(\
                suffix))
            f = open(fn, 'w')
            hdr = ("zmin={0:.4g} zmax={1:.4g} Emin={2:.8e} " +\
                "Emax={3:.8e}").format(self.z.min(), self.z.max(),\
                self.E.min(), self.E.max())
            np.savetxt(fn, self.tau, header=hdr, fmt='%.8e')

        print('Wrote {!s}.'.format(fn))
