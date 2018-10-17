""" 
Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description: 

"""

import glob
import os, re, sys
import numpy as np
from . import Cosmology
from types import FunctionType
from ..util import ParameterFile
from scipy.misc import derivative
from scipy.optimize import fsolve
from ..util.Misc import get_hg_rev
from ..util.Warnings import no_hmf
from scipy.integrate import cumtrapz
from ..util.PrintInfo import print_hmf
from ..util.ProgressBar import ProgressBar
from ..util.ParameterFile import ParameterFile
from ..util.Math import central_difference, smooth
from ..util.Pickling import read_pickle_file, write_pickle_file
from .Constants import g_per_msun, cm_per_mpc, s_per_yr, G, cm_per_kpc, m_H, k_B
from scipy.interpolate import UnivariateSpline, RectBivariateSpline, interp1d

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
    hmf_vers = float(hmf.__version__[0])
except ImportError:
    have_hmf = False
    hmf_vers = 0
    
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

ARES = os.getenv("ARES")    

sqrt2 = np.sqrt(2.)    

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
                
        # Look for tables in input directory
        if ARES is not None and self.pf['hmf_load'] and (self.tab_name is None):
            prefix = self.tab_prefix_hmf(True)
            fn = '{0!s}/input/hmf/{1!s}'.format(ARES, prefix)
            # First, look for a perfect match
            if os.path.exists('{0!s}.{1!s}'.format(fn,\
                self.pf['preferred_format'])):
                self.tab_name = '{0!s}.{1!s}'.format(fn, self.pf['preferred_format'])
            # Next, look for same table different format
            elif os.path.exists('{!s}.pkl'.format(fn)):
                self.tab_name = '{!s}.pkl'.format(fn)
            elif os.path.exists('{!s}.hdf5'.format(fn)):
                self.tab_name = '{!s}.hdf5'.format(fn)
            elif os.path.exists('{!s}.npz'.format(fn)):
                self.tab_name = '{!s}.npz'.format(fn)
            else:
                # Leave resolution blank, but enforce ranges
                prefix = self.tab_prefix_hmf()
                candidates =\
                    glob.glob('{0!s}/input/hmf/{1!s}*'.format(ARES, prefix))

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
                        _Nm, _logMmin, _logMmax, _Nz, _zmin, _zmax = \
                            list(map(int, re.findall(r'\d+', candidate)))
                    
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
            self._cosm = Cosmology(**self.pf)

        return self._cosm
            
    def __getattr__(self, name):
        
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')
            
        if name not in self.__dict__.keys():
            if self.pf['hmf_load']:
                self._load_hmf()

        if name not in self.__dict__:
            s = "May need to run 'python remote.py fresh hmf' or check hmf_* parameters."
            raise KeyError("HMF table element {} not found. {}".format(name, s))
                
        return self.__dict__[name]

    def _load_hmf(self):
        """ Load table from HDF5 or binary. """
                  
        if self._is_loaded:
            return     
                            
        if ('.hdf5' in self.tab_name) or ('.h5' in self.tab_name):
            f = h5py.File(self.tab_name, 'r')
            self.tab_z = f['tab_z'].value
            self.tab_M = f['tab_M'].value
            #self.fcoll_tab = f['fcoll'].value
            self.tab_dndm = f['tab_dndm'].value

            if self.pf['hmf_load_ps']:
                self.tab_k_lin = f['tab_k_lin'].value
                self.tab_ps_lin = f['tab_ps_lin'].value
                self.tab_sigma = f['tab_sigma'].value
                self.tab_dlnsdlnm = f['tab_dlnsdlnm'].value

            self.tab_ngtm = f['tab_ngtm'].value
            self.tab_mgtm = f['tab_mgtm'].value
            if 'tab_MAR' in f:
                self.tab_MAR = f['tab_MAR'].value
            self.tab_growth = f['tab_growth'].value
            
            f.close()
        elif re.search('.npz', self.tab_name):
            f = np.load(self.tab_name)
            self.tab_z = f['tab_z']
            self.tab_M = f['tab_M']
            self.tab_dndm = f['tab_dndm']
            self.tab_ngtm = f['tab_ngtm']
            self.tab_mgtm = f['tab_mgtm']
            if 'tab_MAR' in f:
                self.tab_MAR = f['tab_MAR']
            if 'tab_Mmin_floor' in f:
                self.tab_Mmin_floor = f['tab_Mmin_floor']
            self.tab_growth = f['tab_growth']
            self.tab_sigma = f['tab_sigma']
            self.tab_dlnsdlnm = f['tab_dlnsdlnm']
            self.tab_ps_lin = f['tab_ps_lin']
            self.tab_k_lin = f['tab_k_lin']
            f.close()                        
        elif re.search('.pkl', self.tab_name):
            
            ##
            # In this case, order matters!
            ##
            
            #loaded = read_pickle_file(self.tab_name, nloads=6, verbose=False)
            #(self.z, self.logM, self.fcoll_spline_2d) = loaded[0:3]
            #(self.tab_dndm, self.tab_ngtm, self._tabmgtm) = loaded[3:6]
            #self.M = 10**self.logM

            raise IOError('broken')

            #self.fcoll_spline_2d = pickle.load(f)
            self.tab_dndm = pickle.load(f)            
            self.ngtm = pickle.load(f)
            self.mgtm = pickle.load(f)
            self.tab_MAR = pickle.load(f)
            self.tab_Mmin_floor = pickle.load(f)
            
            if self.pf['hmf_load_ps']:
                self.bias_tab = pickle.load(f)
                self.tab_ps_lin = pickle.load(f)
                self.tab_sigma = pickle.load(f)
                self.tab_dlnsdlnm = pickle.load(f)
                self.tab_k_lin = pickle.load(f)
            
            if self.pf['hmf_load_growth']:
                self.tab_growth = pickle.load(f)
            
            # Axes these?
            self.tab_ngtm = pickle.load(f)
            self.tab_mgtm = pickle.load(f)

            f.close()

        else:
            raise IOError('Unrecognized format for hmf_table.')    
                
        self._is_loaded = True
        
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
            
            dz = self.pf['hmf_dz']
            
            # Introduce ghost zones so that the derivative is defined
            # at the boundaries.
            zmin = max(self.pf['hmf_zmin'] - 2 * dz, 0)
            zmax = self.pf['hmf_zmax'] + 2 * dz
            
            Nz = int(round(((zmax - zmin) / dz) + 1, 1))
            self.tab_z = np.linspace(zmin, zmax, Nz)             

            # Initialize Perturbations class            
            self._MF_ = MassFunction(Mmin=logMmin, Mmax=logMmax, 
                dlog10m=dlogM, z=self.tab_z[0], 
                hmf_model=self.hmf_func, cosmo_params=self.pars_cosmo,
                growth_params=self.pars_growth, sigma_8=self.cosm.sigma8, 
                n=self.cosm.primordial_index, transfer_params=self.pars_transfer,
                dlnk=self.pf['hmf_dlnk'], lnk_min=self.pf['hmf_lnk_min'],
                lnk_max=self.pf['hmf_lnk_max'])
                
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
                self._tab_bias[i] = self.Bias(z)
                
        return self._tab_bias    
                                    
    def TabulateHMF(self):
        """
        Build a lookup table for the halo mass function / collapsed fraction.
        
        Can be run in parallel.
        """
        
        dz = self.pf['hmf_dz']
        zmin = max(self.pf['hmf_zmin'] - 2*dz, 0.0)
        zmax = self.pf['hmf_zmax'] + 2*dz
        dlogM = self.pf['hmf_dlogM']
        
        Nz = int(round(((zmax - zmin) / dz) + 1, 1))
        self.tab_z = np.linspace(zmin, zmax, Nz)
        
        if rank == 0:
            print_hmf(self)
            print("\nComputing {!s} mass function...".format(self.hmf_func))    

        # Initialize the MassFunction object.
        # Will setup an array of masses
        MF = self._MF

        # Masses in hmf are in units of Msun * h
        if hmf_vers < 3:
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
        
        pb = ProgressBar(len(self.tab_z), 'dndm')
        pb.start()

        for i, z in enumerate(self.tab_z):
            
            if i > 0:
                self._MF.update(z=z)
                
            if i % size != rank:
                continue
                

            # Has units of h**4 / cMpc**3 / Msun
            self.tab_dndm[i] = self._MF.dndm.copy() * self.cosm.h70**4
            self.tab_mgtm[i] = self._MF.rho_gtm.copy() * self.cosm.h70**2
            self.tab_ngtm[i] = self._MF.ngtm.copy() * self.cosm.h70**3
             
            self.tab_ps_lin[i] = self._MF.power / self.cosm.h70**3                
            self.tab_growth[i] = self._MF.growth_factor            
                                    
            pb.update(i)
            
        pb.finish()
        
        # All processors will have this.
        self.tab_sigma = self._MF._sigma_0
        self.tab_dlnsdlnm = self._MF._dlnsdlnm
                
        # Collect results!
        if size > 1:
            #tmp1 = np.zeros_like(self.fcoll_tab)
            #nothing = MPI.COMM_WORLD.Allreduce(self.fcoll_tab, tmp1)
            #self.fcoll_tab = tmp1
            
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
        
        # Done!    
            
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
                self.logM_min[i] = np.log10(self.VirialMass(Tmin, z, mu=mu))
            else:
                if type(self.pf['pop_Mmin']) is FunctionType:
                    self.logM_min[i] = np.log10(self.pf['pop_Mmin'](z))
                elif type(self.pf['pop_Mmin']) == np.ndarray:
                    self.logM_min[i] = np.log10(self.pf['pop_Mmin'][i])
                else:
                    self.logM_min[i] = np.log10(self.pf['pop_Mmin'])
                    
            if Mmax_of_z:
                self.logM_max[i] = np.log10(self.VirialMass(self.pf['pop_Tmax'], z, mu=mu))        
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
            
        self.dfcolldz_tab[self.dfcolldz_tab <= tiny_dfcolldz] = tiny_dfcolldz
                    
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
        
    @property
    def dndm_spline_2d(self):
        if not hasattr(self, '_dndm_spline_2d'):
            log10tab = np.log10(self.tab_dndm)
            log10tab[np.isinf(log10tab)] = -100.
            
            _dndm_spline_2d = RectBivariateSpline(self.tab_z, 
                np.log10(self.tab_M), log10tab, kx=3, ky=3)
            
            self._dndm_spline_2d = lambda z, logM: 10**_dndm_spline_2d(z, logM).squeeze()
                
        return self._dndm_spline_2d    

    def Bias(self, z):
                
        G = np.interp(z, self.tab_z, self.tab_growth)
               
        # Note also that this is also HMF's definition of nu
        delta_sc = 1.686
        nu = (delta_sc / self.tab_sigma / G)**2  
        
        # Cooray & Sheth (2002) Equations 68-69
        if self.hmf_func == 'PS':
            bias = 1. + (nu - 1.) / delta_sc
        elif self.hmf_func == 'ST':
            ap, qp = 0.707, 0.3
            
            bias = 1. \
                + (ap * nu - 1.) / delta_sc \
                + (2. * qp / delta_sc) / (1. + (ap * nu)**qp)
        else:
            raise NotImplemented('No bias for non-PS non-ST MF yet!')
    
        return bias
    
    @property
    def LinearPS(self):
        """
        Interpolant for the linear matter power spectrum.
        
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        lnk : int, float
            Nature log of the wavenumber of interest.
            
        """
        if not hasattr(self, '_LinearPS'):
            self._LinearPS = RectBivariateSpline(self.tab_z, 
                np.log(self.tab_k_lin), self.tab_ps_lin, kx=3, ky=3)
        return self._LinearPS
    
    def fcoll_2d(self, z, logMmin):
        """
        Return fraction of mass in halos more massive than 10**logMmin.
        Interpolation in 2D, x = redshift = z, y = logMass.
        """ 
        
        if self.Mmax_ceil is not None:
            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, self.logMmax_ceil))
        elif self.pf['pop_Tmax'] is not None:
            logMmax = np.log10(self.VirialMass(self.pf['pop_Tmax'], z, 
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
    def tab_traj(self):
        """
        For halos with M=self.tab_M at some redshift, find mass at later z.
        """
        if not hasattr(self, '_tab_traj'):

            MM = np.zeros((self.tab_z.size+self.tab_M.size, self.tab_z.size))

            for i in range(self.tab_z.size-1, 0, -1):
                MM[i] = self._run_CND(i,0)
                
            # Find trajectories for more massive halos with zform=zfirst    
            for i in range(1, self.tab_M.size):
                MM[i+self.tab_z.size-1] = self._run_CND(self.tab_z.size-1, i)
                                                    
            self._tab_traj = MM

        # The first dimension is halo identity, the first self.tab_z.size
        # elements are halos with M=self.tab_M[0], the next self.tab_M.size
        # elements are halos with M=self.tab_M and formation redshifts
        # equal to self.tab_z[-1]. The second dimension is a series
        # of masses at corresponding redshifts in self.tab_z.

        return self._tab_traj
    
    @property    
    def tab_MAR(self):
        if not hasattr(self, '_tab_MAR'):
                    
            if not self._is_loaded:
                if self.pf['hmf_load']:
                    self._load_hmf()
                    if hasattr(self, '_tab_MAR'):
                        return self._tab_MAR
                    
            print("Generating MAR. This is slow. What are you up to?")
                    
            # Differentiate trajectories, interpolate to common mass, redshift grid.
            
            
            #arr = np.zeros_like(self._tab_traj)
            
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

            # Convert from trajectories to (z, Mh) table.
            arr = np.zeros((self.tab_z.size, self.tab_M.size))    
            for i, z in enumerate(self.tab_z):

                # At what redshift does an object of a particular mass have 
                # this MAR?
                
                # At this redshift, all objects have some mass. Find it!
                M = self.tab_traj[:,i]
                Mdot = tab_dMdt_of_z[:,i]
                
                ok = np.logical_and(Mdot > 0, np.isfinite(M))
                
                if ok.sum() == 0:
                    continue
                
                #if i % 100 == 0:
                #    pl.loglog(M, Mdot)
                
                arr[i] = np.exp(np.interp(np.log(self.tab_M), np.log(M[ok==1]), 
                    np.log(Mdot[ok==1])))
                
            self._tab_MAR = arr
                        
        return self._tab_MAR
        
    @tab_MAR.setter
    def tab_MAR(self, value):
        self._tab_MAR = value
        
    def MAR_func(self, z, M):
        return self.MAR_func_(z, M)
        
    @property
    def MAR_func_(self):
        if not hasattr(self, '_MAR_func_'):
            mask = np.isfinite(self.tab_MAR)
            
            tab = np.log(self.tab_MAR)
            bad = np.logical_or(np.isnan(self.tab_MAR), np.isinf(tab))
            tab[bad==1] = -50
            
            _MAR_func = RectBivariateSpline(self.tab_z, np.log(self.tab_M), tab)
                
            self._MAR_func_ = lambda z, M: np.exp(_MAR_func(z, np.log(M))).squeeze()
        
        return self._MAR_func_
                                                                  
    def VirialTemperature(self, M, z, mu=0.6):
        """
        Compute virial temperature corresponding to halo of given mass and
        collapse redshift.
        
        Equation 26 in Barkana & Loeb (2001).
        
        Parameters
        ----------
        M : float
            
        """    
        
        return 1.98e4 * (mu / 0.6) * (M / self.cosm.h70 / 1e8)**(2. / 3.) * \
            (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) /
            self.cosm.OmegaMatter(z) / 18. / np.pi**2)**(1. / 3.) * \
            ((1. + z) / 10.)
        
    def VirialMass(self, T, z, mu=0.6):
        """
        Compute virial mass corresponding to halo of given virial temperature 
        and collapse redshift.
        
        Equation 26 in Barkana & Loeb (2001), rearranged.    
        """         
        
        return (1e8 / self.cosm.h70) * (T / 1.98e4)**1.5 * (mu / 0.6)**-1.5 \
            * (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) \
            / self.cosm.OmegaMatter(z) / 18. / np.pi**2)**-0.5 \
            * ((1. + z) / 10.)**-1.5
                
    def VirialRadius(self, M, z, mu=0.6):
        """
        Compute virial radius corresponding to halo of given virial mass 
        and collapse redshift.
        
        Equation 24 in Barkana & Loeb (2001).
        """
        
        return 0.784 * (M / self.cosm.h70 / 1e8)**(1. / 3.) \
            * (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) \
            / self.cosm.OmegaMatter(z) / 18. / np.pi**2)**(-1. / 3.) \
            * ((1. + z) / 10.)**-1.
              
    def CircularVelocity(self, M, z, mu=0.6):
        return np.sqrt(G * M * g_per_msun / self.VirialRadius(M, z, mu) / cm_per_kpc)
              
    def MassFromVc(self, Vc, z):
        cterm = (self.cosm.omega_m_0 * self.cosm.CriticalDensityForCollapse(z) \
            / self.cosm.OmegaMatter(z) / 18. / np.pi**2)
        return (1e8 / self.cosm.h70) \
            *  (Vc / 23.4)**3 / cterm**0.5 / ((1. + z) / 10)**1.5
            
    def BindingEnergy(self, M, z, mu=0.6):
        return (0.5 * G * (M * g_per_msun)**2 / self.VirialRadius(M, z, mu)) \
            * self.cosm.fbaryon / cm_per_kpc
            
    def MassFromEb(self, z, Eb, mu=0.6):
        # Could do this analytically but I'm lazy
        func = lambda M: abs(np.log10(self.BindingEnergy(10**M, z=z, mu=mu)) - np.log10(Eb))
        return 10**fsolve(func, x0=7.)[0]
            
    def MeanDensity(self, M, z, mu=0.6):
        """
        Mean density in g / cm^3.
        """
        V = 4. * np.pi * self.VirialRadius(M, z, mu)**3 / 3.
        return (M / V) * g_per_msun / cm_per_kpc**3

    def JeansMass(self, M, z, mu=0.6):
        rho = self.MeanDensity(M, z, mu)
        T = self.VirialTemperature(M, z, mu)
        cs = np.sqrt(k_B * T / m_H)
        
        l = np.sqrt(np.pi * cs**2 / G / rho)
        return 4. * np.pi * rho * (0.5 * l)**3 / 3. / g_per_msun
        
    def DynamicalTime(self, M, z, mu=0.6):
        return np.sqrt(self.VirialRadius(M, z, mu)**3 * cm_per_kpc**3 \
            / G / M / g_per_msun)
    
    @property
    def tab_Mmin_floor(self):
        if not hasattr(self, '_tab_Mmin_floor'):
            if not self._is_loaded:
                if self.pf['hmf_load']:
                    self._load_hmf()
                    if hasattr(self, '_tab_Mmin_floor'):
                        return self._tab_Mmin_floor
                        
            self._tab_Mmin_floor = np.array(map(self._tegmark, self.tab_z))
        return self._tab_Mmin_floor
            
    @tab_Mmin_floor.setter
    def tab_Mmin_floor(self, value):
        assert len(value) == len(self.tab_z)
        self._tab_Mmin_floor = value
            
    @property        
    def Tegmark(self):
        if not hasattr(self, '_Tegmark'):
            self._Tegmark = lambda z: np.interp(z, self.tab_z, self.tab_Mmin_floor)
        return self._Tegmark
        
    def _tegmark(self, z):
        fH2s = lambda T: 3.5e-4 * (T / 1e3)**1.52
        fH2c = lambda T: 1.6e-4 * ((1. + z) / 20.)**-1.5 \
            * (1. + (10. * (T / 1e3)**3.5) / (60. + (T / 1e3)**4))**-1. \
            * np.exp(512. / T)
    
        to_min = lambda T: abs(fH2s(T) - fH2c(T)) 
        Tcrit = fsolve(to_min, 2e3)[0]

        M = self.VirialMass(Tcrit, z)

        return M
    
    def Mmin_floor(self, zarr):
        if self.pf['feedback_streaming']:
            vbc = self.pf['feedback_vel_at_rec'] * (1. + zarr) / 1100.
            # Anastasia's "optimal fit"
            Vcool = np.sqrt(3.714**2 + (4.015 * vbc)**2)
            Mmin_vbc = self.MassFromVc(Vcool, zarr)
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
        z1, z2 = self.pf['hmf_zmin'], self.pf['hmf_zmax']
        
        # Just use integer redshift bounds please.
        assert z1 % 1 == 0
        assert z2 % 1 == 0
        
        z1 = int(z1)
        z2 = int(z2)
        
        if with_size:
            logMsize = (self.pf['hmf_logMmax'] - self.pf['hmf_logMmin']) \
                / self.pf['hmf_dlogM']                
            zsize = ((self.pf['hmf_zmax'] - self.pf['hmf_zmin']) \
                / self.pf['hmf_dz']) + 1
                
            assert logMsize % 1 == 0
            logMsize = int(logMsize)    
            assert zsize % 1 == 0
            zsize = int(round(zsize, 1))    
             
            s = 'hmf_{0!s}_logM_{1}_{2}-{3}_z_{4}_{5}-{6}'.format(\
                self.hmf_func, logMsize, M1, M2, zsize, z1, z2)            
                                
        else:
            
            s = 'hmf_{0!s}_logM_*_{1}-{2}_z_*_{3}-{4}'.format(\
                self.hmf_func, M1, M2, z1, z2) 
                        
        return s   
                               
    def SaveHMF(self, fn=None, clobber=True, destination=None, format='hdf5'):
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
                self.tab_prefix_hmf(True), format)    
            if rank == 0:
                print("Will save HMF to file {}".format(fn))            
        else:
            if format not in fn:
                print("Suffix of provided filename does not match chosen format.")
                print("Will go with format indicated by filename suffix.")
        
        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else:
                raise IOError(('File {!s} exists! Set clobber=True or ' +\
                    'remove manually.').format(fn))    
        
        # Do this first! (Otherwise parallel runs will be garbage)
        self.TabulateHMF()    
        
        if rank > 0:
            return
        
        
        
        if format == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('tab_z', data=self.tab_z)
            f.create_dataset('tab_M', data=self.tab_M)
            f.create_dataset('tab_dndm', data=self.tab_dndm)
            f.create_dataset('tab_ngtm', data=self.tab_ngtm)
            f.create_dataset('tab_mgtm', data=self.tab_mgtm)        
            f.create_dataset('tab_MAR', data=self.tab_MAR)
            f.create_dataset('tab_Mmin_floor', data=self.tab_Mmin_floor)
            f.create_dataset('tab_ps_lin', data=self.tab_ps_lin)
            f.create_dataset('tab_growth', data=self.tab_growth)
            f.create_dataset('tab_sigma', data=self.tab_sigma)
            f.create_dataset('tab_dlnsdlnm', data=self.tab_dlnsdlnm)
            f.create_dataset('tab_k_lin', data=self.tab_k)
            f.create_dataset('hmf-version', data=hmf_v)
            f.close()

        elif format == 'npz':
            data = {'tab_z': self.tab_z, 'tab_M': self.tab_M, 
                    'tab_dndm': self.tab_dndm,
                    'tab_ngtm': self.tab_ngtm, 
                    'tab_mgtm': self.tab_mgtm,
                    'tab_MAR': self.tab_MAR,
                    'tab_Mmin_floor': self.tab_Mmin_floor,
                    'tab_growth': self.tab_growth,
                    'tab_ps_lin': self.tab_ps_lin,
                    'tab_sigma': self.tab_sigma,
                    'tab_dlnsdlnm': self.tab_dlnsdlnm,
                    'tab_k_lin': self.tab_k_lin,
                    'pars': {'pars_growth': self.pars_growth,
                             'pars_transfer': self.pars_transfer},
                    'hmf-version': hmf_v}
            np.savez(fn, **data)

        # Otherwise, pickle it!    
        else:   
            f = open(fn, 'wb')
            pickle.dump(self.tab_z, f)
            pickle.dump(self.tab_M, f)
            pickle.dump(self.tab_dndm, f)
            pickle.dump(self.tab_ngtm, f)
            pickle.dump(self.tab_mgtm, f)
            pickle.dump(self.tab_MAR, f)
            pickle.dump(self.tab_Mmin_floor, f)
            pickle.dump(self.tab_ps_lin, f)
            pickle.dump(self.tab_sigma, f)
            pickle.dump(self.tab_dlnsdlnm, f)
            pickle.dump(self.tab_k_lin)
            pickle.dump(self.tab_growth, f)
            pickle.dump({'pars_growth': self.pars_growth,
                'pars_transfer': self.pars_transfer}, f)
            pickle.dump(dict(('hmf-version', hmf_v)))
            f.close()
            
        print('Wrote {!s}.'.format(fn))
        return
        
