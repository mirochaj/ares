""" 
Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description: 

"""

import os, re, sys
import numpy as np
from . import Cosmology
from ..util import ParameterFile
from scipy.misc import derivative
from ..util.Warnings import no_hmf
from scipy.integrate import cumtrapz
from ..util.Math import central_difference
from ..util.ProgressBar import ProgressBar
from ..util.ParameterFile import ParameterFile
from .Constants import g_per_msun, cm_per_mpc, s_per_yr
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

import pickle
#try:
#    import dill as pickle
#except ImportError:
#    import pickle
    
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
    from hmf import MassFunction
    have_hmf = True
except ImportError:
    have_hmf = False
    
try:
    import pycamb
    have_pycamb = True
except ImportError:
    have_pycamb = False    

ARES = os.getenv("ARES")    

sqrt2 = np.sqrt(2.)    

tiny_fcoll = 1e-18
tiny_dfcolldz = 1e-18

# Force CAMB to span broader range in wavenumber
transfer_pars = \
{
 'lnk_max': np.log(2e4),
 'dlnk': 0.01,
}

growth_pars = \
{
 'dlna': 0.0001,
}

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
        hmf_func : str
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
        compute_dndM : bool
        compute_MgtM : bool
        compute_NgtM : bool    
            
        Table Format
        ------------
        If HDF5, must have datasets z, logM, and fcoll. fcoll table should 
        have shape [len(z), len(logM)].
        If binary, should contain arrays z, logM, and a 2-D spline for fcoll 
        (which accepts arguments z and logM), in that order.
        
        """
        self.pf = ParameterFile(**kwargs)
        
        # Read in a few parameters for convenience        
        self.fn = self.pf["hmf_table"]
        self.hmf_func = self.pf['hmf_model']
        self.hmf_analytic = self.pf['hmf_analytic']
        
        # Verify that Tmax is set correctly
        if self.pf['pop_Tmax'] is not None:
            assert self.pf['pop_Tmax'] > self.pf['pop_Tmin'], \
                "Tmax must exceed Tmin!"
        
        # Look for tables in input directory
        if ARES is not None and self.pf['hmf_load']:
            fn = '%s/input/hmf/%s' % (ARES, self.table_prefix())
            if os.path.exists('%s.pkl' % fn):
                self.fn = '%s.pkl' % fn
            elif os.path.exists('%s.hdf5' % fn):
                self.fn = '%s.hdf5' % fn    
             
        if self.hmf_func == 'PS' and self.hmf_analytic:
            self.fn = None
        
        # Either create table from scratch or load one if we found a match
        if self.fn is None:
            if have_hmf and have_pycamb:
                pass
            else:
                no_hmf(self)
                sys.exit()
        else:
            self.load_table()
                        
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(
                omega_m_0=self.pf['omega_m_0'], 
                omega_l_0=self.pf['omega_l_0'], 
                omega_b_0=self.pf['omega_b_0'],  
                hubble_0=self.pf['hubble_0'],  
                helium_by_number=self.pf['helium_by_number'], 
                cmb_temp_0=self.pf['cmb_temp_0'], 
                approx_highz=self.pf['approx_highz'], 
                sigma_8=self.pf['sigma_8'],
                primordial_index=self.pf['primordial_index'])        

        return self._cosm
            
    def load_table(self):
        """ Load table from HDF5 or binary. """
        
        if re.search('.hdf5', self.fn) or re.search('.h5', self.fn):
            f = h5py.File(self.fn, 'r')
            self.z = f['z'].value
            self.logM = f['logM'].value
            self.M = 10**self.logM
            self.fcoll_tab = f['fcoll'].value
            self.dndm = f['dndm'].value
            f.close()
                        
        elif re.search('.pkl', self.fn):
            f = open(self.fn, 'rb')
            self.z = pickle.load(f)
            self.logM = pickle.load(f)
            self.M = 10**self.logM
            self.fcoll_spline_2d = pickle.load(f)
            self.dndm = pickle.load(f)
            try:
                self.ngtm = pickle.load(f)
                self.mgtm = pickle.load(f)
            except EOFError:
                pass
            f.close()

        else:
            raise IOError('Unrecognized format for hmf_table.')    
                
        self.lnM = np.log(self.M)
        self.dndlnm = self.M * self.dndm        
        self.Nz = self.z.size
        self.Nm = self.M.size
        
    @property
    def MF(self):
        if not hasattr(self, '_MF'):

            self.logMmin = self.pf['hmf_logMmin']
            self.logMmax = self.pf['hmf_logMmax']
            self.zmin = self.pf['hmf_zmin']
            self.zmax = self.pf['hmf_zmax']
            self.dlogM = self.pf['hmf_dlogM']
            self.dz = self.pf['hmf_dz']
            
            self.Nz = int((self.zmax - self.zmin) / self.dz + 1)        
            self.z = np.linspace(self.zmin, self.zmax, self.Nz)             
                         
            # Initialize Perturbations class
            self._MF = MassFunction(Mmin=self.logMmin, Mmax=self.logMmax, 
                dlog10m=self.dlogM, z=self.z[0], 
                hmf_model=self.hmf_func, cosmo_params=self.cosmo_params,
                growth_params=growth_pars, sigma_8=self.cosm.sigma8, 
                n=self.cosm.primordial_index, **transfer_pars)
                
        return self._MF   

    @MF.setter
    def MF(self, value):
        self._MF = value     
        
    @property
    def fcoll_tab(self):
        if not hasattr(self, '_fcoll_tab'):
            self.build_fcoll_tab()
        return self._fcoll_tab    
    
    @fcoll_tab.setter
    def fcoll_tab(self, value):
        self._fcoll_tab = value
        
    @property
    def cosmo_params(self):
        return {'Om0':self.cosm.omega_m_0,
                'Ob0':self.cosm.omega_b_0,
                'H0':self.cosm.h70*100}    
                
    def build_fcoll_tab(self):
        """
        Build a lookup table for the halo mass function / collapsed fraction.
        
        Can be run in parallel.
        """    
        
        self.logMmin = self.pf['hmf_logMmin']
        self.logMmax = self.pf['hmf_logMmax']
        self.zmin = self.pf['hmf_zmin']
        self.zmax = self.pf['hmf_zmax']
        self.dlogM = self.pf['hmf_dlogM']
        self.dz = self.pf['hmf_dz']
        
        self.Nz = int((self.zmax - self.zmin) / self.dz + 1)        
        self.z = np.linspace(self.zmin, self.zmax, self.Nz)
        
        self.Nm = np.logspace(self.logMmin, self.logMmax, self.dlogM).size

        if rank == 0:    
            print "\nComputing %s mass function..." % self.hmf_func    

        # Initialize Perturbations class
        self.MF = MassFunction(Mmin=self.logMmin, Mmax=self.logMmax, 
            dlog10m=self.dlogM, z=self.z[0], 
            hmf_model=self.hmf_func, cosmo_params=self.cosmo_params, 
            growth_params=growth_pars, sigma_8=self.cosm.sigma8, 
            n=self.cosm.primordial_index, **transfer_pars)
            
        # Masses in hmf are in units of Msun * h
        self.M = self.MF.M / self.cosm.h70
        self.logM = np.log10(self.M)
        self.lnM = np.log(self.M)
        
        self.Nm = self.M.size
        
        self.dndm = np.zeros([self.Nz, self.Nm])
        self.mgtm = np.zeros_like(self.dndm)
        self.ngtm = np.zeros_like(self.dndm)
        fcoll_tab = np.zeros_like(self.dndm)
        
        pb = ProgressBar(len(self.z), 'fcoll')
        pb.start()

        for i, z in enumerate(self.z):
            
            if i > 0:
                self.MF.update(z=z)
                
            if i % size != rank:
                continue
                
            # Compute collapsed fraction
            if self.hmf_func == 'PS' and self.hmf_analytic:
                delta_c = self.MF.delta_c / self.MF.growth.growth_factor(z)
                fcoll_tab[i] = erfc(delta_c / sqrt2 / self.MF._sigma_0)
                
            else:
                
                # Has units of h**4 / cMpc**3 / Msun
                self.dndm[i] = self.MF.dndm.copy() * self.cosm.h70**4
                self.mgtm[i] = self.MF.rho_gtm.copy()
                self.ngtm[i] = self.MF.ngtm.copy() * self.cosm.h70**3
                
                # Remember that mgtm and mean_density have factors of h**2
                # so we're OK here dimensionally
                fcoll_tab[i] = self.mgtm[i] / self.MF.mean_density0
                        
            pb.update(i)
            
        pb.finish()
                
        # Collect results!
        if size > 1:
            tmp = np.zeros_like(fcoll_tab)
            nothing = MPI.COMM_WORLD.Allreduce(fcoll_tab, tmp)
            _fcoll_tab = tmp
            
            tmp2 = np.zeros_like(self.dndm)
            nothing = MPI.COMM_WORLD.Allreduce(self.dndm, tmp2)
            self.dndm = tmp2
            
            tmp3 = np.zeros_like(self.ngtm)
            nothing = MPI.COMM_WORLD.Allreduce(self.ngtm, tmp3)
            self.ngtm = tmp3
            #
            #tmp4 = np.zeros_like(self.mgtm)
            #nothing = MPI.COMM_WORLD.Allreduce(self.mgtm, tmp4)
            #self.mgtm = tmp4
        else:
            _fcoll_tab = fcoll_tab   
                    
        # Fix NaN elements
        _fcoll_tab[np.isnan(_fcoll_tab)] = 0.0
        self._fcoll_tab = _fcoll_tab
                    
    def build_1d_splines(self, Tmin, mu=0.6):
        """
        Construct splines for fcoll and its derivatives given a (fixed) 
        minimum virial temperature.
        """
        
        self.logM_min = np.zeros_like(self.z)        
        self.fcoll_Tmin = np.zeros_like(self.z)
        for i, z in enumerate(self.z):
            if self.pf['pop_Mmin'] is None:
                self.logM_min[i] = np.log10(self.VirialMass(Tmin, z, mu=mu))
            else:
                self.logM_min[i] = np.log10(self.pf['pop_Mmin'])
                    
            self.fcoll_Tmin[i] = self.fcoll(z, self.logM_min[i])
        
        self.ztab, self.dfcolldz_tab = \
            central_difference(self.z, self.fcoll_Tmin)
        self.dfcolldz_tab *= -1.
        
        fcoll_spline = None
        
        # TESTING: force dfcolldz_tab > 0
        self.dfcolldz_tab[self.dfcolldz_tab < tiny_dfcolldz] = tiny_dfcolldz
        
        spline = UnivariateSpline(self.ztab, np.log10(self.dfcolldz_tab), k=3)
        dfcolldz_spline = lambda z: 10**spline.__call__(z)

        return fcoll_spline, dfcolldz_spline, None
        
    @property
    def fcoll_spline_2d(self):
        if not hasattr(self, '_fcoll_spline_2d'):
            self._fcoll_spline_2d = RectBivariateSpline(self.z, 
                self.logM, self.fcoll_tab, kx=3, ky=3)
        return self._fcoll_spline_2d
        
    @fcoll_spline_2d.setter
    def fcoll_spline_2d(self, value):
        self._fcoll_spline_2d = value
        
    def fcoll(self, z, logMmin):
        """
        Return fraction of mass in halos more massive than 10**logMmin.
        Interpolation in 2D, x = redshift = z, y = logMass.
        """ 
        
        if self.pf['pop_Mmax'] is not None:
            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, 
                    np.log10(self.pf['pop_Mmax'])))
        elif self.pf['pop_Tmax'] is not None:
            logMmax = np.log10(self.VirialMass(self.pf['pop_Tmax'], z, 
                mu=self.pf['mu']))
                
            if logMmin >= logMmax:
                return tiny_fcoll
                
            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, logMmax))             
         
        return np.squeeze(self.fcoll_spline_2d(z, logMmin))
                 
    def dfcolldz(self, z, logMmin):
        """
        Return derivative of fcoll(z).
        """
        
        return np.squeeze(self.dfcolldz_spline(z, logMmin))
        
    def MAR_via_AM(self, z):
        """
        Compute mass accretion rate by abundance matching across redshift.
    
        Parameters
        ----------
        z : int, float
            Redshift.
    
        Returns
        -------
        Array of mass accretion rates, each element corresponding to the halo
        masses in self.M.
    
        """
    
        k = np.argmin(np.abs(z - self.z))
    
        if z not in self.z:
            print "WARNING: Rounding to nearest redshift z=%.3g" % self.z[k]
    
        # For some reason flipping the order is necessary for non-bogus results
        dn_gtm_1t = cumtrapz(self.dndlnm[k][-1::-1], 
            x=self.lnM[-1::-1], initial=0.)[-1::-1]
        dn_gtm_2t = cumtrapz(self.dndlnm[k-1][-1::-1], 
            x=self.lnM[-1::-1], initial=0.)[-1::-1]
    
        dn_gtm_1 = dn_gtm_1t[-1] - dn_gtm_1t
        dn_gtm_2 = dn_gtm_2t[-1] - dn_gtm_2t
    
        # Need to reverse arrays so that interpolants are in ascending order
        M_2 = np.exp(np.interp(dn_gtm_1[-1::-1], dn_gtm_2[-1::-1], 
            self.lnM[-1::-1])[-1::-1])
    
        # Compute time difference between z bins
        dz = self.z[k] - self.z[k-1]
        dt = dz * abs(self.cosm.dtdz(z)) / s_per_yr
    
        return (M_2 - self.M) / dt
        
    @property
    def MAR_func(self):
        if not hasattr(self, '_MAR_func'):
            
            func = lambda zz: self.MAR_via_AM(zz)
            
            _MAR_tab = np.ones_like(self.dndm)
            for i, z in enumerate(self.z):
                _MAR_tab[i] = func(z)
            
            mask = np.zeros_like(_MAR_tab)
            mask[np.isnan(_MAR_tab)] = 1
            _MAR_tab[mask == 1] = 0.
            
            self._MAR_tab = np.ma.array(_MAR_tab, mask=mask)
            self._MAR_mask = mask    
            
            spl = RectBivariateSpline(self.z, self.lnM,
                self._MAR_tab, kx=3, ky=3)
                        
            self._MAR_func = lambda z, M: spl(z, np.log(M)).squeeze()
        
        return self._MAR_func
        
    def dlogfdlogt(self, z):
        """
        Logarithmic derivative of fcoll with respect to log-time.
        
        High-z approximation under effect.
        """
        return self.dfcolldz(z) * 2. * (1. + z) / 3. / self.fcoll(z)
                                                   
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
                
    def table_prefix(self):
        """
        What should we name this table?
        
        Convention:
        hmf_FIT_logM_nM_logMmin_logMmax_z_nz_
        
        Read:
        halo mass function using FIT form of the mass function
        using nM mass points between logMmin and logMmax
        using nz redshift points between zmin and zmax
        
        
        """
        
        try:
            M1, M2 = self.pf['hmf_logMmin'], self.pf['hmf_logMmax']
            prefix = 'hmf_%s_logM_%i_%i-%i_z_%i_%i-%i' % (self.hmf_func, 
                self.logM.size, M1, M2, self.z.size, self.zmin, self.zmax)           
        except AttributeError:
            M1, M2 = self.pf['hmf_logMmin'], self.pf['hmf_logMmax']
            z1, z2 = self.pf['hmf_zmin'], self.pf['hmf_zmax']
            logMsize = (self.pf['hmf_logMmax'] - self.pf['hmf_logMmin']) \
                / self.pf['hmf_dlogM']
            zsize = (self.pf['hmf_zmax'] - self.pf['hmf_zmin']) \
                / self.pf['hmf_dz'] + 1 
            return 'hmf_%s_logM_%i_%i-%i_z_%i_%i-%i' % (self.hmf_func, 
                logMsize, M1, M2, zsize, z1, z2) 
                
        return prefix           
               
    def save(self, fn=None, clobber=True, destination=None, format='hdf5'):
        """
        Save mass function table to HDF5 or binary (via pickle).
        
        Parameters
        ----------
        fn : str (optional)
            Name of file to save results to. If None, will use 
            self.table_prefix and value of format parameter to make one up.
        clobber : bool 
            Overwrite pre-existing files of the same name?
        destination : str
            Path to directory (other than CWD) to save table.
        format : str
            Format of output. Can be 'hdf5' or 'pkl'
        
        """
        
        # Do this first! (Otherwise parallel runs will be garbage)
        tab = self.fcoll_tab
        
        if rank > 0:
            return
        
        if destination is None:
            destination = '.'
        
        if format == 'hdf5':
            if fn is None:
                fn = '%s/%s.hdf5' % (destination, self.table_prefix())
            
            if not clobber:
                if os.path.exists(fn):
                    overwrite = raw_input('%s exists. Overwrite? (y/n) ' % fn)
                    if overwrite in ['n', 'no', 'False']:
                        overwrite = False
                    else:
                        overwrite = True
                else:
                    overwrite = True
            else:
                overwrite = True
            
            if not overwrite:
                return
                
            os.system('rm -f %s' % fn)    
                
            f = h5py.File(fn, 'w')
            f.create_dataset('z', data=self.z)
            f.create_dataset('logM', data=self.logM)
            f.create_dataset('fcoll', data=self.fcoll_tab)
            f.create_dataset('dndm', data=self.dndm)
            f.create_dataset('ngtm', data=self.ngtm)
            f.create_dataset('mgtm', data=self.mgtm)            
            f.close()
            
            print 'Wrote %s.' % fn
            return
            
        # Otherwise, pickle it!    
        if fn is None:
            fn = '%s/%s.pkl' % (destination, self.table_prefix())
            
            if not clobber:
                if os.path.exists(fn):
                    overwrite = raw_input('%s exists. Overwrite? (y/n) ' % fn)
                    if overwrite in ['n', 'no', 'False']:
                        overwrite = False
                    else:
                        overwrite = True
                else:
                    overwrite = True
            else:
                overwrite = True
            
            if not overwrite:
                return
                
            os.system('rm -f %s' % fn)  
                            
            f = open(fn, 'wb')            
            pickle.dump(self.z, f)
            pickle.dump(self.logM, f)
            pickle.dump(self.fcoll_spline_2d, f)
            pickle.dump(self.dndm, f)
            pickle.dump(self.ngtm, f)
            pickle.dump(self.mgtm, f)
            f.close()
            
            print 'Wrote %s.' % fn
        
