""" 
Halo.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description: 

"""

import numpy as np
from . import Cosmology
import pickle, os, re, sys
from ..util import ParameterFile
from scipy.misc import derivative
from ..util.Warnings import no_hmf
from ..util.Math import central_difference
from ..util.ProgressBar import ProgressBar
from .Constants import g_per_msun, cm_per_mpc
from ..util.ParameterFile import ParameterFile
from scipy.interpolate import UnivariateSpline, RectBivariateSpline

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

# Force CAMB to span broader range in wavenumber
transfer_pars = \
{
 'transfer__kmax': 0.0,
 'transfer__kmax': 100.,
 'transfer__k_per_logint': 0.,
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
        self.cosm = Cosmology(
            omega_m_0=self.pf['omega_m_0'], 
            omega_l_0=self.pf['omega_l_0'], 
            omega_b_0=self.pf['omega_b_0'],  
            hubble_0=self.pf['hubble_0'],  
            helium_by_number=self.pf['helium_by_number'], 
            cmb_temp_0=self.pf['cmb_temp_0'], 
            approx_highz=self.pf['approx_highz'], 
            sigma_8=self.pf['sigma_8'],
            primordial_index=self.pf['primordial_index'])
        
        self.fn = self.pf["hmf_table"]
        self.hmf_func = self.pf['hmf_func']
        
        self.hmf_analytic = self.pf['hmf_analytic']
        
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
                self.build_fcoll_tab()
            else:
                no_hmf(self)
                sys.exit()
        else:
            self.load_table()
            
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
            
            self.build_2d_spline()
            
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
            cosmology = {'omegav':self.cosm.omega_l_0,
                         'omegac':self.cosm.omega_cdm_0,
                         'omegab':self.cosm.omega_b_0,
                         'sigma_8':self.cosm.sigma8,
                         'h':self.cosm.h70,
                         'n':self.cosm.primordial_index}
                         
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
                mf_fit=self.hmf_func, transfer_options=transfer_pars,
                **cosmology)    
                
        return self._MF   
        
    @MF.setter
    def MF(self, value):
        self._MF = value     
                
    def build_fcoll_tab(self):
        """
        Build a lookup table for the halo mass function / collapsed fraction.
        
        Can be run in parallel.
        """    
        
        cosmology = {'omegav':self.cosm.omega_l_0,
                     'omegac':self.cosm.omega_cdm_0,
                     'omegab':self.cosm.omega_b_0,
                     'sigma_8':self.cosm.sigma8,
                     'h':self.cosm.h70,
                     'n':self.cosm.primordial_index}
        
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
            mf_fit=self.hmf_func, transfer_options=transfer_pars,
            **cosmology)
            
        # Masses in hmf are in units of Msun / h
        self.M = self.MF.M * self.cosm.h70
        self.logM = np.log10(self.M)
        self.lnM = np.log(self.M)
        self.logM_over_h = np.log10(self.MF.M)
        self.Nm = self.M.size
        
        self.dndm = np.zeros([len(self.z), len(self.logM_over_h)])
        self.mgtm = np.zeros_like(self.dndm)
        self.ngtm = np.zeros_like(self.dndm)
        self.fcoll_tab = np.zeros_like(self.dndm)
        
        pb = ProgressBar(len(self.z), 'fcoll')
        pb.start()

        for i, z in enumerate(self.z):

            if i > 0:
                self.MF.update(z=z, **cosmology)
                
            if i % size != rank:
                continue
                
            # Compute collapsed fraction
            if self.hmf_func == 'PS' and self.hmf_analytic:
                delta_c = self.MF.delta_c / self.MF.growth
                self.fcoll_tab[i] = erfc(delta_c / sqrt2 / self.MF._sigma_0)
                
            else:
                
                # Has units of h**4 / cMpc**3 / Msun
                self.dndm[i] = self.MF.dndm.copy() / self.cosm.h70**4
                self.mgtm[i] = self.MF.rho_gtm.copy()
                self.ngtm[i] = self.MF.ngtm.copy() / self.cosm.h70**3
                
                # Remember that mgtm and mean_dens have factors of h**2
                # so we're OK here dimensionally
                self.fcoll_tab[i] = self.mgtm[i] / self.MF.mean_dens
                        
            pb.update(i)
            
        pb.finish()
        
        # Collect results!
        if size > 1:
            tmp = np.zeros_like(self.fcoll_tab)
            nothing = MPI.COMM_WORLD.Allreduce(self.fcoll_tab, tmp)
            self.fcoll_tab = tmp
            
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
        
        # Fix NaN elements
        self.logfcoll_tab = np.log10(self.fcoll_tab)
        self.logfcoll_tab[np.isnan(self.fcoll_tab)] = -np.inf 
        self.fcoll_tab[np.isnan(self.fcoll_tab)] = 0.0
                
        if self.Nz > 3:
            self.build_2d_spline()
            
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
        
        spline = UnivariateSpline(self.ztab, np.log10(self.dfcolldz_tab), k=3)
        dfcolldz_spline = lambda z: 10**spline.__call__(z)

        return fcoll_spline, dfcolldz_spline, None
        
    def build_2d_spline(self):                            
        """ Setup splines for fcoll. """
        
        self.fcoll_spline_2d = RectBivariateSpline(self.z, 
            self.logM, self.fcoll_tab, kx=3, ky=3)

    def fcoll(self, z, logMmin):
        """
        Return fraction of mass in halos more massive than 10**logMmin.
        Interpolation in 2D, x = redshift = z, y = logMass.
        """ 
        
        if self.pf['pop_Mmax'] is not None:
            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, np.log10(self.pf['pop_Mmax'])))
        elif self.pf['pop_Tmax'] is not None:
            logMmax = np.log10(self.VirialMass(self.pf['pop_Tmax'], z, 
                mu=self.pf['mu']))
            return np.squeeze(self.fcoll_spline_2d(z, logMmin)) \
                 - np.squeeze(self.fcoll_spline_2d(z, logMmax))             
         
        return np.squeeze(self.fcoll_spline_2d(z, logMmin))
                 
    def dfcolldz(self, z, logMmin):
        """
        Return derivative of fcoll(z).
        """
        
        return np.squeeze(self.dfcolldz_spline(z, logMmin))

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
        
        return 1.98e4 * (mu / 0.6) * (M * self.cosm.h70 / 1e8)**(2. / 3.) * \
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
                
    def table_prefix(self):
        """
        What should we name this table?
        
        Convention:
        hmt_FIT_logM_nM_logMmin_logMmax_z_nz_
        
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
        