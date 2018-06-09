# Thanks, Jason Sun, for most of this!

import os
import re
import pickle
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
from ..util.ProgressBar import ProgressBar
from .Constants import rho_cgs, c, cm_per_mpc
from .HaloMassFunction import HaloMassFunction

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
    
ARES = os.getenv("ARES")    

class HaloModel(HaloMassFunction):
    
    def mvir_to_rvir(self, m):
        return (3. * m / (4. * np.pi * self.pf['halo_delta'] \
            * self.cosm.mean_density0)) ** (1. / 3.)

    def cm_relation(self, m, z, get_rs):
        """
        The concentration-mass relation
        """
        if self.pf['halo_cmr'] == 'duffy':
            return self._cm_duffy(m, z, get_rs)
        elif self.pf['halo_cmr'] == 'zehavi':
            return self._cm_zehavi(m, z, get_rs)
        else:
            raise NotImplemented('help!')

    def _cm_duffy(self, m, z, get_rs=True):
        c = 6.71 * (m / (2e12)) ** -0.091 * (1 + z) ** -0.44
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _cm_zehavi(self, m, z, get_rs=True):
        c = ((m / 1.5e13) ** -0.13) * 9.0 / (1 + z)
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def _dc_nfw(self, c):
        return c** 3. / (4. * np.pi) / (np.log(1 + c) - c / (1 + c))

    def rho_nfw(self, r, m, z):

        c, r_s = self.cm_relation(m, z, get_rs=True)
        
        x = r / r_s
        rn = x / c

        if np.iterable(x):
            result = np.zeros_like(x)
            result[rn <= 1] = (self._dc_nfw(c) / (c * r_s)**3 / (x * (1 + x)**2))[rn <= 1]

            return result
        else:
            if rn <= 1.0:
                return self._dc_nfw(c) / (c * r_s) ** 3 / (x * (1 + x) ** 2)
            else:
                return 0.0

    def u_nfw(self, k, m, z):
        """
        Normalized Fourier Transform of an NFW profile.
        
        ..note:: This is Equation 81 from Cooray & Sheth (2002).
        
        Parameters
        ----------
        k : int, float
            Wavenumber
        m : 
        """
        c, r_s = self.cm_relation(m, z, get_rs=True)

        K = k * r_s

        asi, ac = sp.sici((1 + c) * K)
        bs, bc = sp.sici(K)

        # The extra factor of np.log(1 + c) - c / (1 + c)) comes in because
        # there's really a normalization factor of 4 pi rho_s r_s^3 / m, 
        # and m = 4 pi rho_s r_s^3 * the log term
        norm = 1. / (np.log(1 + c) - c / (1 + c))

        return norm * (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) \
            + np.cos(K) * (ac - bc))

    def u_isl(self, k, m, z, rmax):
        """
        Normalized Fourier transform of an r^-2 profile.

        rmax : int, float
            Effective horizon. Distance a photon can travel between
            Ly-beta and Ly-alpha.

        """

        asi, aco = sp.sici(rmax * k)

        return asi / rmax / k

    def u_isl_exp(self, k, m, z, rmax, rstar): 
        return np.arctan(rstar * k) / rstar / k
    
    def u_exp(self, k, m, z, rmax):
        rs = 1.
    
        L0 = (m / 1e11)**1.
        c = rmax / rs
    
        kappa = k * rs
    
        norm = rmax / rs**3 
    
        return norm / (1. + kappa**2)**2.

    def FluxProfile(self, r, m, z, lc=False):
        return m * self.ModulationFactor(z, r=r, lc=lc) / (4. * np.pi * r**2)
    
    #@RadialProfile.setter
    #def RadialProfile(self, value):
    #    pass
    
    def FluxProfileFT(self, k, m, z, lc=False):
        _numerator = lambda r: 4. * np.pi * r**2 * np.sin(k * r) / (k * r) \
            * self.FluxProfile(r, m, z, lc=lc)
        _denominator = lambda r: 4. * np.pi * r**2 *\
            self.FluxProfile(r, m, z, lc=lc)
        _r_LW = 97.39 * self.ScalingFactor(z)
        temp = quad(_numerator, 0., _r_LW)[0] / quad(_denominator, 0., _r_LW)[0]
        return temp
    
    def ScalingFactor(self, z):
        return (self.cosm.h70 / 0.7)**-1 * (self.cosm.omega_m_0 / 0.27)**-0.5 * ((1. + z) / 21.)**-0.5
        
    def ModulationFactor(self, z0, z=None, r=None, lc=False):
        """
        Return the modulation factor as a function of redshift or comoving distance
        - Reference: Ahn et al. 2009
        :param z0: source redshift
        :param z: the redshift (whose LW intensity is) of interest
        :param r: the distance from the source in cMpc
        :lc: True or False, including the light cone effect
        :return:
        """
        if z != None and r == None:
            r_comov = self.cosm.ComovingRadialDistance(z0, z)
        elif z == None and r != None:
            r_comov = r
        else:
            raise ValueError('Must specify either "z" or "r".')
        alpha = self.ScalingFactor(z0)
        _a = 0.167
        r_star = c * _a * self.cosm.HubbleTime(z0) * (1.+z0) / cm_per_mpc
        ans = np.maximum(1.7 * np.exp(-(r_comov / 116.29 / alpha)**0.68) - 0.7, 0.0)
        if lc == True:
            ans *= np.exp(-r/r_star)
        return ans
    
    def PS_OneHalo(self, z, k, profile_1=None, Mmin_1=None, profile_2=None,
        Mmin_2=None):
        """
        Compute the one halo term of the halo model for given input profile.
        """

        iz = np.argmin(np.abs(z - self.tab_z))

        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_1 is None:
            profile_1 = self.u_nfw
        
        prof1 = np.abs(map(lambda M: profile_1(k, M, z), self.tab_M))
            
        if profile_2 is None:
            prof2 = prof1
        else:
            prof2 = np.abs(map(lambda M: profile_2(k, M, z), self.tab_M))

        dndlnm = self.tab_dndm[iz,:] * self.tab_M
        rho_bar = self.cosm.rho_m_z0 * rho_cgs

        if Mmin_1 is None:
            fcoll_1 = 1.
            iM_1 = 0
        else:
            fcoll_1 = self.fcoll_Tmin[iz]
            iM_1 = np.argmin(np.abs(Mmin_1 - self.tab_M))
        
        if Mmin_2 is None:
            fcoll_2 = 1.
            iM_2 = 0
        else:
            fcoll_2 = self.fcoll_Tmin[iz]
            iM_2 = np.argmin(np.abs(Mmin_2 - self.tab_M))
        
        iM = max(iM_1, iM_2)
        
        integrand = dndlnm * (self.tab_M / rho_bar / fcoll_1) \
            * (self.tab_M / rho_bar / fcoll_2) * prof1 * prof2 

        result = np.trapz(integrand[iM:], x=np.log(self.tab_M[iM:]))

        return result

    def PS_TwoHalo(self, z, k, profile_1=None, Mmin_1=None, profile_2=None, 
        Mmin_2=None):
        """
        Compute the two halo term of the halo model for given input profile.
        
        .. note :: Assumption of linearity?
        
        Parameters
        ----------
        
        """
        
        iz = np.argmin(np.abs(z - self.tab_z))

        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_1 is None:
            profile_1 = self.u_nfw
        
        prof1 = np.abs(map(lambda M: profile_1(k, M, z), self.tab_M))
            
        if profile_2 is None:
            prof2 = prof1
        else:
            prof2 = np.abs(map(lambda M: profile_2(k, M, z), self.tab_M))
                        
        # Short-cuts
        dndlnm = self.tab_dndm[iz,:] * self.tab_M
        bias = self.Bias(z)
        #rho_bar = self.mgtm[iz,0]
        rho_bar = self.cosm.rho_m_z0 * rho_cgs

        if Mmin_1 is None:
            fcoll_1 = 1.#self.mgtm[iz,0] / rho_bar
            iM_1 = 0
            
            # Small halo correction.
            # Make use of Cooray & Sheth Eq. 71
            _integrand = dndlnm * (self.tab_M / rho_bar) * bias
            correction_1 = 1. - np.trapz(_integrand, x=np.log(self.tab_M))
        else:
            fcoll_1 = self.fcoll_Tmin[iz]
            iM_1 = np.argmin(np.abs(Mmin_1 - self.tab_M))
            correction_1 = 0.0
                    
        if Mmin_2 is None:
            fcoll_2 = 1.#self.mgtm[iz,0] / rho_bar
            iM_2 = 0
            _integrand = dndlnm * (self.tab_M / rho_bar) * bias
            correction_2 = 1. - np.trapz(_integrand, x=np.log(self.tab_M))
        else:
            fcoll_2 = self.fcoll_Tmin[iz]
            iM_2 = np.argmin(np.abs(Mmin_2 - self.tab_M))
            correction_2 = 0.0

        # Compute two-halo integral with profile in there
        integrand1 = dndlnm * (self.tab_M / rho_bar / fcoll_1) * prof1 * bias
        integral1 = np.trapz(integrand1[iM_1:], x=np.log(self.tab_M[iM_1:])) \
                  + correction_1

        if profile_2 is not None:
            integrand2 = dndlnm * (self.tab_M / rho_bar / fcoll_2) * prof2 * bias
            integral2 = np.trapz(integrand2[iM_2:], x=np.log(self.tab_M[iM_2:])) \
                      + correction_2
        else:
            integral2 = integral1

        return integral1 * integral2 * float(self.LinearPS(z, np.log(k)))

    def PowerSpectrum(self, z, k, profile_1=None, Mmin_1=None, profile_2=None, 
        Mmin_2=None, exact_z=True):
        
        # Tabulation only implemented for density PS at the moment.
        if self.pf['hmf_load_ps'] and (profile_1 is None):
            iz = np.argmin(np.abs(z - self.tab_z))
            if exact_z:
                assert abs(z - self.tab_z[iz]) < 1e-2, \
                    'Supplied redshift (%g) not in table!' % z
            if len(k) == len(self.tab_k):
                if np.allclose(k, self.tab_k):
                    return self.tab_ps_mm[iz]
                
            return np.interp(np.log(k), np.log(self.k), self.tab_ps_mm[iz])
                    
        if type(k) == np.ndarray:
            f1 = lambda kk: self.PS_OneHalo(z, kk, profile_1, Mmin_1, profile_2, 
                Mmin_2)
            f2 = lambda kk: self.PS_TwoHalo(z, kk, profile_1, Mmin_1, profile_2, 
                Mmin_2)
            ps1 = np.array(map(f1, k))
            ps2 = np.array(map(f2, k))
                        
            return ps1 + ps2
        else:    
            return self.PS_OneHalo(z, k, profile_1, Mmin_1, profile_2, Mmin_2) \
                 + self.PS_TwoHalo(z, k, profile_1, Mmin_1, profile_2, Mmin_2)
    
    def _integrand_iFT_3d_to_1d(self, P, k, R):
        """
        For an isotropic field, the 3-D Fourier transform can be simplified
        to a 1-D integral with this integrand.
        
        Parameters
        ----------
        P : 1-D array
            Power spectrum of the field as a function of k
        k : 1-D array
            Corresponding set of wavenumbers.
        R : int, float
            Scale (in real space) of interest.
            
        """
        return k**2 * P * np.sin(k * R) / k / R
        
    def _integrand_FT_3d_to_1d(self, cf, k, R): 
        return R**2 * cf * np.sin(k * R) / k / R
        
    def CorrelationFunction(self, z, R, k=None, Pofk=None, load=True):
        """
        Compute the correlation function of the matter power spectrum.
        
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        R : int, float, np.ndarray
            Scale(s) of interest
            
        """
        
        ##
        # Load from table
        ##
        if self.pf['hmf_load_ps'] and load:
            iz = np.argmin(np.abs(z - self.tab_z))
            assert abs(z - self.z[iz]) < 1e-2, \
                'Supplied redshift (%g) not in table!' % z
            assert np.allclose(dr, self.R_cr)
            return self.cf_dd[iz]
        
        ##
        # Compute from scratch
        ##
        
        # Has P(k) already been computed?
        if Pofk is not None:
            if k is None:
                k = self.tab_k
                assert len(Pofk) == len(self.tab_k), \
                    "Mismatch in shape between Pofk and k!"

        else:        
            k = self.tab_k
            Pofk = self.PowerSpectrum(z, self.tab_k)
        
        # Integrate over k        
        func = lambda R: self._integrand_iFT_3d_to_1d(Pofk, k, R)
        
        if type(R) in [int, float]:
            return np.trapz(func(R) * k, x=np.log(k)) / 2. / np.pi
        else:    
            return np.array([np.trapz(func(R) * k, x=np.log(k)) \
                for R in self.tab_R]) / 2. / np.pi
                
    @property            
    def tab_k(self):
        """
        k-vector constructed from mpowspec parameters.
        """
        if not hasattr(self, '_tab_k'):
            dlogk = self.pf['mpowspec_dlnk']
            kmi, kma = self.pf['mpowspec_lnk_min'], self.pf['mpowspec_lnk_max']
            logk = np.arange(kmi, kma+dlogk, dlogk)
            self._tab_k = np.exp(logk)
        
        return self._tab_k
        
    @tab_k.setter
    def tab_k(self, value):
        self._tab_k = value
    
    @property
    def tab_R(self):
        """
        R-vector constructed from mpowspec parameters.
        """
        if not hasattr(self, '_tab_R'):        
            dlogR = self.pf['mpowspec_dlnR']
            Rmi, Rma = self.pf['mpowspec_lnR_min'], self.pf['mpowspec_lnR_max']
            logR = np.arange(Rmi, Rma+dlogR, dlogR)
            self._tab_R = np.exp(logR)
            
        return self._tab_R
        
    @tab_R.setter
    def tab_R(self, value):
        self._tab_R = value
        
        print('Setting R attribute. Should verify it matches PS.')
            
    def __getattr__(self, name):
                
        if hasattr(HaloMassFunction, name):
            return HaloMassFunction.__dict__[name].__get__(self, HaloMassFunction)
        
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')
    
        if name not in self.__dict__.keys():
            self._load_hmf()
            
            if name not in self.__dict__.keys():
                self._load_ps()
    
        return self.__dict__[name]
    
    def _load_ps(self, suffix='npz'):
        """ Load table from HDF5 or binary. """
        
        fn = '%s/input/hmf/%s.%s' % (ARES, self.tab_prefix_ps(), suffix)
    
        if re.search('.hdf5', fn) or re.search('.h5', fn):
            f = h5py.File(fn, 'r')
            self.tab_z = f['tab_z'].value
            self.tab_R = f['tab_R'].value
            self.tab_k = f['tab_k'].value
            self.tab_ps_mm = f['tab_ps_mm'].value
            self.tab_cf_mm = f['tab_cf_mm'].value
            f.close()
        elif re.search('.npz', fn):
            f = np.load(fn)
            self.tab_z = f['tab_z']
            self.tab_R = f['tab_R']
            self.tab_k = f['tab_k']
            self.tab_ps_mm = f['tab_ps_mm']
            self.tab_cf_mm = f['tab_cf_mm']    
            f.close()                        
        elif re.search('.pkl', fn):
            f = open(fn, 'rb')
            self.tab_z = pickle.load(f)
            self.tab_R = pickle.load(f)
            self.tab_k = pickle.load(f)
            self.tab_ps_mm = pickle.load(f)
            self.tab_cf_mm = pickle.load(f)
            f.close()
        else:
            raise IOError('Unrecognized format for mps_table.')    

    def tab_prefix_ps(self, with_size=True):
        """
        What should we name this table?
        
        Convention:
        ps_FIT_logM_nM_logMmin_logMmax_z_nz_
        
        Read:
        halo mass function using FIT form of the mass function
        using nM mass points between logMmin and logMmax
        using nz redshift points between zmin and zmax
        
        """
        
        M1, M2 = self.pf['hmf_logMmin'], self.pf['hmf_logMmax']
        z1, z2 = self.pf['hmf_zmin'], self.pf['hmf_zmax']
        
        dlogk = self.pf['mpowspec_dlnk']
        kmi, kma = self.pf['mpowspec_lnk_min'], self.pf['mpowspec_lnk_max']
        #logk = np.arange(kmi, kma+dlogk, dlogk)
        #karr = np.exp(logk)
        
        dlogR = self.pf['mpowspec_dlnR']
        Rmi, Rma = self.pf['mpowspec_lnR_min'], self.pf['mpowspec_lnR_max']
        #logR = np.arange(np.log(Rmi), np.log(Rma)+dlogR, dlogR)
        #Rarr = np.exp(logR)
        
                
        if with_size:
            logMsize = (self.pf['hmf_logMmax'] - self.pf['hmf_logMmin']) \
                / self.pf['hmf_dlogM']                
            zsize = ((self.pf['hmf_zmax'] - self.pf['hmf_zmin']) \
                / self.pf['hmf_dz']) + 1

            assert logMsize % 1 == 0
            logMsize = int(logMsize)
            assert zsize % 1 == 0
            zsize = int(round(zsize, 1))
                
            # Should probably save NFW information etc. too
            return 'mps_%s_logM_%s_%i-%i_z_%s_%i-%i_lnR_%.1f-%.1f_dlnr_%.2f_lnk_%.1f-%.1f_dlnk_%.2f' \
                % (self.hmf_func, logMsize, M1, M2, zsize, z1, z2,
                   Rmi, Rma, dlogR, kmi, kma, dlogk)
        else:
            raise NotImplementedError('help')

    def TabulatePS(self, clobber=False, checkpoint=True):
        """
        Tabulate the matter power spectrum as a function of redshift and k.
        """
        
        pb = ProgressBar(len(self.tab_z), 'ps_dd')
        pb.start()

        # Lists to store any checkpoints that are found        
        _z = []
        _ps = []
        _cf = []
        if checkpoint:
            if (not os.path.exists('tmp')):
                os.mkdir('tmp')

            fn = 'tmp/{}.{}.pkl'.format(self.tab_prefix_ps(True), 
                str(rank).zfill(3))    
                
            if os.path.exists(fn) and (not clobber):
                                
                # Should delete if clobber == True?
                
                if rank == 0:
                    print("Checkpoints for this model found in tmp/.")
                    print("Re-run with clobber=True to overwrite.")
                
                f = open(fn, 'rb')
                while True:
                    
                    try:
                        tmp = pickle.load(f)
                    except EOFError:
                        break
                   
                    _z.append(tmp[0])
                    _ps.append(tmp[1])
                    _cf.append(tmp[2])
                
                if _z != []:
                    print "Processor {} loaded checkpoints for z={}-{}".format(rank, 
                        min(_z), max(_z))
            
            elif os.path.exists(fn):
                os.remove(fn)
                            
        self.tab_ps_mm = np.zeros((len(self.tab_z), len(self.tab_k)))
        self.tab_cf_mm = np.zeros((len(self.tab_z), len(self.tab_R)))
        for i, z in enumerate(self.tab_z):
        
            if i % size != rank:
                continue
            
            ##
            # Load checkpoint, if one exists.
            ##
            if z in _z:
                
                j = _z.index(z)
                self.tab_ps_mm[i] = _ps[j]
                self.tab_cf_mm[i] = _cf[j]

                pb.update(i)
                continue

            ##
            # Calculate from scratch
            ##                

            # Must interpolate back to fine grid (uniformly sampled 
            # real-space scales) to do FFT and obtain correlation function
            self.tab_ps_mm[i] = self.PowerSpectrum(z, self.tab_k)
                 
            # Once we have correlation function, degrade it to coarse grid.
            self.tab_cf_mm[i] = self.CorrelationFunction(z, self.tab_R, 
                self.tab_ps_mm[i])
            
            pb.update(i)
                        
            if not checkpoint:
                continue
                                
            with open(fn, 'ab') as f:
                pickle.dump((z, self.tab_ps_mm[i], self.tab_cf_mm[i]), f)
                #print("Processor {} wrote checkpoint for z={}".format(rank, z))
            
        pb.finish()

        # Collect results!
        if size > 1:
            tmp1 = np.zeros_like(self.tab_ps_mm)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_ps_mm, tmp1)
            self.tab_ps_mm = tmp1
        
            tmp2 = np.zeros_like(self.tab_cf_mm)
            nothing = MPI.COMM_WORLD.Allreduce(self.tab_cf_mm, tmp2)
            self.tab_cf_mm = tmp2
            
        # Done!    

    def SavePS(self, fn=None, clobber=True, destination=None, format='npz',
        checkpoint=True):
        """
        Save matter power spectrum table to HDF5 or binary (via pickle).
    
        Parameters
        ----------
        fn : str (optional)
            Name of file to save results to. If None, will use 
            self.tab_prefix_ps and value of format parameter to make one up.
        clobber : bool
            Overwrite pre-existing files of the same name?
        destination : str
            Path to directory (other than CWD) to save table.
        format : str
            Format of output. Can be 'hdf5' or 'pkl'
    
        """
        
        if destination is None:
            destination = '.'
        
        # Determine filename
        if fn is None:
            fn = '%s/%s.%s' % (destination, self.tab_prefix_ps(True), format)
        else:
            if format not in fn:
                print "Suffix of provided filename does not match chosen format."
                print "Will go with format indicated by filename suffix."
        
        if os.path.exists(fn):
            if clobber:
                os.system('rm -f %s' % fn)
            else:
                raise IOError('File %s exists! Set clobber=True or remove manually.' % fn) 

        # Do this first! (Otherwise parallel runs will be garbage)
        self.TabulatePS(clobber=clobber, checkpoint=checkpoint)

        if rank > 0:
            return
    
        self._write_ps(fn, clobber, format)
    
    def _write_ps(self, fn, clobber, format=format):
        
        try:
            import hmf
            hmf_v = hmf.__version__
        except AttributeError:
            hmf_v = 'unknown'
            
        if os.path.exists(fn):
            if clobber:
                os.system('rm -f %s' % fn)
            else:
                raise IOError('File %s exists! Set clobber=True or remove manually.' % fn) 
    
        if format == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('tab_z', data=self.tab_z)
            f.create_dataset('tab_R', data=self.tab_R)
            f.create_dataset('tab_k', data=self.tab_k)
            f.create_dataset('tab_ps_mm', data=self.tab_ps_mm)
            f.create_dataset('tab_cf_mm', data=self.tab_cf_mm)
  
            f.close()
    
        elif format == 'npz':
            data = {'tab_z': self.tab_z, 
                    'tab_R': self.tab_R,
                    'tab_k': self.tab_k,
                    'tab_ps_mm': self.tab_ps_mm,
                    'tab_cf_mm': self.tab_cf_mm,
                    'hmf-version': hmf_v}
                    
            try:
                np.savez(fn, **data)
            except IOError:
                f = open(fn, 'wb')
                np.savez(f, **data)
    
        # Otherwise, pickle it!    
        else:   
            f = open(fn, 'wb')
            pickle.dump(self.tab_z, f)
            pickle.dump(self.tab_R, f)
            pickle.dump(self.tab_k, f)
            pickle.dump(self.tab_ps_mm, f)
            pickle.dump(self.tab_cf_mm, f)
            pickle.dump(dict(('hmf-version', hmf_v)))
            f.close()
    
        print 'Wrote %s.' % fn
        return
    
    