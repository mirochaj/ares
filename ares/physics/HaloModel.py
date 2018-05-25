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

        iz = np.argmin(np.abs(z - self.z))

        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_1 is None:
            profile_1 = self.u_nfw
        
        prof1 = np.abs(map(lambda M: profile_1(k, M, z), self.M))
            
        if profile_2 is None:
            prof2 = prof1
        else:
            prof2 = np.abs(map(lambda M: profile_2(k, M, z), self.M))

        dndlnm = self.dndm[iz,:] * self.M
        rho_bar = self.cosm.rho_m_z0 * rho_cgs

        if Mmin_1 is None:
            fcoll_1 = 1.
            iM_1 = 0
        else:
            fcoll_1 = self.fcoll_Tmin[iz]
            iM_1 = np.argmin(np.abs(Mmin_1 - self.M))
        
        if Mmin_2 is None:
            fcoll_2 = 1.
            iM_2 = 0
        else:
            fcoll_2 = self.fcoll_Tmin[iz]
            iM_2 = np.argmin(np.abs(Mmin_2 - self.M))
        
        iM = max(iM_1, iM_2)
        
        integrand = dndlnm * (self.M / rho_bar / fcoll_1) \
            * (self.M / rho_bar / fcoll_2) * prof1 * prof2 

        result = np.trapz(integrand[iM:], x=self.lnM[iM:])

        return result

    def PS_TwoHalo(self, z, k, profile_1=None, Mmin_1=None, profile_2=None, 
        Mmin_2=None):
        """
        Compute the two halo term of the halo model for given input profile.
        
        .. note :: Assumption of linearity?
        
        Parameters
        ----------
        
        """
        
        iz = np.argmin(np.abs(z - self.z))

        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_1 is None:
            profile_1 = self.u_nfw
        
        prof1 = np.abs(map(lambda M: profile_1(k, M, z), self.M))
            
        if profile_2 is None:
            prof2 = prof1
        else:
            prof2 = np.abs(map(lambda M: profile_2(k, M, z), self.M))
                        
        # Short-cuts
        dndlnm = self.dndm[iz,:] * self.M
        bias = self.bias_of_M(z)
        #rho_bar = self.mgtm[iz,0]
        rho_bar = self.cosm.rho_m_z0 * rho_cgs

        if Mmin_1 is None:
            fcoll_1 = 1.#self.mgtm[iz,0] / rho_bar
            iM_1 = 0
            
            # Small halo correction.
            # Make use of Cooray & Sheth Eq. 71
            _integrand = dndlnm * (self.M / rho_bar) * bias
            correction_1 = 1. - np.trapz(_integrand, x=self.lnM)
        else:
            fcoll_1 = self.fcoll_Tmin[iz]
            iM_1 = np.argmin(np.abs(Mmin_1 - self.M))
            correction_1 = 0.0
                    
        if Mmin_2 is None:
            fcoll_2 = 1.#self.mgtm[iz,0] / rho_bar
            iM_2 = 0
            _integrand = dndlnm * (self.M / rho_bar) * bias
            correction_2 = 1. - np.trapz(_integrand, x=self.lnM)
        else:
            fcoll_2 = self.fcoll_Tmin[iz]
            iM_2 = np.argmin(np.abs(Mmin_2 - self.M))
            correction_2 = 0.0

        # Compute two-halo integral with profile in there
        integrand1 = dndlnm * (self.M / rho_bar / fcoll_1) * prof1 * bias
        integral1 = np.trapz(integrand1[iM_1:], x=self.lnM[iM_1:]) \
                  + correction_1

        if profile_2 is not None:
            integrand2 = dndlnm * (self.M / rho_bar / fcoll_2) * prof2 * bias
            integral2 = np.trapz(integrand2[iM_2:], x=self.lnM[iM_2:]) \
                      + correction_2
        else:
            integral2 = integral1

        return integral1 * integral2 * float(self.psCDM(z, k))

    def PowerSpectrum(self, z, k, profile_1=None, Mmin_1=None, profile_2=None, 
        Mmin_2=None, exact_z=True):
        
        # Tabulation only implemented for density PS at the moment.
        if self.pf['hmf_load_ps'] and (profile_1 is None):
            iz = np.argmin(np.abs(z - self.z))
            if exact_z:
                assert abs(z - self.z[iz]) < 1e-2, \
                    'Supplied redshift (%g) not in table!' % z
            if len(k) == len(self.k_cr_pos):
                if np.allclose(k, self.k_cr_pos):
                    return self.ps_dd[iz]
                
            return np.interp(np.log(k), np.log(self.k_cr_pos), self.ps_dd[iz])
                    
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
    
    def CorrelationFunction(self, z, dr):
        if self.pf['hmf_load_ps']:
            iz = np.argmin(np.abs(z - self.z))
            assert abs(z - self.z[iz]) < 1e-2, \
                'Supplied redshift (%g) not in table!' % z
            assert np.allclose(dr, self.R_cr)
            return self.cf_dd[iz]
                
        raise NotImplemented('help')
        
    def __getattr__(self, name):
                
        if hasattr(HaloMassFunction, name):
            return HaloMassFunction.__dict__[name].__get__(self, HaloMassFunction)
        
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')
    
        if name not in self.__dict__.keys():
            self.load_hmf()
            
            if name not in self.__dict__.keys():
                self.load_ps()
    
        return self.__dict__[name]
    
    def load_ps(self, suffix='npz'):
        """ Load table from HDF5 or binary. """
        
        fn = '%s/input/hmf/%s.%s' % (ARES, self.table_prefix_ps(), suffix)
    
        if re.search('.hdf5', fn) or re.search('.h5', fn):
            f = h5py.File(fn, 'r')
            self.z = f['z'].value
            self.R_cr = f['r'].value
            self.k_cr_pos = f['k'].value
            self.ps_dd = f['ps'].value
            self.cf_dd = f['cf'].value
            f.close()
        elif re.search('.npz', fn):
            f = np.load(fn)
            self.z = f['z']
            self.R_cr = f['r']
            self.k_cr_pos = f['k']
            self.ps_dd = f['ps']
            self.cf_dd = f['cf']    
            f.close()                        
        elif re.search('.pkl', fn):
            f = open(fn, 'rb')
            self.z = pickle.load(f)
            self.R_cr = pickle.load(f)
            self.k_cr_pos = pickle.load(f)
            self.ps_dd = pickle.load(f)
            self.cf_dd = pickle.load(f)    
            f.close()
        else:
            raise IOError('Unrecognized format for mps_table.')    

    def table_prefix_ps(self, with_size=True):
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
        
        rarr = self.pf['fft_scales']
                
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
            return 'mps_%s_logM_%s_%i-%i_z_%s_%i-%i_logR_%.1f-%.1f_dlogr_%.2f_dlogk_%.2f' \
                % (self.hmf_func, logMsize, M1, M2, zsize, z1, z2,
                   np.log10(min(rarr)), np.log10(max(rarr)), 
                   self.pf['mpowspec_dlogr'],
                   self.pf['mpowspec_dlogk'])
        else:
            raise NotImplementedError('help')
            #return 'ps_%s_logM_*_%i-%i_z_*_%i-%i' \
            #    % (self.hmf_func, M1, M2, z1, z2)
    
    def tabulate_ps(self, clobber=False, checkpoint=True):
        """
        Tabulate the (density) power spectrum.
        """
        
        pb = ProgressBar(len(self.z), 'ps_dd')
        pb.start()
        
        step = np.diff(self.pf['fft_scales'])[0]
        R = self.pf['fft_scales']
        logR = np.log(R)
        
        # Find corresponding wavenumbers
        k = np.fft.fftfreq(R.size, step)
        absk = np.abs(k)
        logk = np.log(absk)
        
        # Zero-frequency shifted to center so values are monotonically rising
        k_sh = np.fft.fftshift(k)
        absk_sh = np.abs(k_sh)
        logk_sh = np.log(absk_sh)

        # The frequency array has half the number of unique elements (plus 1)
        # as the scales array.

        # Set up coarse grid for evaluation of halo model
        k_mi, k_ma = absk_sh[absk_sh>0].min(), absk_sh.max()
        dlogk = self.pf['mpowspec_dlogk']
        logk_cr_pos = np.arange(np.log(k_mi), np.log(k_ma)+dlogk, dlogk)
        self.k_cr_pos = np.exp(logk_cr_pos)

        # Setup degraded scales array
        r_mi, r_ma = R.min(), R.max()
        dlogr = self.pf['mpowspec_dlogr']
        logR_cr = np.arange(np.log(r_mi), np.log(r_ma)+dlogr, dlogr)
        self.R_cr = R_cr = np.exp(logR_cr)

        ct = 0
        
        _z = []
        _ps = []
        _cf = []
        if checkpoint:
            if (not os.path.exists('tmp')):
                os.mkdir('tmp')

            fn = 'tmp/{}.{}.pkl'.format(self.table_prefix_ps(True), 
                str(rank).zfill(3))    
                
            if os.path.exists(fn) and (not clobber):
                                
                # Should delete if clobber == True?
                
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
            
        # Must tabulate onto coarse grid otherwise we'll run out of memory.
        self.ps_dd = np.zeros((len(self.z), len(self.k_cr_pos)))
        self.cf_dd = np.zeros((len(self.z), len(self.R_cr)))
        for i, z in enumerate(self.z):
        
            if i % size != rank:
                continue
            
            # Load checkpoint
            if z in _z:
                
                j = _z.index(z)
                self.ps_dd[i] = _ps[j]
                self.cf_dd[i] = _cf[j]
                
                #print rank, z, _z[j]
                            
                pb.update(i)
                continue
                
            ##
            # Calculate from scratch
            ##                
                            
            ps_k_cr_pos = self.PowerSpectrum(z, self.k_cr_pos)

            # Must interpolate back to fine grid (uniformly sampled 
            # real-space scales) to do FFT and obtain correlation function
            self.ps_dd[i] = ps_k_cr_pos.copy()
            
            # Recall that logk contains +/- frequencies so this power spectrum
            # is (properly) mirrored about zero-frequency 
            ps_all_k = np.exp(np.interp(logk, logk_cr_pos, 
                np.log(ps_k_cr_pos)))
                
            cf_R = 2 * np.fft.ifft(ps_all_k)

            # Once we have correlation function, degrade it to coarse grid.
            self.cf_dd[i] = np.interp(logR_cr, logR, cf_R.real)
            
            pb.update(i)
                        
            if not checkpoint:
                continue
                                
            with open(fn, 'ab') as f:
                pickle.dump((z, self.ps_dd[i], self.cf_dd[i]), f)
                #print "Processor {} wrote checkpoint for z={}".format(rank, z)
            
        pb.finish()

        # Collect results!
        if size > 1:
            tmp1 = np.zeros_like(self.ps_dd)
            nothing = MPI.COMM_WORLD.Allreduce(self.ps_dd, tmp1)
            self.ps_dd = tmp1
        
            tmp2 = np.zeros_like(self.cf_dd)
            nothing = MPI.COMM_WORLD.Allreduce(self.cf_dd, tmp2)
            self.cf_dd = tmp2

    def save_ps(self, fn=None, clobber=True, destination=None, format='npz',
        checkpoint=True):
        """
        Save matter power spectrum table to HDF5 or binary (via pickle).
    
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
        
        if destination is None:
            destination = '.'
        
        # Determine filename
        if fn is None:
            fn = '%s/%s.%s' % (destination, self.table_prefix_ps(True), format)
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
        self.tabulate_ps(clobber=clobber, checkpoint=checkpoint)
    
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
            f.create_dataset('z', data=self.z)
            f.create_dataset('r', data=self.R_cr)
            f.create_dataset('k', data=self.k_cr_pos)
            f.create_dataset('ps', data=self.ps_dd)
            f.create_dataset('cf', data=self.cf_dd)
  
            f.close()
    
        elif format == 'npz':
            data = {'z': self.z, 
                    'r': self.R_cr,
                    'k': self.k_cr_pos,
                    'ps': self.ps_dd,
                    'cf': self.cf_dd,
                    'hmf-version': hmf_v}
            np.savez(fn, **data)
    
        # Otherwise, pickle it!    
        else:   
            f = open(fn, 'wb')
            pickle.dump(self.z, f)
            pickle.dump(self.r, f)
            pickle.dump(self.k_cr_pos, f)
            pickle.dump(self.ps_dd, f)
            pickle.dump(self.cf_dd, f)
            pickle.dump(dict(('hmf-version', hmf_v)))
            f.close()
    
        print 'Wrote %s.' % fn
        return
    
    