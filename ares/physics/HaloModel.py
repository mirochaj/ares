# Thanks, Jason Sun, for most of this!

import os, re
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
from ..util.ProgressBar import ProgressBar
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
        Normalized Fourier Transform of rho.
        
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
        return (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) \
            + np.cos(K) * (ac - bc)) / (np.log(1 + c) - c / (1 + c))
        
    def PS_OneHalo(self, z, k, profile_ft=None):
        """
        Compute the one halo term of the halo model for given input profile.
        """
        
        iz = np.argmin(np.abs(z - self.z))
                        
        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_ft is None:
            profile_ft = self.u_nfw

        prof = np.abs(map(lambda M: profile_ft(k, M, z), self.M))
                        
        #if mass_dependence is not None:
        #    prof *= mass_dependence(Mh=self.M, z=z)                
                        
        dndlnm = self.dndm[iz,:] * self.M
        rho_bar = self.mgtm[iz,0]
                        
        integrand = dndlnm * (self.M / rho_bar)**2 * prof**2
         
        result = np.trapz(integrand, x=self.lnM) 
        
        return result
        
    def PS_TwoHalo(self, z, k, profile_ft=None):
        """
        Compute the two halo term of the halo model for given input profile.
        
        .. note :: Assumption of linearity?
        
        Parameters
        ----------
        
        """
        
        iz = np.argmin(np.abs(z - self.z))

        # Can plug-in any profile, but will default to dark matter halo profile
        if profile_ft is None:
            profile_ft = self.u_nfw

        prof = np.abs(map(lambda M: profile_ft(k, M, z), self.M))
        
        #if mass_dependence is not None:
        #    Mterm = mass_dependence(Mh=self.M, z=z) 
        #    norm = np.trapz(Mterm, x=self.M)
        #    
        #    prof *= Mterm / norm
                
        # Short-cuts
        dndlnm = self.dndm[iz,:] * self.M
        bias = self.bias_of_M(z)
        rho_bar = self.mgtm[iz,0] # Should be equal to cosmic mean density * fcoll
        
        # Small halo correction.
        # Make use of Cooray & Sheth Eq. 71
        _integrand = dndlnm * (self.M / rho_bar) * bias
        correction = 1. - np.trapz(_integrand, x=self.lnM)
        
        # Compute two-halo integral with profile in there
        integrand = dndlnm * (self.M / rho_bar) * \
            prof * bias
            
        return (np.trapz(integrand, x=self.lnM) + correction)**2 \
            * float(self.psCDM(z, k))

    def PowerSpectrum(self, z, k, profile_ft=None):
        
        if self.pf['hmf_load_ps']:
            iz = np.argmin(np.abs(z - self.z))
            assert abs(z - self.z[iz]) < 1e-2, \
                'Supplied redshift (%g) not in table!' % z
            assert np.allclose(k, self.k_pos)
            return self.ps_dd[iz]
                    
        if type(k) == np.ndarray:
            f1 = lambda kk: self.PS_OneHalo(z, kk, profile_ft=profile_ft)
            f2 = lambda kk: self.PS_TwoHalo(z, kk, profile_ft=profile_ft)
            ps1 = np.array(map(f1, k))
            ps2 = np.array(map(f2, k))
            return ps1 + ps2
        else:    
            return self.PS_OneHalo(z, k, profile_ft=profile_ft) \
                 + self.PS_TwoHalo(z, k, profile_ft=profile_ft)
    
    def CorrelationFunction(self, z, dr):
        if self.pf['hmf_load_ps']:
            iz = np.argmin(np.abs(z - self.z))
            assert abs(z - self.z[iz]) < 1e-2, \
                'Supplied redshift (%g) not in table!' % z
            assert np.allclose(dr, self.dr_coarse)
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
            self.dr_coarse = f['r'].value
            self.k_pos = f['k'].value
            self.ps_dd = f['ps'].value
            self.cf_dd = f['cf'].value
            f.close()
        elif re.search('.npz', fn):
            f = np.load(fn)
            self.z = f['z']
            self.dr_coarse = f['r']
            self.k_pos = f['k']
            self.ps_dd = f['ps']
            self.cf_dd = f['cf']    
            f.close()                        
        elif re.search('.pkl', fn):
            f = open(fn, 'rb')
            self.z = pickle.load(f)
            self.dr_coarse = pickle.load(f)
            self.k_pos = pickle.load(f)
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
            return 'mps_%s_logM_%s_%i-%i_z_%s_%i-%i_logR_%i-%i_dlogr_%.2f_dlogk_%.2f' \
                % (self.hmf_func, logMsize, M1, M2, zsize, z1, z2,
                   np.log10(min(rarr)), np.log10(max(rarr)), 
                   self.pf['powspec_dlogr'],
                   self.pf['powspec_dlogk'])
        else:
            raise NotImplementedError('help')
            #return 'ps_%s_logM_*_%i-%i_z_*_%i-%i' \
            #    % (self.hmf_func, M1, M2, z1, z2)
    
    def tabulate_ps(self):
        """
        Tabulate the (density) power spectrum.
        """
        
        pb = ProgressBar(len(self.z), 'ps_dd')
        pb.start()
        
        step = np.diff(self.pf['fft_scales'])[0]
        dr = self.pf['fft_scales']
        k = np.fft.fftfreq(dr.size, step)
        
        k_mi, k_ma = k.min(), k.max()
        dlogk = self.pf['powspec_dlogk']
        k_pos = 10**np.arange(np.log10(k[k>0].min()), np.log10(k_ma)+dlogk, dlogk)
        self.k_coarse = np.concatenate((-1 * k_pos[-1::-1], [0], k_pos))
        self.k_pos = k_pos
        
        r_mi, r_ma = dr.min(), dr.max()
        dlogr = self.pf['powspec_dlogr']
        self.dr_coarse = 10**np.arange(np.log10(r_mi), np.log10(r_ma)+dlogr, dlogr)
        
        # Must tabulate onto coarse grid otherwise we'll run out of memory.
        self.ps_dd = np.zeros((len(self.z), len(self.k_pos)))
        self.cf_dd = np.zeros((len(self.z), len(self.dr_coarse)))
        for i, z in enumerate(self.z):
        
            if i % size != rank:
                continue

            #ps_posk = np.zeros_like(k_pos)
            #for j, _k in enumerate(k_pos):
            ps_posk = self.PowerSpectrum(z, k_pos, profile_ft=None)

            # Must interpolate to uniformly (in real space) sampled
            # grid points to do inverse FFT

            #ps_fold = np.concatenate((ps_posk[-1::-1], [0], ps_posk))
            #ps_fold = np.concatenate(([0], ps_posk, ps_posk[-1::-1]))
            #ps_dd = np.interp(self.k, self.k_coarse, ps_fold)
            ps_dd = np.interp(np.abs(k), self.k_pos, ps_posk)
            #ps_dd = self.field.halos.PowerSpectrum(z, np.abs(self.k))
            self.ps_dd[i] = ps_posk.copy()

            cf_dd = np.fft.ifft(ps_dd)

            # Interpolate onto coarser grid
            self.cf_dd[i] = np.interp(self.dr_coarse, dr, cf_dd.real)

            pb.update(i)

        pb.finish()

        # Collect results!
        if size > 1:
            tmp1 = np.zeros_like(self.ps_dd)
            nothing = MPI.COMM_WORLD.Allreduce(self.ps_dd, tmp1)
            self.ps_dd = tmp1
        
            tmp2 = np.zeros_like(self.cf_dd)
            nothing = MPI.COMM_WORLD.Allreduce(self.cf_dd, tmp2)
            self.cf_dd = tmp2

    def save_ps(self, fn=None, clobber=True, destination=None, format='hdf5'):
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
    
        try:
            import hmf
            hmf_v = hmf.__version__
        except AttributeError:
            hmf_v = 'unknown'
    
        # Do this first! (Otherwise parallel runs will be garbage)
        self.tabulate_ps()
    
        if rank > 0:
            return
    
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
    
        if format == 'hdf5':
            f = h5py.File(fn, 'w')
            f.create_dataset('z', data=self.z)
            f.create_dataset('r', data=self.dr_coarse)
            f.create_dataset('k', data=self.k_pos)
            f.create_dataset('ps', data=self.ps_dd)
            f.create_dataset('cf', data=self.cf_dd)
  
            f.close()
    
        elif format == 'npz':
            data = {'z': self.z, 
                    'r': self.dr_coarse,
                    'k': self.k_pos,
                    'ps': self.ps_dd,
                    'cf': self.cf_dd,
                    'hmf-version': hmf_v}
            np.savez(fn, **data)
    
        # Otherwise, pickle it!    
        else:   
            f = open(fn, 'wb')
            pickle.dump(self.z, f)
            pickle.dump(self.r, f)
            pickle.dump(self.k_pos, f)
            pickle.dump(self.ps, f)
            pickle.dump(self.cf, f)
            pickle.dump(dict(('hmf-version', hmf_v)))
            f.close()
    
        print 'Wrote %s.' % fn
        return
    
    