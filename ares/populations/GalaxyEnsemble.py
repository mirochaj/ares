"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description: 

"""

import os
import gc
import time
import pickle
import numpy as np
from ..util.Math import smooth
from ..util import ProgressBar
from .Halo import HaloPopulation
from .GalaxyCohort import GalaxyCohort
from scipy.integrate import quad, cumtrapz
from ..analysis.BlobFactory import BlobFactory
from ..util.Stats import bin_e2c, bin_c2e, bin_samples
from ..sources.SynthesisModelSBS import SynthesisModelSBS
from scipy.interpolate import RectBivariateSpline, interp1d
from ..physics.Constants import rhodot_cgs, s_per_yr, s_per_myr, \
    g_per_msun, c, Lsun, cm_per_kpc

try:
    import h5py
except ImportError:
    pass
       
pars_affect_mars = ["pop_MAR", "pop_MAR_interp", "pop_MAR_corr"]
pars_affect_sfhs = ["pop_scatter_sfr", "pop_scatter_sfe", "pop_scatter_mar"]
pars_affect_sfhs.extend(["pop_update_dt","pop_thin_hist"])

class GalaxyEnsemble(HaloPopulation,BlobFactory):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # May not actually need this...
        HaloPopulation.__init__(self, **kwargs)
        
    def __dict__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        
        raise NotImplemented('help!')
        
    #@property
    #def dust(self):
    #    if not hasattr(self, '_dust'):
    #        self._dust = DustCorrection(**self.pf)
    #    return self._dust    
    
    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            h = self._gen_halo_histories()
        return self._tab_z    
            
    @tab_z.setter
    def tab_z(self, value):
        self._tab_z = value
    
    @property
    def tab_t(self):
        if not hasattr(self, '_tab_t'):
            # Array of times corresponding to all z' > z [years]
            self._tab_t = self.cosm.t_of_z(self.tab_z) / s_per_yr    
        return self._tab_t
        
    @property
    def tab_dz(self):
        if not hasattr(self, '_tab_dz'):
            dz = np.diff(self.tab_z)
            if np.allclose(np.diff(dz), 0):
                dz = dz[0]
            self._tab_dz = dz    
                
        return self._tab_dz
        
    def run(self):
        return
        
    def SFRD(self, z):
        """
        Will convert to internal cgs units.
        """
        
        iz = np.argmin(np.abs(z - self.histories['z']))
        sfr = self.histories['SFR'][:,iz]
        w = self.histories['nh'][:,iz]
        
        # Really this is the number of galaxies that formed in a given
        # differential redshift slice.
        
        #Mh = self.histories['Mh'][:,iz]
        
        #dw = np.diff(_w) / np.diff(Mh)
        
        # The 'bins' in the first dimension have some width...
        
        return np.sum(sfr * w) / rhodot_cgs
        #return np.trapz(sfr[0:-1] * dw, dx=np.diff(Mh)) / rhodot_cgs

    def tile(self, arr, thin, renorm=False):
        """
        Expand an array to `thin` times its size. Group elements such that
        neighboring bundles of `thin` elements are objects that formed
        at the same redshift.
        """
        if arr is None:
            return None
        
        if thin in [0, 1]:
            return arr.copy()

        assert thin % 1 == 0

        # First dimension: formation redshifts
        # Second dimension: observed redshifts / times        
        #new = np.tile(arr, (int(thin), 1))
        new = np.repeat(arr, int(thin), axis=0)
        #N = arr.shape[0]
        #_new = np.tile(arr, int(thin) * N)
        #new = _new.reshape(N * int(thin), N)
        
        if renorm:
            return new / float(thin)
        else:
            return new
    
    def noise_normal(self, arr, sigma):
        noise = np.random.normal(scale=sigma, size=arr.size)
        return np.reshape(noise, arr.shape)
    
    def noise_lognormal(self, arr, sigma):
        lognoise = np.random.normal(scale=sigma, size=arr.size)        
        #noise = 10**(np.log10(arr) + np.reshape(lognoise, arr.shape)) - arr
        noise = np.power(10, np.log10(arr) + np.reshape(lognoise, arr.shape)) \
              - arr
        return noise
            
    @property    
    def tab_scatter_mar(self):
        if not hasattr(self, '_tab_scatter_mar'):            
            self._tab_scatter_mar = np.random.normal(scale=sigma, 
                size=np.product(self.tab_shape))
        return self._tab_scatter_mar
    
    @tab_scatter_mar.setter
    def tab_scatter_mar(self, value):
        assert value.shape == self.tab_shape
        self._tab_scatter_mar = value
        
    @property
    def tab_shape(self):
        if not hasattr(self, '_tab_shape'):
            raise AttributeError('help')
        return self._tab_shape
    
    @tab_shape.setter
    def tab_shape(self, value):
        self._tab_shape = value
        
    @property
    def _cache_halos(self):
        if not hasattr(self, '_cache_halos_'):
            self._cache_halos_ = self._gen_halo_histories()
        return self._cache_halos_
        
    @_cache_halos.setter
    def _cache_halos(self, value):
        self._cache_halos_ = value
            
    def _gen_halo_histories(self):
        """
        From a set of smooth halo assembly histories, build a bigger set
        of histories by thinning, and (optionally) adding scatter to the MAR. 
        """            
        
        if hasattr(self, '_cache_halos_'):
            return self._cache_halos

        raw = self.load()

        thin = self.pf['pop_thin_hist']

        sigma_mar = self.pf['pop_scatter_mar']
        sigma_env = self.pf['pop_scatter_env']
                                        
        # Just read in histories in this case.
        if raw is None:
            print('Running halo trajectories...')
            zall, raw = self.guide.Trajectories()
            print('Done with halo trajectories.')
        else:
            zall = raw['z']

        mar_raw = raw['MAR']
        nh_raw = raw['nh']
        Mh_raw = raw['Mh']

        # Could optionally thin out the bins to allow more diversity.
        if thin > 0:
            # Doesn't really make sense to do this unless we're
            # adding some sort of stochastic effects.
        
            # Remember: first dimension is the SFH identity.
            nh = self.tile(nh_raw, thin, True)
            Mh = self.tile(Mh_raw, thin)
        else:
            nh = nh_raw.copy()
            Mh = Mh_raw.copy()
        
        self.tab_shape = Mh.shape
        
        ##
        # Allow scatter in things
        ##            

        # Two potential kinds of scatter in MAR    
        mar = self.tile(mar_raw, thin)
        if sigma_env > 0:
            mar *= (1. + self.noise_normal(mar, sigma_env))

        if sigma_mar > 0:
            noise = self.noise_lognormal(mar, sigma_mar)
            mar += noise
            # Normalize by mean of log-normal to preserve mean MAR?
            mar /= np.exp(0.5 * sigma_mar**2)
            del noise

        # SFR = (zform, time (but really redshift))
        # So, halo identity is wrapped up in axis=0
        # In Cohort, formation time defines initial mass and trajectory (in full)
        #z2d = np.array([zall] * nh.shape[0])
        histories = {'Mh': Mh, 'MAR': mar, 'nh': nh}

        # Add in formation redshifts to match shape (useful after thinning)
        histories['zthin'] = self.tile(zall, thin)

        histories['z'] = zall
        
        if self.pf['conserve_memory']:
            dtype = np.float32
        else:
            dtype = np.float64
        
        histories['t'] = np.array(map(self.cosm.t_of_z, zall)) \
            / s_per_myr
                            
        if self.pf['pop_dust_yield'] > 0:
            r = np.reshape(np.random.rand(Mh.size), Mh.shape)
            if self.pf['conserve_memory']:
                histories['rand'] = r.astype(np.float32)
            else:
                histories['rand'] = r
        else:
            pass

        if 'SFR' in raw:
            #assert sigma_mar == sigma_env == 0
            histories['SFR'] = raw['SFR']

        if 'Z' in raw:
            histories['Z'] = raw['Z']

        self.tab_z = zall
        #self._cache_halos = histories
        
        del raw
        gc.collect()

        return histories
        
    def get_timestamps(self, zform):
        """
        For a halo forming at z=zform, return series of time and redshift
        steps based on halo dynamical time or other.
        """
        
        t0 = self.cosm.t_of_z(zform) / s_per_yr
        tf = self.cosm.t_of_z(0) / s_per_yr
        
        if self.pf['pop_update_dt'] == 'dynamical':
            t = t0
            steps_t = [t0]           # time since Big Bang
            steps_z = [zform]
            while t < tf:
                z = self.cosm.z_of_t(t * s_per_yr)
                _tdyn = self.halos.DynamicalTime(z, 1e10) / s_per_yr
            
                if t + _tdyn > tf:
                    steps_t.append(self.cosm.t_of_z(0.) / s_per_yr)
                    steps_z.append(0.)
                    break
            
                steps_t.append(t+_tdyn)
                steps_z.append(self.cosm.z_of_t((t+_tdyn) * s_per_yr))
                t += _tdyn
                
            steps_t = np.array(steps_t)
            steps_z = np.array(steps_z)
                
        else:
            dt = self.pf['pop_update_dt'] * 1e6     
            steps_t = np.arange(t0, tf+dt, dt)
            
            # Correct last timestep to make sure final timestamp == tf
            if steps_t[-1] > tf:
                steps_t[-1] = tf
            
            steps_z = np.array(map(self.cosm.z_of_t, steps_t * s_per_yr))
            
        return steps_t, steps_z
        
    @property
    def histories(self):
        if not hasattr(self, '_histories'):
            self._histories = self.RunSAM()
        return self._histories
        
    @histories.setter
    def histories(self, value):
        self._histories = value
        
    def Trajectories(self):
        return self.RunSAM()
    
    def RunSAM(self):
        """
        Run models. If deterministic, will just return pre-determined
        histories. Otherwise, will do some time integration.
        """
                
        return self._gen_galaxy_histories()
    
    @property
    def guide(self):
        if not hasattr(self, '_guide'):
            if self.pf['pop_guide_pop'] is not None:
                self._guide = self.pf['pop_guide_pop']
            else:
                tmp = self.pf.copy()
                tmp['pop_ssp'] = False
                self._guide = GalaxyCohort(**tmp)
        
        return self._guide
         
    def cmf(self, M):
        # Allow ParameterizedQuantity here
        pass
            
    @property
    def tab_cmf(self):
        if not hasattr(self, '_tab_cmf'):
            pass

    @property
    def _norm(self):
        if not hasattr(self, '_norm_'):
            mf = lambda logM: self.ClusterMF(10**logM)
            self._norm_ = quad(lambda logM: mf(logM) * 10**logM, -3, 10., 
                limit=500)[0]
        return self._norm_
        
    def ClusterMF(self, M, beta=-2, Mmin=50.):
        return (M / Mmin)**beta * np.exp(-Mmin / M)
        
    @property
    def tab_Mcl(self):
        if not hasattr(self, '_tab_Mcl'):
            self._tab_Mcl = np.logspace(-1., 8, 10000)
        return self._tab_Mcl
        
    @tab_Mcl.setter
    def tab_Mcl(self, value):
        self._tab_Mcl = value
    
    @property
    def tab_cdf(self):
        if not hasattr(self, '_tab_cdf'):
            mf = lambda logM: self.ClusterMF(10**logM)
            f_cdf = lambda M: quad(lambda logM: mf(logM) * 10**logM, -3, np.log10(M), 
                limit=500)[0] / self._norm
            self._tab_cdf = np.array(map(f_cdf, self.tab_Mcl))
            
        return self._tab_cdf    
    
    @tab_cdf.setter
    def tab_cdf(self, value):
        assert len(value) == len(self.tab_Mcl)
        self._tab_cdf = value
            
    def ClusterCDF(self):
        if not hasattr(self, '_cdf_cl'):            
            self._cdf_cl = lambda MM: np.interp(MM, self.tab_Mcl, self.tab_cdf)
        
        return self._cdf_cl
        
    @property
    def Mcl(self):
        if not hasattr(self, '_Mcl'):
            mf = lambda logM: self.ClusterMF(10**logM)
            self._Mcl = quad(lambda logM: mf(logM) * (10**logM)**2, -3, 10., 
                limit=500)[0] / self._norm
                
        return self._Mcl
        
    @property
    def tab_imf_me(self):
        if not hasattr(self, '_tab_imf_me'):
            self._tab_imf_me = 10**bin_c2e(self.src.pf['source_imf_bins'])
        return self._tab_imf_me
    
    @property
    def tab_imf_mc(self):
        if not hasattr(self, '_tab_imf_mc'):
            self._tab_imf_mc = 10**self.src.pf['source_imf_bins']
        return self._tab_imf_mc
        
    def _cache_ehat(self, key):
        if not hasattr(self, '_cache_ehat_'):
            self._cache_ehat_ = {}
            
        if key in self._cache_ehat_:
            return self._cache_ehat_[key]    
        
        return None    
        
    def _TabulateEmissivity(self, E=None, Emin=None, Emax=None, wave=None):
        """
        Compute emissivity over a grid of redshifts and setup interpolant.
        """   
        
        dz = self.pf['pop_synth_dz']
        zarr = np.arange(self.pf['pop_synth_zmin'], 
            self.pf['pop_synth_zmax'] + dz, dz)
        
        if (Emin is not None) and (Emax is not None):
            band = (Emin, Emax)
        else:
            band = None
        
        if (band is not None) and (E is not None):
            raise ValueError("You're being confusing! Supply `E` OR `Emin` and `Emax`")
            
        if wave is not None:
            raise NotImplemented('careful')    
            
        tab = np.zeros_like(zarr)
        
        for i, z in enumerate(zarr):
            _zarr, _tarr, L = self.SpectralSynthesis(zobs=z, band=band)
            
            # OK, we've got a whole population here.
            
            nh = self.get_field(z, 'nh')
            Mh = self.get_field(z, 'Mh')

            tab[i] = np.sum(L * nh)

        return zarr, tab
        
    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].

        Parameters
        ----------
        z : int, float

        Returns
        -------
        Emissivity in units of erg / s / c-cm**3 [/ eV]

        """

        on = self.on(z)
        if not np.any(on):
            return z * on

        # Need to build an interpolation table first.
        # Cache also by E, Emin, Emax
        
        cached_result = self._cache_ehat((E, Emin, Emax))
        if cached_result is not None:
            func = cached_result
            #return cached_result(z)
        else:
            
            zarr, tab = self._TabulateEmissivity(E, Emin, Emax)
            
            func = interp1d(zarr, np.log10(tab), kind='cubic', 
                bounds_error=False, fill_value=-np.inf)
            
            self._cache_ehat_[(E, Emin, Emax)] = func#zarr, tab
            
         
        return 10**func(z)
        #return self._cache_ehat_[(E, Emin, Emax)](z)
        
    def _gen_stars(self, idnum, Mh):
        """
        Take draws from cluster mass function until stopping criterion met.
        
        Return the amount of mass formed in this burst.
        """
        
        z = self._arr_z[idnum]
        t = self._arr_t[idnum]
        dt = (self._arr_t[idnum+1] - self._arr_t[idnum]) # in Myr
        
        E_h = self.halos.BindingEnergy(z, Mh)
        
        # Statistical approach from here on out.
        Ms = 0.0
        
        Mg = self._arr_Mg_c[idnum] * 1.
        
        # Number of supernovae from stars formed previously
        N_SN_p = self._arr_SN[idnum] * 1.
        E_UV_p = self._arr_UV[idnum] * 1.
        # Number of supernovae from stars formed in this timestep.
        N_SN_0 = 0. 
        E_UV_0 = 0.
        
        N_MS_now = 0
        m_edg = self.tab_imf_me
        m_cen = self.tab_imf_mc
        imf = np.zeros_like(m_cen)
        
        fstar_gmc = self.pf['pop_fstar_cloud']
        
        delay_fb_sne = self.pf['pop_delay_sne_feedback']
        delay_fb_rad = self.pf['pop_delay_rad_feedback']
                            
        vesc = self.halos.EscapeVelocity(z, Mh) # cm/s
                        
        # Form clusters until we use all the gas or blow it all out.    
        ct = 0
        Mw = 0.0
        Mw_rad = 0.0
        while (Mw + Mw_rad + Ms) < Mg * fstar_gmc:
            
            r = np.random.rand()
            Mc = np.interp(r, self.tab_cdf, self.tab_Mcl)
            
            # If proposed cluster would take up all the rest of our 
            # gas (and then some), don't let it happen.
            if (Ms + Mc + Mw) >= Mg:
                break
            
            ##
            # First things first. Figure out the IMF in this cluster.
            ##
                
            # Means we're doing cheap spectral synthesis.
            if self.pf['pop_sample_imf']:
                # For now, just scale UV luminosity with Nsn?
                
                # Uniform IMF placeholder
                #r2 = 0.1 + np.random.rand(1000) * (200. - 0.1)
                r2 = self._stars.draw_stars(1000000)

                # Integrate until we get Mc. 
                m2 = np.cumsum(r2)

                # What if, by chance, first star drawn is more massive than
                # Mc?
                
                cut = np.argmin(np.abs(m2 - Mc)) + 1
             
                if cut >= len(r2):
                    cut = len(r2) - 1
                    #print(r2.size, Mc, m2[-1] / Mc)
                    #raise ValueError('help')
                
                hist, edges = np.histogram(r2[0:cut+1], bins=m_edg)
                imf += hist
                                
                N_MS = np.sum(hist[m_cen >= 8.])
                
                # Ages of the stars
                
                #print((Mw + Ms) / Mg, Nsn, Mc / 1e3)
                                                                
            else:
                # Expected number of SNe if we were forming lots of clusters.
                lam = Mc * self._stars.nsn_per_m
                N_MS = np.random.poisson(lam)
                hist = 0.0
                
            ##
            # Now, take stars and let them feedback on gas.
            ##    

            ## 
            # Increment stuff
            ##      
            Ms += Mc
            imf += hist
            
            # Move along.
            if N_MS == 0:
                ct += 1
                continue
            
            ## 
            # Can delay feedback or inject it instantaneously.
            ##
            if delay_fb_sne == 0:
                N_SN_0 += N_MS  
            elif delay_fb_sne == 1:
                ##
                # SNe all happen at average delay time from formation.
                ##
                
                if self.pf['pop_sample_imf']:
                    raise NotImplemented('help')

                # In Myr
                avg_delay = self._stars.avg_sn_delay
                
                # Inject right now if timestep is long.
                if dt > avg_delay:
                    N_SN_0 += N_MS     
                else:
                    # Figure out when these guys will blow up.
                    #N_SN_next += N_MS
                    
                    tnow = self._arr_t[idnum]
                    tfut = self._arr_t[idnum:]
                                            
                    iSNe = np.argmin(np.abs((tfut - tnow) - avg_delay))
                    #if self._arr_t[idnum+iSNe] < avg_delay:
                    #    iSNe += 1
                    
                    print('hey 2', self._arr_t[idnum+iSNe] - tnow)
                    
                    self._arr_SN[idnum+iSNe] += N_MS                    
                                        
            elif delay_fb_sne == 2:
                ##
                # Actually spread SNe out over time according to DTD.
                ##
                delays = self._stars.draw_delays(N_MS)
                
                tnow = self._arr_t[idnum]
                tfut = self._arr_t[idnum:]
                
                # Could be more precise since closest index may be
                # slightly shorter than delay time.
                iSNe = np.array([np.argmin(np.abs((tfut - tnow) - delay)) \
                    for delay in delays])
                
                # Add SNe one by one.
                for _iSNe in iSNe:
                    self._arr_SN[idnum+_iSNe] += 1    
                
                # Must make some of the SNe happen NOW!
                N_SN_0 += sum(iSNe == 0)
                
            ##
            # Make a wind
            ##        
            Mw = 2 * (N_SN_0 + N_SN_p) * 1e51 * self.pf['pop_coupling_sne'] \
               / vesc**2 / g_per_msun
               
            ##                                 
            # Stabilize with some radiative feedback?
            ##
            if not self.pf['pop_feedback_rad']:
                ct += 1
                continue
                 
            ##
            # Dump in UV from 'average' massive star.
            ##
            if self.pf['pop_delay_rad_feedback'] == 0:
            
                # Mask out low-mass stuff? Only because scaling by N_MS
                massive = self._stars.Ms >= 8.
                
                LUV = self._stars.tab_LUV
                  
                Lavg = np.trapz(LUV[massive==1] * self._stars.tab_imf[massive==1], 
                    x=self._stars.Ms[massive==1]) \
                     / np.trapz(self._stars.tab_imf[massive==1], 
                    x=self._stars.Ms[massive==1])
                   
                life = self._stars.tab_life 
                tavg = np.trapz(life[massive==1] * self._stars.tab_imf[massive==1], 
                    x=self._stars.Ms[massive==1]) \
                     / np.trapz(self._stars.tab_imf[massive==1], 
                    x=self._stars.Ms[massive==1])    
                    
                corr = np.minimum(tavg / dt, 1.)
                                    
                E_UV_0 += Lavg * N_MS * dt * corr * 1e6 * s_per_yr
                                
                #print(z, 2 * E_rad / vesc**2 / g_per_msun / Mw)
                
                Mw_rad += 2 * (E_UV_0 + E_UV_p) * self.pf['pop_coupling_rad'] \
                    / vesc**2 / g_per_msun
            
            elif self.pf['pop_delay_rad_feedback'] >= 1:
                raise NotImplemented('help')
                
                delays = self._stars.draw_delays(N_MS)
                
                tnow = self._arr_t[idnum]
                tfut = self._arr_t[idnum:]
                
                # Could be more precise since closest index may be
                # slightly shorter than delay time.
                iSNe = np.array([np.argmin(np.abs((tfut - tnow) - delay)) \
                    for delay in delays])
                
                # Add SNe one by one.
                for _iSNe in iSNe:
                    self._arr_SN[idnum+_iSNe] += 1
                
            # Count clusters    
            ct += 1    
            
        ##
        # Back outside loop over clusters.
        ##    
                                    
        return Ms, Mw, imf
        
    def deposit_in(self, tnow, delay):
        """
        Determin index of time-array in which to deposit some gas, energy,
        etc., given current time and requested delay.
        """
        
        inow = np.argmin(np.abs(tnow - self._arr_t))
        
        tfut = self._arr_t[inow:]
                                
        ifut = np.argmin(np.abs((tfut - tnow) - delay))
                
        return inow + ifut
                
    def _gen_galaxy_history(self, halo, zobs=0):
        """
        Evolve a single galaxy in time. 
        
        Parameters
        ----------
        halo : dict
            Contains growth history of the halo of interest in order of
            *ascending redshift*. Must contain (at least) 'z', 't', 'Mh',
            and 'nh' keys.
        
        """
        
        # Grab key pieces of info
        z = halo['z'][-1::-1]
        t = halo['t'][-1::-1]
        Mh_s = halo['Mh'][-1::-1]
        MAR = halo['MAR'][-1::-1]
        nh = halo['nh'][-1::-1]
        
        self._arr_t = t
        self._arr_z = z
        
        zeros_like_t = np.zeros_like(t)
        
        self._arr_SN = zeros_like_t.copy()
        self._arr_UV = zeros_like_t.copy()
        self._arr_Mg_c = zeros_like_t.copy()
        self._arr_Mg_t = zeros_like_t.copy()
        
        # Short-hand
        fb = self.cosm.fbar_over_fcdm
        Mg_s = fb * Mh_s
        Nt = len(t)
        
        assert np.all(np.diff(t) >= 0)

        zform = max(z[Mh_s>0])

        SFR = np.zeros_like(Mh_s)
        Ms  = np.zeros_like(Mh_s)
        Msc = np.zeros_like(Mh_s)
        #Mg_t  = np.zeros_like(Mh_s)
        #Mg_c = np.zeros_like(Mh_s)
        Mw  = np.zeros_like(Mh_s)
        E_SN  = np.zeros_like(Mh_s)
        Nsn  = np.zeros_like(Mh_s)
        L1600 = np.zeros_like(Mh_s)
        bursty = np.zeros_like(Mh_s)
        imf = np.zeros((Mh_s.size, self.tab_imf_mc.size))
        #fc_r = np.ones_like(Mh_s)
        #fc_i = np.ones_like(Mh_s)
                        
        ok = Mh_s > 0                
                        
        # Generate smooth histories 'cuz sometimes we need that.
        MAR_s = np.array([self.guide.MAR(z=z[k], Mh=Mh_s[k]).squeeze() \
            for k in range(Nt)])
        SFE_s = np.array([self.guide.SFE(z=z[k], Mh=Mh_s[k]) \
            for k in range(Nt)])
        SFR_s = fb * SFE_s * MAR_s
        
        # Some characteristic timescales...
        tdyn = self.halos.DynamicalTime(z) / s_per_myr
        
        # in Myr
        delay_fb = self.pf['pop_delay_sne_feedback']
        
        
        ###
        ## THIS SHOULD ONLY HAPPEN WHEN NOT DETERMINISTIC
        ###
        ct = 0
        for i, _Mh in enumerate(Mh_s):

            if _Mh == 0:
                continue

            if z[i] < zobs:
                break
                
            if i == Nt - 1:
                break

            # In years    
            dt = (t[i+1] - t[i]) * 1e6

            if z[i] == zform:
                self._arr_Mg_t[i] = fb * _Mh
                
            # Determine gas supply
            if self.pf['pop_multiphase']:
                ifut = self.deposit_in(t[i], tdyn[i])
                self._arr_Mg_c[ifut] = self._arr_Mg_t[i] * 1
            else:
                self._arr_Mg_c[i] = self._arr_Mg_t[i] * 1
                    
            E_h = self.halos.BindingEnergy(z[i], _Mh)
            
            ##
            # Override switch to smooth inflow-driven star formation model.s
            ##
            if E_h > (1e51 * self.pf['pop_force_equilibrium']):
                vesc = self.halos.EscapeVelocity(z[i], _Mh)
                NSN_per_M = self._stars.nsn_per_m
                
                # Assume 1e51 * SNR * dt = 1e51 * SFR * SN/Mstell * dt = E_h
                eta = 2. * self.pf['pop_coupling_sne'] * 1e51 * NSN_per_M \
                    / g_per_msun / vesc**2
                                
                # SFR = E_h / 1e51 / (SN/Ms) / dt
                SFR[i]  = fb * MAR[i] / (1. + eta)
                Ms[i+1] = 0.5 * (SFR[i] + SFR[i-1]) * dt
                Mg[i+1] = Mg[i] + Macc - Ms[i+1]
                continue
                        
            ##
            # FORM STARS!
            ##
            
            # Gas we will accrete on this timestep
            Macc = fb * 0.5 * (MAR[i+1] + MAR[i]) * dt
            
            # What fraction of gas is in a phase amenable to star formation?
            if self.pf['pop_multiphase']:
                ifut = self.deposit_in(t[i], tdyn[i])
                self._arr_Mg_c[ifut] += Macc
            else:
                # New gas available to me right away in this model.
                self._arr_Mg_c[i] += Macc
                
            ##
            # Here we go.
            ##    
            _Mnew, _Mw, _imf = self._gen_stars(i, _Mh)    

            self._arr_Mg_t[i+1] = \
                max(self._arr_Mg_t[i] + Macc - _Mnew - _Mw, 0.)
 
            # Deal with cold gas.    
            if self.pf['pop_multiphase']:    
                #pass
                # Add remaining cold gas to reservoir for next timestep?
                # Keep gas hot for longer?
                # Subtract wind from cold gas reservoir?
                # Question is, do we feedback on gas that's already hot,
                # or gas that was "on deck" to form stars?
                
                #ifut = self.deposit_in(t[i], tdyn[i])
                #
                #self._arr_Mg_c[i:ifut] -= _Mw / (float(ifut - i))
                
                if self._arr_Mg_c[i+1] < 0:
                    print("Correcting for negative mass.", z[i])
                    self._arr_Mg_c[i+1] = 0
                
            #else:    
            #    self._arr_Mg_c[i+1] = self._arr_Mg_t[i+1]
            
            # Flag this step as bursty.
            bursty[i] = 1
                            
            # Save SFR. Set Ms, Mg for next iteration.
            SFR[i]   = _Mnew / dt
            imf[i]   = _imf
            Ms[i+1]  = _Mnew # just the stellar mass *formed this step*
            Msc[i+1] = Msc[i] + _Mnew
            
            ct += 1
            
        
        keep = np.ones_like(z)#np.logical_and(z > zobs, z <= zform)
        
        data = \
        { 
         'SFR': SFR[keep==1],
         'MAR': MAR_s[keep==1],
         'Mg': self._arr_Mg_t[keep==1], 
         'Mg_c':self._arr_Mg_c[keep==1], 
         'Ms': Msc[keep==1], # *cumulative* stellar mass!
         'Mh': Mh_s[keep==1], 
         'nh': nh[keep==1],
         'Nsn': Nsn[keep==1],
         'bursty': bursty[keep==1],
         'imf': imf[keep==1],
         'z': z[keep==1],
         't': t[keep==1],
         'zthin': halo['zthin'][-1::-1],
        }
        
        if 'rand' in halo:
            data['rand'] = halo['rand'][-1::-1]
        
                
        return data
            
    def _gen_galaxy_histories(self, zstop=0):     
        """
        Take halo histories and paint on galaxy histories in some way.
        
        If pop_stochastic, must operate on each galaxy individually using
        `self._gen_galaxy_history`, otherwise, can 'evolve' galaxies
        deterministically all at once.
        """
        
        # First, grab halos
        halos = self._gen_halo_histories()
        
        ## 
        # Stochastic model
        ##
        if self.pf['pop_sample_cmf']:
            
            fields = ['SFR', 'MAR', 'Mg', 'Ms', 'Mh', 'nh', 
                'Nsn', 'bursty', 'imf', 'rand']
            num = halos['Mh'].shape[0]
            
            hist = {key:np.zeros_like(halos['Mh']) for key in fields}
            
            # This guy is 3-D
            hist['imf'] = np.zeros((halos['Mh'].size, halos['z'].size,
                self.tab_imf_mc.size))
            
            for i in range(num): 
                
                print(i)
                #halo = {key:halos[key][i] for key in keys}
                halo = {'t': halos['t'], 'z': halos['z'], 
                    'zthin': halos['zthin'], 'rand': halos['rand'],
                    'Mh': halos['Mh'][i], 'nh': halos['nh'][i],
                    'MAR': halos['MAR'][i]}
                data = self._gen_galaxy_history(halo, zstop)
                
                for key in fields:
                    hist[key][i] = data[key]
        
            hist['z'] = halos['z']
            hist['t'] = halos['t']
            hist['zthin'] = halos['zthin']
        
            flip = {key:hist[key][-1::-1] for key in hist.keys()}
        
            self.histories = flip
            return flip                                            
        
        
        ##
        # Simpler models. No need to loop over all objects individually.
        ##
        
        # Eventually generalize
        assert self.pf['pop_update_dt'].startswith('native')
        native_sampling = True
        
        Nhalos = halos['Mh'].shape[0]
        
        # Flip arrays to be in ascending time.
        z = halos['z'][-1::-1]
        #z2d = halos['z2d'][:,-1::-1]
        #z2d = np.array([z] * Nhalos)[:,-1::-1]
        z2d = np.vstack([z] * Nhalos)[:,-1::-1]
        t = halos['t'][-1::-1]
        Mh = halos['Mh'][:,-1::-1]
        nh = halos['nh'][:,-1::-1]
        MAR = halos['MAR'][:,-1::-1]
        
        # 't' is in Myr
        dt = np.abs(np.diff(t)) * 1e6

        ##
        # OK. We've got a bunch of halo histories and we need to integrate them
        # to get things like stellar mass, metal mass, etc. This means we need
        # to make an assumption about how halos grow between our grid points.
        # If we assume smooth histories, we should be trying to do a trapezoidal
        # integration in log-space. However, given that we often add noise to
        # MARs, in that case we should probably just assume flat MARs within
        # the timestep. 
        #
        # Note also that we'll need to zero-pad these arrays to keep their 
        # shapes the same after we integrate (just so we don't have indexing
        # problems later). Since we're doing a simple sum, we'll fill the 
        # elements corresponding to the lowest redshift bin with zeros since we
        # can't compute luminosities after that point (no next grid pt to diff
        # with).
        ##
        
        
        fb = self.cosm.fbar_over_fcdm
        fZy = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield']
        fd = self.guide.dust_yield(z=z)   
        fml = (1. - self.pf['pop_mass_yield'])
        
        # Integrate (crudely) mass accretion rates
        #_Mint = cumtrapz(_MAR[:,:], dx=dt, axis=1)
        #_MAR_c = 0.5 * (np.roll(MAR, -1, axis=1) + MAR)
        #_Mint = np.cumsum(_MAR_c[:,1:] * dt, axis=1)
        
        if 'SFR' in halos:
            SFR = halos['SFR'][:,-1::-1]
        else:
            
            # Use cell-centered mass and redshift to compute SFE?
            dz = np.diff(z2d, axis=1)
            dM = np.diff(Mh, axis=1)
            #            
            Mh_cc = Mh.copy()
            Mh_cc[:,0:-1] += dM
            
            z2d_cc = z2d.copy()
            z2d_cc[:,0:-1] += dz
            
            SFR = MAR * fb * self.guide.SFE(z=z2d, Mh=Mh)
            
            #print(dz)
                        
            del dM, dz, Mh_cc, z2d_cc

        # 50% duty cycle
        #fduty = np.minimum(0.3 * (Mh / 1e10)**0.3, 1.)
        #delt = 30e6 * (Mh / 1e10)**0.4
        #
        #di = 0.5 * delt / np.hstack(([0], dt))[None,:]
        #
        #rand = np.reshape(np.random.rand(Mh.size), Mh.shape)
        #
        ###
        ## Method 1:
        ##
        #Pon = (fduty / delt) * np.hstack(([0], dt))[None,:]
        #
        #r = np.reshape(np.random.rand(Mh.size), Mh.shape)
        #
        ## Discrete events
        #on = np.array(r < Pon, dtype=int)
        #
        ## Pad by `delt` interval
        #a = np.argwhere(on == 1)
        #        
        #for ij in a:
        #    iM, iz = ij
        #    
        #    l = int(di[iM,iz])
        #    
        #    on[iM,iz-l:iz+l+1] = 1
        #    
        ##on = np.array(np.logical_not(off), dtype=int)
        #    
        #SFR = SFR * on
            
        if self.pf['conserve_memory']:
            dtype = np.float32
        else:
            dtype = np.float64
            
        zeros_like_Mh = np.zeros((Nhalos, 1), dtype=dtype)
        
        # Stellar mass should have zeros padded at the 0th time index
        Ms = np.hstack((zeros_like_Mh,
            np.cumsum(SFR[:,0:-1] * dt * fml, axis=1)))
                  
        # Dust           
        if np.any(fd > 0):
            
            # As long as the yields are constant then dust mass just scales with
            # stellar mass
            #Md = fd * fZy * Ms
            
            # Dust scale length
            #Rd = self.guide.dust_scale(z=z, Mh=Mh)
            
            Sd = fd * fZy * Ms * g_per_msun \
                / 4. / np.pi / (self.guide.dust_scale(z=z, Mh=Mh) * cm_per_kpc)**2
                        
        else:
            Md = Sd = 0.
            Rd = np.inf
                
        # Metal mass
        if 'Z' in halos:
            Z = halos['Z']
            Mg = 0.0
        else:
            if self.pf['pop_enrichment']:
                MZ = Ms * fZy
                
                # Gas mass
                Mg = np.hstack((zeros_like_Mh, 
                    np.cumsum((MAR[:,0:-1] * fb - SFR[:,0:-1]) * dt, axis=1)))
                    
                Z = MZ / Mg / self.pf['pop_fpoll']
            
                Z[Mg==0] = 1e-3
                Z = np.maximum(Z, 1e-3)    
                    
            else:
                MZ = Mg = Z = 0.0            
                
        # Pack up                
        results = \
        {
         'nh': nh,#[:,-1::-1],
         'Mh': Mh,#[:,-1::-1],
         't': t,#[-1::-1],
         'z': z,#[-1::-1],
         'zthin': halos['zthin'][-1::-1],
         #'z2d': z2d,
         'SFR': SFR,#[:,-1::-1],
         'Ms': Ms,#[:,-1::-1],
         #'MZ': MZ,#[:,-1::-1],
         #'Md': Md, # Only need 'Sd' for now so save some memory
         'Sd': Sd,
         'Mh': Mh,#[:,-1::-1],
         'Mg': Mg,#[:,-1::-1],
         'Z': Z,
         'bursty': zeros_like_Mh,
         'imf': np.zeros((Mh.shape[0], self.tab_imf_mc.size)),
         'Nsn': zeros_like_Mh,
         'Z': Z,
        }
        
        if self.pf['pop_dust_yield'] > 0:
            results['rand'] = halos['rand'][:,-1::-1]
            
        # Reset attribute!
        self.histories = results
                
        return results
                
    def Slice(self, z, slc):
        """
        slice format = {'field': (lo, hi)}
        """
        
        iz = np.argmin(np.abs(z - self.tab_z))
        hist = self.histories
        
        c = np.ones(hist['Mh'].shape[0], dtype=int)
        for key in slc:
            lo, hi = slc[key]
            
            ok = np.logical_and(hist[key][:,iz] >= lo, hist[key][:,iz] <= hi)
            c *= ok
    
        # Build output
        to_return = {}        
        for key in self.histories:
            if self.histories[key].ndim == 1:
                to_return[key] = self.histories[key][c==1]
            else:    
                to_return[key] = self.histories[key][c==1,iz]
                
        return to_return

    def get_field(self, z, field):
        iz = np.argmin(np.abs(z - self.histories['z']))
        return self.histories[field][:,iz]
        
    def StellarMassFunction(self, z, bins=None, units='dex'):
        """
        Could do a cumulative sum to get all stellar masses in one pass. 
        
        For now, just do for the redshift requested.
        
        Parameters
        ----------
        z : int, float
            Redshift of interest.
        bins : array, float
            log10 stellar masses at which to evaluate SMF
        
        """
                
        cached_result = self._cache_smf(z, bins)
        if cached_result is not None:
            return cached_result
        
        #iz = np.argmin(np.abs(z - self.tab_z))
        
        iz = np.argmin(np.abs(z - self.histories['z']))
        Ms = self.histories['Ms'][:,iz]
        Mh = self.histories['Mh'][:,iz]
        nh = self.histories['nh'][:,iz]
        
        # Need to re-bin
        #dMh_dMs = np.diff(Mh) / np.diff(Ms)
        #rebin = np.concatenate((dMh_dMs, [dMh_dMs[-1]]))
                
        if (bins is None) or (type(bins) is not np.ndarray):
            bin = 0.1
            bin_c = np.arange(6., 13.+bin, bin)
        else:
            dx = np.diff(bins)
            assert np.allclose(np.diff(dx), 0)
            bin = dx[0]
            bin_c = bins
            
        bin_e = bin_c2e(bin_c)
        
        #logM = np.log10(Ms)
        #ok = Ms > 0
        #print('bins', bin, bin_c, bin_e)
        phi, _bins = np.histogram(Ms, bins=10**bin_e, weights=nh)
        
        if units == 'dex':
            # Convert to dex**-1 units
            phi /= bin
        else:
            raise NotImplemented('help')
                        
        self._cache_smf_[z] = bin_c, phi
        
        return self._cache_smf(z, bins)
        
    def SMHM(self, z, Mh=None, return_mean_only=False, Mbin=0.1):
        """
        Compute stellar mass -- halo mass relation at given redshift `z`.
        
        .. note :: Because in general this is a scatter plot, this routine
            returns the mean and variance in stellar mass as a function of
            halo mass, the latter of which is defined via `Mh`.
        
        Parameters
        ----------
        z : int, float 
            Redshift of interest
        Mh : int, np.ndarray
            Halo mass bins (their centers) to use for histogram.
            Must be evenly spaced in log10
        
        """
        
        iz = np.argmin(np.abs(z - self.histories['z']))
        
        _Ms = self.histories['Ms'][:,iz]
        _Mh = self.histories['Mh'][:,iz]
        logMh = np.log10(_Mh)
        
        fstar_raw = _Ms / _Mh
        
        if (Mh is None) or (type(Mh) is not np.ndarray):
            bin_c = np.arange(6., 14.+Mbin, Mbin)
        else:
            dx = np.diff(np.log10(Mh))
            assert np.allclose(np.diff(dx), 0)
            Mbin = dx[0]
            bin_c = np.log10(Mh)
            
        nh = self.get_field(z, 'nh')    
        x, y, z = bin_samples(logMh, np.log10(fstar_raw), bin_c, weights=nh)    
            
        #bin_e = bin_c2e(bin_c)
        #
        #std = []
        #fstar = []
        #for i, lo in enumerate(bin_e):
        #    if i == len(bin_e) - 1:
        #        break
        #        
        #    hi = bin_e[i+1]
        #        
        #    ok = np.logical_and(logMh >= lo, logMh < hi)
        #    ok = np.logical_and(ok, _Mh > 0)
        #    
        #    f = np.log10(fstar_raw[ok==1])
        #    
        #    if f.size == 0:
        #        std.append(-np.inf)
        #        fstar.append(-np.inf)
        #        continue
        #                
        #    #spread = np.percentile(f, (16., 84.))
        #    #
        #    #print(i, np.mean(f), spread, np.std(f))
        #    
        #    std.append(np.std(f))
        #    fstar.append(np.mean(f))
        #
        #std = np.array(std)
        #fstar = np.array(fstar)

        if return_mean_only:
            return y

        return x, y, z

    def L_of_Z_t(self, wave):
        
        if not hasattr(self, '_L_of_Z_t'):
            self._L_of_Z_t = {}
            
        if wave in self._L_of_Z_t:
            return self._L_of_Z_t[wave]

        tarr = self.src.times
        Zarr = np.sort(list(self.src.metallicities.values()))
        L = np.zeros((tarr.size, Zarr.size))
        for j, Z in enumerate(Zarr):
            L[:,j] = self.src.L_per_SFR_of_t(wave, Z=Z)
            
        # Interpolant
        self._L_of_Z_t[wave] = RectBivariateSpline(np.log10(tarr), 
            np.log10(Zarr), L, kx=3, ky=3)
            
        return self._L_of_Z_t[wave]
        
    def find_trajectory(self, Mh, zh):
        l = np.argmin(np.abs(zh - self.tab_z))
        mass_at_zh = np.zeros_like(self.tab_z)
        for j, zform in enumerate(self.tab_z):
            if zform < zh:
                continue
                
            hist = self.histories['Mh'][j]
            
            mass_at_zh[j] = hist[l]
        
        # Find trajectory with mass closest to Mh at zh
        k = np.argmin(np.abs(Mh - mass_at_zh))
        
        return k
        
    @property
    def _stars(self):
        if not hasattr(self, '_stars_'):
            self._stars_ = SynthesisModelSBS(**self.src_kwargs)
        return self._stars_
        
    def _cache_ss(self, key):
        if not hasattr(self, '_cache_ss_'):
            self._cache_ss_ = {}
            
        if key in self._cache_ss_:
            return self._cache_ss_[key]    
        
        return None        
    
    def SpectralSynthesis(self, hist=None, idnum=None, zobs=None, wave=1600., 
        weights=1, band=None):
        """
        Yield spectrum for a single galaxy or the entire population.
        
        Parameters
        ----------
        hist : dict
            Dictionary containing the trajectory of a single object. Most
            importantly, must contain 'SFR', as a function of *increasing
            time*, stored in the key 't'.
            
        For example, hist could be the output of a call to _gen_galaxy_history.
            
        """            
        
        batch_mode = False
        
        # This means we're doing synthesis for a single object.
        if idnum is not None:
            
            assert hist is None, "Must supply `hist` OR `idnum`!"
            hist = self.get_history(idnum)
            
            cached_result = self._cache_ss((idnum, wave, band))
            if cached_result is not None:
                                
                if zobs is None:
                    return cached_result
                
                # Do zobs correction    
                izobs = np.argmin(np.abs(hist['z'] - zobs))
                if hist['z'][izobs] > zobs:
                    izobs += 1

                zarr, tarr, L = cached_result
                return zarr[0:izobs+1], tarr[0:izobs+1], L[0:izobs+1]
        # If no 'hist' provided, do spectral synthesis for all objects.
        elif hist is None:
            hist = self.histories   
            batch_mode = True

        # Must find index corresponding to requested redshift of observation.
        if zobs is None:
            izobs = None
        else:
            # Need to be sure that we grab a grid point exactly at or just
            # below the requested redshift (?)
            izobs = np.argmin(np.abs(hist['z'] - zobs))
            if hist['z'][izobs] > zobs:
                izobs += 1
        
        # Can load batch results from cache as well
        if batch_mode:
            cached_result = self._cache_ss((zobs, wave, band))
            if cached_result is not None:        
                zarr, tarr, L = cached_result
                return zarr[0:izobs+1], tarr[0:izobs+1], L
                
        # Must be supplied in increasing time order, decreasing redshift.
        assert np.all(np.diff(hist['t']) >= 0)
        
        #if not np.any(hist['SFR'] > 0):
        #    print('nipped this in the bud')
        #    return hist['t'], hist['z'], np.zeros_like(hist['z'])
        #
        izform = 0#min(np.argwhere(hist['Mh'] > 0))[0]
        
        if zobs is None:
            slc = Ellipsis
        else:    
            if batch_mode:
                #raise NotImplemented('help')
                # Need to slice over first dimension now...
                slc = Ellipsis, slice(0, izobs+1)
            else:    
                slc = slice(0, izobs+1)

        ##
        # Dust model?
        ##
        fd = self.guide.dust_yield(z=hist['z'][slc])
        if np.any(fd > 0) and (band is None):
            fcov = self.guide.dust_fcov(z=hist['z'][slc], Mh=hist['Mh'][slc])
            kappa = self.guide.dust_kappa(wave=wave)
            Sd = hist['Sd'][slc]
            tau = kappa * Sd
            rand = hist['rand'][slc]            
        else:
            tau = fcov = 0.0
            rand = 1#np.ones_like(hist['SFR'])

        ##
        # First. Simple case without stellar population aging.
        ##
        if not self.pf['pop_aging']:
            assert not self.pf['pop_ssp'], \
                "Should not have pop_ssp==True if pop_aging==False."
            
            if band is not None:
                raise NotImplemented('help!')
            
            L_per_sfr = self.src.L_per_sfr(wave)
            return hist['z'][slc], hist['t'][slc], L_per_sfr * hist['SFR'][slc]
        
        ##
        # Second. Harder case where aging is allowed.
        ##          
        assert self.pf['pop_ssp']
                
        # SFH        
        Nsn  = hist['Nsn'][slc]
        SFR  = hist['SFR'][slc]
        bursty = hist['bursty'][slc]
        imf = hist['imf'][slc]
        
        if izobs is not None:
            tarr = hist['t'][0:izobs+1] # in Myr
            zarr = hist['z'][0:izobs+1]
        else:
            tarr = hist['t']    
            zarr = hist['z']
        
        
        fill = np.zeros(1, 
            dtype=np.float32 if self.pf['conserve_memory'] else np.float64)
        tyr = tarr * 1e6
        dt = np.hstack((np.diff(tyr), fill))
        
        oversample = (zobs is not None) and self.pf['pop_ssp_oversample'] and \
            (dt[-2] >= 2e6)
        
        if band is not None:
            Loft = self.src.IntegratedEmission(band[0], band[1])
            #raise NotImplemented('help!')        
            print("Note this now has different units.")
        else:        
            Loft = self.src.L_per_SFR_of_t(wave)
            
        #if self.pf['conserve_memory']:
        #    Loft = np.array(Loft, dtype=np.float32)
                    
        _func = interp1d(np.log(self.src.times), np.log(Loft),
            kind='cubic', bounds_error=False, 
            fill_value=Loft[-1])
            
        # Extrapolate linearly at times < 1 Myr
        _m = (Loft[1] - Loft[0]) \
          / (self.src.times[1] - self.src.times[0])
        L_small_t = lambda age: _m * age + Loft[0]
        
        # Extrapolate as PL at t < 1 Myr based on first two
        # grid points
        #m = np.log(Loft[1] / Loft[0]) \
        #  / np.log(self.src.times[1] / self.src.times[0])
        #func = lambda age: np.exp(m * np.log(age) + np.log(Loft[0]))
                
        if zobs is None:
            Lhist = np.zeros_like(SFR)
        else:
            pass
            # Lhist will just get made once. Don't need to initialize
        
        # Loop over the history of object(s) and compute the luminosity of 
        # simple stellar populations of the corresponding ages (relative to
        # zobs).
        
        # Start from initial redshift and move forward in time.
        for i, _tobs in enumerate(tarr):
            
            #print(i, len(tarr), _tobs, zarr[i], zobs)
            
            # If zobs is supplied, we only have to do one iteration
            # of this loop. This is just a dumb way to generalize this function
            # to either do one redshift or return a whole history.
            if (zobs is not None):
                if (zarr[i] > zobs):
                    continue

                assert i == tarr.size - 1    

            ages = tarr[i] - tarr[0:i+1]

            # Treat metallicity evolution? If so, need to grab luminosity as 
            # function of age and Z.
            if self.pf['pop_enrichment']:

                if batch_mode:
                    Z = hist['Z'][slc][:,0:i+1]
                else:
                    Z = hist['Z'][slc][0:i+1]

                logA = np.log10(ages)
                logZ = np.log10(Z)
                L_per_msun = self.L_of_Z_t(wave)(logA, logZ, grid=False)
                                        
                # erg/s/Hz
                if batch_mode:
                    Lall = L_per_msun * SFR[:,0:i+1]
                else:
                    Lall = L_per_msun * SFR[0:i+1]
                    
                if oversample:
                    raise NotImplemented('help!')    
                else:
                    _dt = dt
            else:    
                
                # If time resolution is >= 2 Myr, over-sample final interval.
                if oversample:
                    
                    # Use 1 Myr time resolution for final stretch
                    extra = np.arange(ages[-1], ages[-2], 1.)[-1::-1]
                    
                    # Must augment ages and dt accordingly
                    _ages = np.hstack((ages[0:-1], extra))
                    _dt = np.abs(np.diff(_ages) * 1e6)
                    
                    # 
                    _dt = np.hstack((_dt, [0]))
                    
                    # Now, compute luminosity at expanded ages.
                    L_per_msun = np.exp(_func(np.log(_ages)))    
                    
                    #L_per_msun = np.exp(np.interp(np.log(_ages), 
                    #    np.log(self.src.times), np.log(Loft), 
                    #    left=-np.inf, right=np.log(Loft[-1])))
                        
                    # Interpolate linearly at t < 1 Myr    
                    func = lambda age: age * Loft[0]
                    L_per_msun[_ages < 1] = func(_ages[_ages < 1])    
                    
                    # Must reshape SFR to match. Assume constant SFR within
                    # over-sampled integral.                    
                    xSFR = SFR[:,-1][:,None] * np.ones((SFR.shape[0], extra.size))
                    
                    if batch_mode:
                        _SFR = np.hstack((SFR[:,0:-1], xSFR))
                    else:
                        _SFR = np.hstack((SFR[0:i], SFR[i+1] * np.ones_like(extra)))
                    
                    # erg/s/Hz
                    if batch_mode:
                        Lall = L_per_msun * _SFR
                    else:    
                        Lall = L_per_msun * _SFR
                                                                 
                else:    
                    L_per_msun = np.exp(np.interp(np.log(ages), 
                        np.log(self.src.times), np.log(Loft), 
                        left=np.log(Loft[0]), right=np.log(Loft[-1])))
                    
                    _dt = dt
                    
                    if self.pf['conserve_memory']:
                        L_per_msun = L_per_msun.astype(np.float32)
                                                            
                    # Fix early time behavior
                    L_per_msun[ages < 1] = L_small_t(ages[ages < 1])
        
                    # erg/s/Hz
                    if batch_mode:
                        Lall = L_per_msun * SFR[:,0:i+1]
                    else:    
                        Lall = L_per_msun * SFR[0:i+1]
                                                    
                # Correction for IMF sampling (can't use SPS).
                #if self.pf['pop_sample_imf'] and np.any(bursty):
                #    life = self._stars.tab_life
                #    on = np.array([life > age for age in ages])
                #
                #    il = np.argmin(np.abs(wave - self._stars.wavelengths))
                #
                #    if self._stars.aging:
                #        raise NotImplemented('help')
                #        lum = self._stars.tab_Ls[:,il] * self._stars.dldn[il]
                #    else:
                #        lum = self._stars.tab_Ls[:,il] * self._stars.dldn[il]
                #
                #    # Need luminosity in erg/s/Hz
                #    #print(lum)
                #
                #    # 'imf' is (z or age, mass)
                #
                #    integ = imf[bursty==1,:] * lum[None,:]
                #    Loft = np.sum(integ * on[bursty==1], axis=1)
                #
                #    Lall[bursty==1] = Loft

            # Integrate over all times up to this tobs
            
            if batch_mode:
                if (zobs is not None):
                    Lhist = np.trapz(Lall, dx=_dt[0:-1], axis=1)
                else:
                    Lhist[:,i] = np.trapz(Lall, dx=_dt[0:i], axis=1)
            else:
                assert not batch_mode, "still working geez"
                
                Lhist[i] = np.trapz(Lall, x=tyr[0:i+1])                

            ##
            # In this case, we only need one iteration of this loop.
            ##
            if zobs is not None:
                break
                                        
        # Redden away!        
        if np.any(fd) > 0 and (band is None):
            # Reddening is binary and probabilistic
            clear = rand > fcov
            block = ~clear
                        
            Lout = Lhist * clear[:,-1] + Lhist * np.exp(-tau[:,-1]) * block[:,-1]
        else:
            Lout = Lhist.copy()
                    
        del Lhist, tau, Lall    
        gc.collect()
                            
        # Only cache if we've got the whole history
        if (zobs is None) and (idnum is not None):
            self._cache_ss_[(idnum, wave, band)] = zarr, tarr, Lout
        # Or if we did batch mode
        elif (zobs is not None) and (idnum is None):
            init = self._cache_ss((zobs, wave, band))
            self._cache_ss_[(zobs, wave, band)] = zarr, tarr, Lout
          
        if zobs is not None:
            return zarr[slc], tarr[slc], Lout
        else:
            return zarr[slc], tarr[slc], Lout[slc]
          
    def _cache_lf(self, z, x=None):
        if not hasattr(self, '_cache_lf_'):
            self._cache_lf_ = {}
            
        if z in self._cache_lf_:            
                        
            _x, _phi = self._cache_lf_[z]  
            
            # If no x supplied, return bin centers
            if x is None:
                return _x, _phi  
            
            if type(x) != np.ndarray:
                k = np.argmin(np.abs(x - _x))
                if abs(x - _x[k]) < 1e-3:
                    return _phi[k]
                else:
                    phi = 10**np.interp(x, _x, np.log10(_phi),
                        left=-np.inf, right=-np.inf)
                    
                    # If _phi is 0, interpolation will yield a NaN
                    if np.isnan(phi):
                        return 0.0
                    return phi
            if _x.size == x.size:
                if np.allclose(_x, x):
                    return _phi
            
            return 10**np.interp(x, _x, np.log10(_phi), 
                left=-np.inf, right=-np.inf)
                
        return None
        
    def _cache_smf(self, z, Mh):
        if not hasattr(self, '_cache_smf_'):
            self._cache_smf_ = {}
    
        if z in self._cache_smf_:
            _x, _phi = self._cache_smf_[z]    
                
            if Mh is None:
                return _phi        
            elif type(Mh) != np.ndarray:
                k = np.argmin(np.abs(Mh - _x))
                if abs(Mh - _x[k]) < 1e-3:
                    return _phi[k]
                else:
                    return 10**np.interp(Mh, _x, np.log10(_phi),
                        left=-np.inf, right=-np.inf)
            elif _x.size == Mh.size:
                if np.allclose(_x, Mh):
                    return _phi
            
            return 10**np.interp(Mh, _x, np.log10(_phi),
                left=-np.inf, right=-np.inf)
            
        return None  
        
    def get_history(self, i):
        """
        Extract a single halo's trajectory from full set, return in dict form.
        """
        
        # These are kept in ascending redshift just to make life difficult.
        raw = self.histories
                                
        hist = {'t': raw['t'], 'z': raw['z'], 
            'SFR': raw['SFR'][i], 'Mh': raw['Mh'][i], 
            'bursty': raw['bursty'][i], 'Nsn': raw['Nsn'][i],
            'imf': raw['imf'][i]}
            
        if self.pf['pop_dust_yield'] > 0:
            hist['rand'] = raw['rand'][i]
            hist['Sd'] = raw['Sd'][i]
            
        if self.pf['pop_enrichment']:
            hist['Z'] = raw['Z'][i]
        
        return hist
        
    def get_histories(self, z):
        for i in range(self.histories['Mh'].shape[0]):
            yield self.get_history(i)
            
    def LuminosityFunction(self, z, x, mags=True, wave=1600., band=None,
        batch=False):
        """
        Compute the luminosity function from discrete histories.
        
        Need to be a little careful about indexing here. For example, if we
        request the LF at a redshift not present in the grid, we need to...
        
        """
        
        cached_result = self._cache_lf(z, x)
        if cached_result is not None:
            return cached_result
                                
        # These are kept in descending redshift just to make life difficult.
        # [The last element corresponds to observation redshift.]
        raw = self.histories
                        
        keys = raw.keys()
        Nt = raw['t'].size
        Nh = raw['Mh'].shape[0]
        
        # Find the z grid point closest to that requested.
        # Must be >= z requested.
        izobs = np.argmin(np.abs(self.histories['z'] - z))
        if z > self.histories['z'][izobs]:
            # Go to one grid point lower redshift
            izobs += 1
                                            
        ##
        # Run in batch? 
        if batch:
            
            # Must supply in time-ascending order
            zarr, tarr, L = self.SpectralSynthesis(idnum=None, 
                hist=None, zobs=z, wave=wave)        
                        
        else:
        
            L = np.zeros(Nh)
            for i in range(Nh):                                        
                # Must supply in time-ascending order
                zarr, tarr, Lt[i] = self.SpectralSynthesis(idnum=i,
                    zobs=z, wave=wave)
            

        # Need to be more careful here as nh can change when using
        # simulated halos
        w = raw['nh'][:,izobs+1]
                                                
        _MAB = self.magsys.L_to_MAB(L, z=z)
        
        if self.pf['dustcorr_method'] is not None:
            MAB = self.dust.Mobs(z, _MAB)
        else:
            MAB = _MAB
                        
        # If L=0, MAB->inf. Hack these elements off if they exist.
        # This should be a clean cut, i.e., there shouldn't be random
        # spikes where L==0, all L==0 elements should be a contiguous chunk.
        Misok = np.logical_and(L > 0, np.isfinite(L))
        
        # Always bin to setup cache, interpolate from then on.
        _x = np.arange(-28, 5., self.pf['pop_mag_bin'])

        hist, bin_edges = np.histogram(MAB[Misok==1], 
            weights=w[Misok==1], bins=bin_c2e(_x), density=True)
            
        bin_c = _x
        
        N = np.sum(w[Misok==1])
        
        phi = hist * N
                
        self._cache_lf_[z] = bin_c, phi
        
        return self._cache_lf(z, x)
        
    def _cache_beta(self, z, wave, wave_MUV):
        if not hasattr(self, '_cache_beta_'):
            self._cache_beta_ = {}
            
        if (z, wave, wave_MUV) in self._cache_beta_:
            return self._cache_beta_[(z, wave, wave_MUV)]
            
        return None
        
    def get_beta(self, idnum=None, zobs=None, wave=1600., dlam=100):
        """
        Get UV slope for a single object.
        """

        if self.src.pf['source_sed'] == 'eldridge2009':
            _lo = np.argmin(np.abs(wave - self.src.wavelengths))    
            _hi = np.argmin(np.abs(wave + dlam - self.src.wavelengths))    
               
            lo = self.src.wavelengths[_lo] 
            hi = self.src.wavelengths[_hi]  
                
            ok = np.logical_or(lo == self.src.wavelengths, 
                               hi == self.src.wavelengths)
            
            ok = np.logical_or(ok, wave == self.src.wavelengths)
            
        else:
            lo = np.argmin(np.abs(wave - dlam - self.src.wavelengths))
            me = np.argmin(np.abs(wave - self.src.wavelengths))
            hi = np.argmin(np.abs(wave + dlam - self.src.wavelengths))
            
            ok = np.zeros_like(self.src.wavelengths)
            ok[lo] = 1; ok[me] = 1; ok[hi] = 1

        if ok.sum() < 2:
            raise ValueError('Need at least two wavelength points to compute slope! Have {}.'.format(ok.sum()))
            
        arr = self.src.wavelengths[ok==1]
        
        if idnum is None:
            batch_mode = True
        else:
            batch_mode = False

        Lh = []
        MAB = []
        #Lh = np.zeros((arr.size, izobs+1))
        for j, w in enumerate(arr):
            zarr, tarr, _L = self.SpectralSynthesis(idnum=idnum, zobs=zobs, 
                wave=w)
            Lh.append(_L * 1.)
            
            if j == 1:
                MAB = self.magsys.L_to_MAB(Lh[j], z=zarr[-1])
        
        #if batch_mode:
        #    Lh_l = np.array(Lh) / self.src.dwdn[ok==1,None]
        #else:    
        Lh_l = np.array(Lh) / self.src.dwdn[ok==1,None]
        
        logw = np.log(arr)
        logL = np.log(Lh_l)

        #if batch_mode:
        #    beta = (logL[0,:,:] - logL[-1,:,:]) / (logw[0,None,None] - logw[-1,None,None])
        #else:
        beta = (logL[0,:] - logL[-1,:]) / (logw[0,None] - logw[-1,None])

        return MAB, beta

    def Beta(self, z, MUV=None, wave=1600., wave_MUV=1600., dlam=100,
        return_binned=True, Mbins=None, batch=False):
        """
        UV slope for all objects in model.
        
        Parameters
        ----------
        z : int, float
            Redshift.
        MUV : int, float, np.ndarray
            Optional. Set of magnitudes at which to return Beta.
            Note: these need not be at the same wavelength as that used
                  to compute Beta (see wave_MUV).
        wave : int, float
            Wavelength at which to compute Beta.
        wave_MUV : int, float
            Wavelength assumed for input MUV. Allows us to return, e.g., the
            UV slope at 2200 Angstrom corresponding to input 1600 Angstrom
            magnitudes.
        dlam : int, float
            Interval over which to average UV slope [Angstrom]
        return_binned : bool
            If True, return binned version of MUV(Beta), including 
            standard deviation of Beta in each MUV bin.
        
        Returns
        -------
        if MUV is None:
            Tuple containing (magnitudes, slopes)
        else:
            Slopes at corresponding user-supplied MUV (assumed to be at 
            wavelength `wave_MUV`).
        """
        
        cached_result = self._cache_beta(z, wave, wave_MUV)
        
        if cached_result is None:
            _lo = np.argmin(np.abs(wave - self.src.wavelengths))    
            _hi = np.argmin(np.abs(wave + dlam - self.src.wavelengths))    
               
            lo = self.src.wavelengths[_lo] 
            hi = self.src.wavelengths[_hi]  
                
            ok = np.logical_or(lo == self.src.wavelengths, 
                               hi == self.src.wavelengths)
            
            ok = np.logical_or(ok, wave == self.src.wavelengths)
                        
            if ok.sum() < 2:
                raise ValueError('Need at least two wavelength points to compute slope! Have {}.'.format(ok.sum()))
            
            arr = self.src.wavelengths[ok==1]
            
            if batch:
                _MAB, _beta = self.get_beta(None, z, wave, dlam)
                MAB = _MAB[:]
                beta = _beta[:]
            else:    
            
                beta = []
                MAB = []
                for i, hist in enumerate(self.get_histories(z)):
                    _muv1, _beta1 = self.get_beta(i, z, wave, dlam)
                    if wave_MUV != wave:
                        _muv2, _beta2 = self.get_beta(i, z, wave, dlam)
                        MAB.append(_muv2[-1])
                    else:
                        MAB.append(_muv1[-1])
                        
                    beta.append(_beta1[-1])
                    
                beta = np.array(beta)
                MAB = np.array(MAB) 
            
            # Only cache
            self._cache_beta_[(z, wave, wave_MUV)] = MAB, beta
                                     
        else:
            MAB, beta = cached_result
            
        if return_binned:
            if Mbins is None:
                Mbins = np.arange(-25, -10, 0.1)
                
            nh = self.get_field(z, 'nh')
                
            _x, _y, _z = bin_samples(MAB, beta, Mbins, weights=nh)
                        
            MAB = _x
            beta = _y
            std = _z 
        else:
            std = None   
            assert MUV is None

        if MUV is not None:
            assert return_binned
            _beta = np.interp(MUV, MAB, beta, left=-9999, right=-9999)
            return _beta
                                                                
        return MAB, beta, std
        
    def AUV(self, z, MUV=None, wave=1600., dlam=100., eff=False,
        return_binned=True, Mbins=None):
        """
        Compute UV extinction.
        """
        
        kappa = self.guide.dust_kappa(wave=wave)
        Sd = self.get_field(z, 'Sd')
        tau = kappa * Sd
        
        AUV = np.log10(np.exp(-tau)) / -0.4
            
        # Just do this to get MAB array of same size as Mh
        MAB, beta, std = self.Beta(z, MUV=None, wave=wave, dlam=dlam,
            return_binned=False, batch=True)
                
        if return_binned:            
            if Mbins is None:
                Mbins = np.arange(-25, -10, 0.1)
                
            nh = self.get_field(z, 'nh')    
            _x, _y, _z = bin_samples(MAB, AUV, Mbins, weights=nh)
                    
            MAB = _x
            AUV = _y
            std = _z 
        else:
            #MAB = np.flip(MAB)
            #beta = np.flip(beta)
            std = None
                
            assert MUV is None    
                
        if MUV is not None:
            _AUV = np.interp(MUV, MAB[-1::-1], AUV[-1::-1], left=0., right=0.)
            return _AUV

        return MAB, AUV, std
        
    def get_contours(self, x, y, bins):
        """
        Take 'raw' data and recover contours. 
        """

        bin_e = bin_c2e(bins)
        
        band = []
        for i in range(len(bins)):
            ok = np.logical_and(x >= bin_e[i], x < bin_e[i+1])
            
            if ok.sum() == 0:
                band.append((-np.inf, -np.inf, -np.inf))
                continue
            
            yok = y[ok==1]
            
            y1, y2 = np.percentile(yok, (16., 84.))
            ym = np.mean(yok)
            
            band.append((y1, y2, ym))
            
        return np.array(band).T
        
    def MainSequence(self, z):
        """
        How best to plot this?
        """
        pass
        
    def SFRF(self, z):
        pass
            
    def PDF(self, z, **kwargs):
        # Look at distribution in some quantity at fixed z, potentially other stuff.
        pass
    
    def load(self):
        """
        Load results from past run.
        """
                
        fn_hist = self.pf['pop_histories']
        
        # Look for results attached to hmf table
        if fn_hist is None:
            prefix = self.guide.halos.tab_prefix_hmf(True)
            fn = self.guide.halos.tab_name
        
            suffix = fn[fn.rfind('.')+1:]
            path = os.getenv("ARES") + '/input/hmf/'
            fn_hist = path + prefix + '.hist.' + suffix
                    
        # Read output
        if type(fn_hist) is str:
            if fn_hist.endswith('.pkl'):
                f = open(fn_hist, 'rb')
                prefix = fn_hist.split('.pkl')[0]
                zall, traj_all = pickle.load(f)
                f.close()
                
                hist = traj_all
                      
            elif fn_hist.endswith('.hdf5'):
                f = h5py.File(fn_hist, 'r')
                prefix = fn_hist.split('.hdf5')[0]
                
                if self.pf['conserve_memory']:
                    dtype = np.float32
                else:
                    dtype = np.float64
                
                hist = {}
                for key in f.keys():
                    
                    if isinstance(f[(key)], h5py._hl.dataset.Dataset):
                        hist[key] = np.array(f[(key)], dtype=dtype)
                    else:    
                        hist[key] = np.array(f[(key)])
                
                zall = hist['z']
                
                f.close()
                
            else:
                # Assume pickle?
                f = open(fn_hist+'.pkl', 'rb')
                prefix = fn_hist
                zall, traj_all = pickle.load(f)
                f.close()
            
                hist = traj_all
                
            hist['zform'] = zall
            hist['zobs'] = np.array([zall] * hist['nh'].shape[0])
            
            ## Check to see if parameters match
            print("Need to check that HMF parameters match!")
        elif type(self.pf['pop_histories']) is dict:
            hist = self.pf['pop_histories']
            # Assume you know what you're doing.
        else:
            hist = None
            
        return hist
    
    def save(self, prefix, clobber=False):
        """
        Output model (i.e., galaxy trajectories) to file.
        """
        
        fn = prefix + '.pkl'
                
        if os.path.exists(fn) and (not clobber):
            raise IOError('File \'{}\' exists! Set clobber=True to overwrite.'.format(fn))
        
        zall, traj_all = self._gen_halo_histories()

        f = open(fn, 'wb')
        pickle.dump((zall, traj_all), f)
        f.close()
        print("Wrote {}".format(fn))
        
        # Also save parameters.
        f = open('{}.parameters.pkl'.format(prefix))
        pickle.dump(self.pf)
        f.close()
        
        