"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description: 

"""

import os
import time
import pickle
import numpy as np
from ..util import ProgressBar
from .Halo import HaloPopulation
from .GalaxyCohort import GalaxyCohort
from ..util.Stats import bin_e2c, bin_c2e
from scipy.integrate import quad, cumtrapz
from ..analysis.BlobFactory import BlobFactory
from scipy.interpolate import RectBivariateSpline
from ..physics.Constants import rhodot_cgs, s_per_yr, s_per_myr

pars_affect_mars = ["pop_MAR", "pop_MAR_interp", "pop_MAR_corr"]
pars_affect_sfhs = ["pop_scatter_sfr", "pop_scatter_sfe", "pop_scatter_mar"]
pars_affect_sfhs.extend(["pop_update_dt","pop_thin_hist"])

class GalaxyEnsemble(HaloPopulation,BlobFactory):
    
    def __init__(self, **kwargs):
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
        
        iz = np.argmin(np.abs(z - self.tab_z))
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
        
        if thin == 0:
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
        noise = 10**(np.log10(arr) + np.reshape(lognoise, arr.shape))
        return np.reshape(noise, arr.shape)
        
    def gen_kelson(self, guide, thin):
        raise NotImplementedError('help')
        dt = self.pf['pop_update_dt']
        t = np.arange(30., 2000+dt)[-1::-1]
        z = self.cosm.z_of_t(t * s_per_myr)
        N = z.size
        
        # In this case we need to thin before generating SFHs?
        
        if thin == 0:
            thin = 1
            
        sfr0_func = lambda zz: self.guide.SFR(z=zz, Mh=guide.Mmin(zz))
        
        dsfr0_func = derivative(sfr0_func)
        
        zthin = self.tile(z, thin, False)
        sfr = np.zeros((z.size * thin, z.size))
        nh  = np.zeros((z.size * thin, z.size))
        Mh  = np.zeros((z.size * thin, z.size))
        k = 0
        for i in range(N):
            nh[k,0:i+1] = np.interp(z[i], guide.halos.tab_z, guide._tab_n_Mmin)
            Mh[k,0:i+1] = 0.0

            sfr0 = sfr0_func(z[i])
            #s = 

            for j in range(0, thin):
                                
                sfr[k,0:i+1] = self._gen_kelson_hist(sfr0, t[0:i+1], s)[-1::-1]
                k += 1

        # Need to come out of this block with sfr, z, at a minimum.
        zall = z
        
        nh = self.tile(nh, thin, True)
        Mh = self.tile(Mh, thin)

        raw = {'z': z, 'SFR': sfr, 'nh': nh, 
            'Mh': Mh, 'MAR': None, 'SFE': None}

        return zall, raw
        
    def _gen_kelson_history(self, sfr0, t, sigma=0.3):
        sfr = np.zeros_like(t)
        sfr[0] = sfr0
        for i in range(1, t.size):
            sfr[i] = sfr[i-1] * np.random.lognormal(mean=0., sigma=sigma)
        
        return np.array(sfr)
    
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
        
    def _gen_halo_histories(self):
        """
        From a set of smooth halo assembly histories, build a bigger set
        of histories by thinning, and (optionally) adding scatter to the MAR. 
        """            
                    
        raw = self.load()
                
        thin = self.pf['pop_thin_hist']
                
        sigma_sfe = self.pf['pop_scatter_sfe']
        sigma_mar = self.pf['pop_scatter_mar']
        sigma_env = self.pf['pop_scatter_env']
                                        
        # Just read in histories in this case.
        if raw is None:
            print('Running halo trajectories...')
            zall, raw = self.guide.Trajectories()
            print('Done with halo trajectories.')
        else:
            zall = raw['z']        
        
        self.raw = raw
        
        sfe_raw = raw['SFE']    
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
        # Just adding scatter in SFR
        #if have_sfr:
        #    sfr = self.tile(sfr_raw, thin)
        #
        #if sigma_sfr > 0:
        #    assert have_sfr
        #    assert not (self.pf['pop_scatter_sfe'] or self.pf['pop_scatter_mar'])                
        #    sfr += self.noise_lognormal(sfr, sigma_sfr)
        
        # Can add SFE scatter
        sfe = self.tile(sfe_raw, thin)
        if sigma_sfe > 0:
            sfe += self.noise_lognormal(sfe, sigma_sfe)
        
        # Two potential kinds of scatter in MAR    
        mar = self.tile(mar_raw, thin)
        if sigma_env > 0:
            mar *= (1. + self.noise_normal(mar, sigma_env))
            
        if sigma_mar > 0:
            mar += self.noise_lognormal(mar, sigma_mar)
            #sfr = sfe * mar * self.cosm.fbar_over_fcdm
        #else:
        #    if not have_sfr:
        #        sfr = sfe * mar * self.cosm.fbar_over_fcdm 
                
        # SFR = (zform, time (but really redshift))
        # So, halo identity is wrapped up in axis=0
        # In Cohort, formation time defines initial mass and trajectory (in full)
        zobs = np.array([zall] * nh.shape[0])
        histories = {'zuni': zall, 'zobs': zobs, 'Mh': Mh,
            'MAR': mar, 'nh': nh}
            
        # Add in formation redshifts to match shape (useful after thinning)
        histories['zform'] = self.tile(zall, thin)
                        
        self.tab_z = zall
                        
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
                _tdyn = self.halos.DynamicalTime(1e10, z) / s_per_yr
            
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
                self._guide = GalaxyCohort(**self.pf)        
        
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
        
    def _gen_stars(self, z, Mh, Mg, E_SN=0.0):
        """
        Take draws from cluster mass function until stopping criterion met.
        
        Return the amount of mass formed in this burst.
        """
        
        E_h = self.halos.BindingEnergy(Mh, z)
        
        # Statistical approach from here on out.
        Ms = 0.0
        N_sn = 0
        
        if self.pf['pop_delay_feedback']:
            Efb = E_SN * 1.
            print('Preemptive feedback', Efb >= E_h)
        else:
            Efb = 0.0
        
        # Form clusters until we use all the gas or blow it all out.    
        while Efb < E_h:

            r = np.random.rand()
            Mc = np.interp(r, self.tab_cdf, self.tab_Mcl)

            # Poisson-ify the SN feedback
            Mavg = 1e-1
            fsn = 1e-2
            
            # Expected number of SNe if we were forming lots of clusters.
            lam = Mc * fsn / Mavg

            Nsn = np.random.poisson(lam)
                                    
            # May want to hang on to few bins worth of Ms to track
            # SNe, HMXBs, etc.
            
            #gas_avail = (Mr * fc_r[i] + Ma * fc_i[i]) #* fstar
            gas_limited = False
            if Ms + Mc >= Mg:                
                gas_limited = True
                break
            
            ## 
            # Increment stuff
            ##      
            Ms += Mc 
            N_sn += Nsn
            
            #Efb += 1e50 * Mc
            
            # Only increment energy injection now if we're assuming
            # instantaneous feedback.
            if not self.pf['pop_delay_feedback']:
                Efb += 1e51 * Nsn

        return Ms, N_sn
        
    def _gen_galaxy_history(self, halo, zobs=0):
        """
        Evolve a single galaxy in time. Only gets called if not deterministic.
        """
        
        # Grab key pieces of info
        z = halo['z'][-1::-1]
        t = halo['t'][-1::-1]
        Mh_s = halo['Mh'][-1::-1]
        
        # Short-hand
        fb = self.cosm.fbar_over_fcdm
        Mg_s = fb * Mh_s
        Nt = len(t)
        
        assert np.all(np.diff(t) >= 0)

        zform = max(z[Mh_s>0])

        SFR = np.zeros_like(Mh_s)
        Ms  = np.zeros_like(Mh_s)
        Mg  = np.zeros_like(Mh_s)
        E_SN  = np.zeros_like(Mh_s)
        N_SN  = np.zeros_like(Mh_s)
        #fc_r = np.ones_like(Mh_s)
        #fc_i = np.ones_like(Mh_s)
        nsn = np.zeros_like(Mh_s)
                
        # Generate smooth histories 'cuz sometimes we need that.
        MAR_s = np.array([self.guide.MAR(z=z[k], Mh=Mh_s[k]).squeeze() \
            for k in range(Nt)])
        SFE_s = np.array([self.guide.SFE(z=z[k], Mh=Mh_s[k]) \
            for k in range(Nt)])
        SFR_s = fb * SFE_s * MAR_s
        
        # in Myr
        delay_fb = self.pf['pop_delay_feedback']
        
        ###
        ## THIS SHOULD ONLY HAPPEN WHEN NOT DETERMINISTIC
        ###
        for i, _Mh in enumerate(Mh_s):

            if _Mh == 0:
                continue

            if z[i] < zobs:
                break
                
            if i == Nt - 1:
                break    

            dt = (t[i+1] - t[i]) * 1e6


            if z[i] == zform:
                Mg[i] = fb * _Mh

            # Gas we will accrete on this timestep
            Macc = fb * 0.5 * (MAR_s[i+1] + MAR_s[i]) * dt #* fc_i[i]
        
            E_h = self.halos.BindingEnergy(_Mh, z[i])
            
            ##
            # Override switch to smooth inflow-driven star formation model.s
            ##
            if E_h > (1e51 * self.pf['pop_force_equilibrium']):
                SFR[i] = SFR_s[i]
                Ms[i+1]  = 0.5 * (SFR[i] + SFR[i-1]) * dt
                Mg[i+1]  = Mg[i] + Macc - Ms[i+1]
                continue
            
            ##
            # FORM STARS!
            ##
            Mnew, Nsn = self._gen_stars(z[i], _Mh, Mg[i] + Macc, E_SN[i])
                
            if delay_fb:
                j = np.argmin(np.abs(t - (t[i] + delay_fb)))
            else:
                j = i
            
            # Track this no matter what. _gen_stars will only use it
            # if delay_fb == True.
            N_SN[j] += Nsn
            E_SN[j] += 1e51 * Nsn
                
            # Save SFR. Set Ms, Mg for next iteration.
            SFR[i]   = Mnew / dt
            Ms[i+1]  = Mnew # just the stellar mass *formed this step*
            Mg[i+1]  = Mg[i] + Macc - Mnew

        
        data = {'SFR': SFR, 'Mg': Mg, 'Ms': Ms, 'z': z, 't': t}       
                
        return data
        
    def _gen_galaxy_histories(self):     
        """
        Take halo histories and paint on galaxy histories in some way.
        """
        
        # First, grab halos
        halos = self._gen_halo_histories()
                                                        
        # Array of unique redshifts
        zarr = halos['zuni']  
        
        # Thinned array of redshifts, i.e., one per unique object bin.
        all_zform = halos['zform']#[0]
        
        nh = halos['nh']
        
        # Useful to have a uniform grid for output?
        zform_max = max(all_zform)

        
        native_sampling = False
        if self.pf['pop_update_dt'].startswith('native'):
            # This means just use the redshift sampling of the halo
            # population, which is set by hmf_* parameters.
            native_sampling = True

                
        # Setup redshift and time arrays.
        if native_sampling:    
            if '/' in self.pf['pop_update_dt']:
                _pre, _step = self.pf['pop_update_dt'].split('/')
                step = -int(_step)
            else:
                step = -1
                
            zall = zarr[-1::step]
            
            assert np.allclose(zarr, self.tab_z)
            
            tall = self.tab_t[-1::step]
            zobs = halos['zobs']
            #tobs = np.array([tall] * halos['nh'].shape[0])
            tobs = tall#self.cosm.t_of_z(zobs) / s_per_yr
        else:
            native_sampling = False
            tall, zall = self.get_timestamps(zform_max)
        
        # In this model we only use 'thin' to set size of arrays
                
        ##
        # Simpler case. If using 'native' sampling, it means every halo
        # has outputs at the same times, which means we can integrate
        # all trajectories simultaneously using array operations.
        ##
        if native_sampling:
                    
            # These will be (zform, zarr).
            # We need to reverse the ordering of the second dimension
            # to be in ascending *time*. We'll switch it back at the end.
            Nz = len(zall)
            _Mh = halos['Mh'][:,-1::step]
            _MAR = halos['MAR'][:,-1::step]
            _SFE = self.guide.SFE(z=zall, Mh=_Mh)#halos['SFE'][:,-1::step]
            shape = halos['nh'].shape

            fbar = self.cosm.fbar_over_fcdm 
            
            # 'tall' is in Myr
            dt = np.abs(np.diff(tall))
            
            # These means there is no internal feedback, so we can
            # just apply the time evolution all at once.
            # This isn't quite true -- feedback is baked in via the SFE,
            # but aside from that, no other feedback.
            if not self.pf['pop_internal_feedback']:
                                        
                # Integrate (crudely) mass accretion rates
                #_Mint = cumtrapz(_MAR[:,:], dx=dt, axis=1)
                _MAR_c = 0.5 * (np.roll(_MAR, -1, axis=1) + _MAR)
                _Mint = np.cumsum(_MAR_c[:,1:] * dt, axis=1)
                                
                # Increment relative to initial masses. Need to put in 
                # a set of zeros for first element to get shape right.
                #Mh0 = _Mh[:,0]
                Mh0 = self.guide.Mmin(zall)
                _fill = np.zeros((_MAR.shape[0], 1))
                #Mh = Mh0 + np.concatenate((_fill, _Mint), axis=1)
                
                Mh = _Mh
                
                _SFE_c = 0.5 * (np.roll(_SFE, -1, axis=1) + _SFE)
                SFR = _MAR * _SFE * fbar
                SFR_c = _MAR_c * _SFE_c * fbar
                
                MGR = _MAR * fbar
                MGR_c = _MAR_c * fbar

                # Stellar mass
                fml = (1. - self.pf['pop_mass_yield'])
                Ms0 = np.zeros((SFR.shape[0], 1))
                Msj = np.cumsum(SFR_c[:,1:] * dt * fml, axis=1)
                #Msj = cumtrapz(SFR[:,:], dx=dt, axis=1)
                #Msj = 0.5 * cumtrapz(SFR * tall, x=np.log(tall), axis=1)
                Ms = np.concatenate((Ms0, Msj), axis=1)

                # Metal mass
                fZy = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield']
                MZ0 = np.zeros((SFR.shape[0], 1))
                MZj = np.cumsum(SFR_c[:,1:] * dt * fZy, axis=1)
                #Msj = cumtrapz(SFR[:,:], dx=dt, axis=1)
                #Msj = 0.5 * cumtrapz(SFR * tall, x=np.log(tall), axis=1)
                MZ = np.concatenate((MZ0, MZj), axis=1)
                
                # Gas mass
                Mg0 = Mh0 * fbar
                Mg = Mg0 + np.concatenate((_fill, _Mint * fbar), axis=1)
                #Mgj = np.cumsum(MGR_c[:,1:] * dt, axis=1)
                #Msj = cumtrapz(SFR[:,:], dx=dt, axis=1)
                #Msj = 0.5 * cumtrapz(SFR * tall, x=np.log(tall), axis=1)
                #Mg = np.concatenate((Mg0, Mgj), axis=1)
                                
                #
                ## Dust mass
                fd = self.pf['pop_dust_yield']
                #MD_tot[i,j+1] = 0.0
            else:
                # Loop over time.
                for j, _t in enumerate(tall):
                    
                    if j == jmax:
                        break
                            
                    Mh[:,j+1]   = Mh[:,j] + _MAR[:,j] * dt[j]
                    
                    
                    Mg_c[i,j+1] = Mg_c[i,j] * (1. - c_to_h) \
                                + Mg_h[i,j] * h_to_c \
                                + D_Mg_tot  * n_to_c
                    Mg_h[i,j+1] = Mg_h[i,j] * (1. - h_to_c) \
                                + Mg_c[i,j] * c_to_h \
                                + D_Mg_tot  * (1. - n_to_c)
                    
                    # Total gas mass
                    Mg_tot[i,j+1] = Mg_tot[i,j] - SFR[i,j] * dt + D_Mg_tot
                    
                    # Stellar mass
                    Ms[i,j+1] = Ms[i,j] + SFR[i,j] * dt # no losses yet
                    
                    # Metal mass
                    MZ_tot[i,j+1] = 0.0
                    
                    # Dust mass
                    MD_tot[i,j+1] = 0.0
        
            results = \
            {
             'nh': nh,
             'Mh': Mh[:,-1::-1],
             'zform': all_zform,
             'zobs': zobs[-1::-1],  # Just because it's what LF needs
             'tobs': tobs[-1::-1],  # Just because it's what LF needs
             #'mask': mask,
             'SFR': SFR[:,-1::-1],
             'Mg': Mg,
             #'Mg_h': Mg_h,
             #'Mg_tot': Mg_tot,
             'Ms': Ms[:,-1::-1],
             'MZ': MZ[:,-1::-1],
             'Mh': Mh[:,-1::-1],
             'Z': MZ[:,-1::-1] / Mg[:,-1::-1],
             #'Na': Na,
             #'Np': Np,
            }
            
        
        else:
        
            # One of these output arrays could eventually have a third dimension
            # for wavelength. Well...we do synthesis after the fact, so no.
            Mh = np.zeros((len(all_zform), tall.size))
            Mg_tot = np.zeros_like(Mh)
            Mg_c = np.zeros_like(Mh)
            Mg_h = np.zeros_like(Mh)
            MZ_tot = np.zeros_like(Mh)
            MD_tot = np.zeros_like(Mh)
            Ms =  np.zeros_like(Mh)
            SFR = np.zeros_like(Mh)
            Np = np.zeros_like(Mh)
            Na = np.zeros_like(Mh)
            zobs = np.zeros_like(Mh)
            tobs = np.zeros_like(Mh)
            mask = np.zeros_like(Mh)
            
            # Results arrays don't have 1:1 mapping between index:time.
            # Deal with it!
            
            # Shortcuts
            fbar = self.cosm.fbar_over_fcdm 
            
            # Loop over formation redshifts, run the show.
            for i, zform in enumerate(all_zform):
                t, z = self.get_timestamps(zform)
                
                #print(zform, len(t))
                
                #tdyn = self.halos.DynamicalTime(1e10, z) / s_per_yr
                k = t.size
                
                # Grab the things that we can't modify (too much)
                _Mh = np.interp(z, zarr, hist['Mh'][i,:])
                _MAR = np.interp(z, zarr, hist['MAR'][i,:])
                
                # May have use for this.
                _SFE = self.guide.SFE(z=z, Mh=_Mh)
            
                zobs[i,:k] = z.copy()
                tobs[i,:k] = t.copy()
                
                # Unused time elements (corresponding to z > zform)
                mask[i,k:] = 1
                
                jmax = t.size - 1
                
                # Sure would be nice to eliminate the following loop.
                # Can only do that if there's no randomness that depends on
                # previous timesteps.
                # Note 12.03. Should eliminate 'i' loop since all galaxies
                # are independent.
                
                # This can go faster...
                if self.pf['pop_bcycling'] == False:
                    # Really want an internal feedback determinism switch.
                    pass
                
                Mh[i,0] = _Mh[0]            
                Mg_h[i,0] = 0.0
                Mg_c[i,0] = fbar * _Mh[0]
                Mg_tot[i,0] = fbar * _Mh[0]
                for j, _t in enumerate(t):
                                    
                    if j == jmax:
                        break
                    
                    dt = t[j+1] - _t
                    
                    # Use _MAR or _Mh here? Should be solving for Mh!
                    this_MAR = _MAR[j] #(_MAR[j+1]+_MAR[j]) * 0.5
                    D_Mg_tot = max(fbar * this_MAR * dt, 0.)
            
                    fstar = _SFE[j]
                    
                    fgmc = 0.1
                    
                    _sfr_new = D_Mg_tot * fstar / dt
                    _sfr_res = Mg_c[i,j] * 0.02 * fgmc / dt
                    
                    #SFR[i,j] = _sfr_res#max(_sfr_new * (1. + np.random.normal(scale=0.5)), 0)
                    
                    #f_cl = np.random.rand()
                    f_cl = 0.02 # I would think this is an absolute maximum...unles
                    # star formation really (usually) proceeds until it can't.
                    
                    if np.isnan(D_Mg_tot):
                        continue
                                            
                    #N = int(D_Mg_tot * fstar / self.Mcl)
                    N = int((_sfr_new + _sfr_res) * dt / self.Mcl)
                    
                    if self.pf['pop_poisson']: 
                        _N = np.random.poisson(lam=N)
                    else:
                        _N = N
                    
                    Np[i,j] = N
                    Na[i,j] = _N
                    
                    # Incorporate newly accreted gas into this estimate?
                    # super-efficient means we use up all the reservoir AND
                    # some new gas
                    if self.pf['pop_poisson']:
                        
                        print('hey poisson')
                    
                        if N > 1e2:
                            print('N huge')
                            SFR[i,j] = _sfr_new
                            masses = _sfr_new * dt * np.ones(N) / float(N)
                        else:    
                            print('N small')
                            r = np.random.rand(_N)
                            masses = 10**np.interp(np.log10(r), np.log10(self.tab_cdf), 
                                np.log10(self.tab_Mcl))
                                
                            # Cap SFR so it can't exceed 0.02 * Mgc / dt?
                            SFR[i,j] = min(np.sum(masses), Mg_c[i,j] * f_cl * fgmc) / dt        
                    else:
                        #print('not poisson')
                        #masses = self.Mcl * np.ones(N)
                        #SFR[i,j] = _sfr_new
                    
                    # This happens occasionally!
                    #if np.sum(masses) > Mg_c[i,j] * 0.02:
                    #    print('hey', i, z[j], _N)
                    
                    
                                    
                    
                        if self.pf['pop_sf_via_inflow']:
                            SFR[i,j] += _sfr_new
                        elif self.pf['pop_sf_via_reservoir']:
                            SFR[i,j] += _sfr_res
                                    
                                    
                                    
                                    
                                    
                    ##
                    # UPDATES FOR NEXT TIMESTEP
                    ##
                    
                    # Is gas cold or hot?
                    if self.pf['pop_bcycling'] == 'random':
                        n_to_c = np.random.rand()
                        c_to_h = np.random.rand()
                        h_to_c = np.random.rand()
                    elif self.pf['pop_bcycling'] == False:    
                        n_to_c = 1.  # new gas -> cold
                        c_to_h = 0.  # turning cold -> hot
                        h_to_c = 0.  # turning hot  -> cold
                    else:
                        raise NotImplemented('help')
                        
                    # How to make function of global galaxy properties, potentially
                    # on past timestep?
                    
                    # Impart fractional change
                    # Simplest model: increase or decrease.
                    
                    Mh[i,j+1]   = Mh[i,j] + _MAR[j] * dt
                    
                    Mg_c[i,j+1] = Mg_c[i,j] * (1. - c_to_h) \
                                + Mg_h[i,j] * h_to_c \
                                + D_Mg_tot  * n_to_c
                    Mg_h[i,j+1] = Mg_h[i,j] * (1. - h_to_c) \
                                + Mg_c[i,j] * c_to_h \
                                + D_Mg_tot  * (1. - n_to_c)
                    
                    # Total gas mass
                    Mg_tot[i,j+1] = Mg_tot[i,j] - SFR[i,j] * dt + D_Mg_tot
                    
                    # Stellar mass
                    Ms[i,j+1] = Ms[i,j] + SFR[i,j] * dt # no losses yet
                    
                    # Metal mass
                    MZ_tot[i,j+1] = 0.0
                    
                    # Dust mass
                    MD_tot[i,j+1] = 0.0
                
            results = \
            {
             'nh': nh,
             'Mh': Mh,
             'zform': all_zform, # These are thinned
             'zobs': zobs,
             'tobs': tobs,
             #'mask': mask,
             'SFR': SFR,
             #'Mg_c': Mg_c,
             #'Mg_h': Mg_h,
             #'Mg_tot': Mg_tot,
             'Ms': Ms,
             #'Na': Na,
             #'Np': Np,
            }
        
        # Reset attributes!
        #self.tab_z = results['zobs']
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
        iz = np.argmin(np.abs(z - self.tab_z))
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
        
        iz = np.argmin(np.abs(z - self.tab_z))
        
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
            assert np.all(np.diff(dx) == 0)
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
        
    def SMHM(self, z):
        iz = np.argmin(np.abs(z - self.tab_z))
        
        return self.histories['Ms'][:,iz], self.histories['Mh'][:,iz]
        
    def L_of_Z_t(self, wave):
        
        if not hasattr(self, '_L_of_Z_t'):
            self._L_of_Z_t = {}
            
        if wave in self._L_of_Z_t:
            return self._L_of_Z_t[wave]

        print('making interpolant')

        tarr = self.src.times
        Zarr = np.sort(list(self.src.metallicities.values()))
        L = np.zeros((tarr.size, Zarr.size))
        for j, Z in enumerate(Zarr):
            L[:,j] = self.src.L_per_SFR_of_t(wave)
            
        # Interpolant
        self._L_of_Z_t[wave] = RectBivariateSpline(np.log10(tarr), 
            np.log10(Zarr), L, kx=3, ky=3)
    
        print('done with interpolant')
        
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
        
    def SpectralSynthesis(self, hist, zobs=None, wave=1600., weights=1):
        """
        Yield spectrum for a single object.
        
        note: need to generalize SAM so that single object run is possible.
        """
                
        if zobs is None:
            iz = None
        else:
            iz = np.argmin(np.abs(zobs - hist['z']))
        
        assert np.all(np.diff(hist['t']) >= 0)
        
        SFR = hist['SFR'][:iz]
        
        ##
        # First. Simple case without stellar population aging.
        ##
        if not self.pf['pop_aging']:
            assert not self.pf['pop_ssp'], \
                "Should not have pop_ssp==True if pop_aging==False."
            
            L_per_sfr = self.src.L_per_sfr(wave)
            return L_per_sfr * SFR
        
        ##
        # Second. Harder case where aging is allowed.
        ##          
        assert self.pf['pop_ssp']
        
        # Birth redshift
        ib = min(np.argwhere(SFR > 0))[0]
        SFR = SFR[ib:iz]
        
        tarr = hist['t'][ib:iz] # in Myr
        zarr = hist['z'][ib:iz]
        dt = np.diff(tarr) * 1e6
        
        ages = np.abs(tarr[-1] - tarr)
          
        if self.pf['pop_enrichment']:
            raise NotImplementedError('help')
            Z = self.histories['Z']
            L_per_msun = self.L_of_Z_t(wave)(np.log10(ages),
                np.log10(Z))
        else:    
            L_per_msun = np.interp(ages, self.src.times, 
                self.src.L_per_SFR_of_t(wave))
                    
        return zarr, tarr, np.cumsum(L_per_msun[0:-1] * SFR[0:-1] * dt)
          
          
          
        ###
        ### OLD OLD OLD OLD
        ###  
                 
        if self.is_deterministic:
                 
            zarr = self.tab_z[iz:]
            
            # Array of times corresponding to all z' > z.
            tarr = self.tab_t[iz:]
            
            # Ages of stellar populations formed at each z' > z
            ages = (tarr[0] - tarr) / 1e6
                            
            dz = self.tab_dz
            
            # Eventually shouldn't be necessary
            #assert np.allclose(dz, self.pf['sam_dz']), \
            #    "dz={}, sam_dz={}".format(dz, self.pf['sam_dz'])
            
            # in years
            dt = dz * self.cosm.dtdz(zarr) / s_per_yr
            
            # Is this all just an exercise in careful binning?
            # Also, within a bin, if we assume a constant SFR, there could
            # be a non-negligible age gradient....ugh.
                                        
            
            if self.pf['pop_enrichment']:
                Z = self.histories['Z']
                
                print('hey', ages.shape, Z.shape)
                L_per_msun = self.L_of_Z_t(wave)(np.log10(ages),
                    np.log10(Z))
                
                
            else:    
                L_per_msun = np.interp(ages, self.src.times, 
                    self.src.L_per_SFR_of_t(wave))
            
            # Note: for SSP, this is in [erg/s/Msun]    
            
            # Conceptually, at this observed redshift, for all sources
            # with a unique identity (in simplest case: formation redshift)
            # we must sum over the ages present in those galaxies.
            
            # Also: must use burst model for star formation, which leads
            # to complication with fact that we defined a set of SFHs...
            # maybe only once we add some noise.
            
            # Shape = formation redshifts or ID
            L = np.zeros_like(self.histories['SFR'][:,0])
            for k in range(self.histories['SFR'].shape[0]):
                # For each unique object (or object bin), sum emission
                # over past episodes of star formation.
        
                L[k] = np.sum(L_per_msun * self.histories['SFR'][k,iz:] * dt)
                #L[k] = np.trapz(L_per_msun * self.histories['SFR'][k,iz:],
                #    dx=dt[0:-1])
                    
        else:
            iz = np.argmin(np.abs(z - self.tab_z))   
            # In this case, the time-stepping is different for each 
            # trajectory. 
            #tnow = self.cosm.t_of_z(z) / s_per_yr
            znow = self.tab_z[iz]
            tnow = self.tab_t[iz]
            
            # In this case, can do some stuff up-front.
            if self.pf['pop_update_dt'].startswith('native'):
                native_sampling = True
                all_tarr = self.histories['tobs'][-1::-1]
                all_zarr = self.histories['zobs'][0][-1::-1]
                all_dt = np.diff(all_tarr)
                all_dz = np.diff(all_zarr)
                all_SFR = self.histories['SFR'][:,-1::-1]
                imax = len(all_tarr)
                inow = np.argmin(np.abs(all_tarr - tnow))
                if all_tarr[inow] < tnow:
                    inow += 1
            else:
                native_sampling = False                                    
                
            times = self.src.times    
                
            # Loop over all objects.
            L = np.zeros_like(self.histories['zobs'][:,0])
            #corr = np.ones_like(L)
            for k in range(self.histories['zobs'].shape[0]):
                
                # This galaxy formed after redshift of interest.
                if not native_sampling:
                    if np.all(self.histories['zobs'][k] < z):
                        print('should this only happen for non-native dt?')
                        continue
                
                # In ascending time, descending redshift.
                # First redshift element is zform, arrays are filled until
                # object(s) reach z=0 (i.e., not necessarily the last element
                # of the array).
                
                if not native_sampling:
                    tarr = self.histories['tobs'][k] # [yr]
                    zarr = self.histories['zobs'][k]
                    SFR  = self.histories['SFR'][k]
                    Z = np.maximum(self.histories['Z'][k], 1e-3)
                    
                    if not np.all(np.diff(tarr) > 0):
                        tarr = tarr[-1::-1]
                        zarr = zarr[-1::-1]
                        SFR  = SFR[-1::-1]   
                        
                    _imax = np.argwhere(tarr == 0)
                    if len(_imax) > 0:
                        imax = min(_imax[0]) - 1
                    else:
                        imax = len(tarr)    
                        
                    inow = np.argmin(np.abs(tarr - tnow))
                    if tarr[inow] < tnow:
                        inow += 1    
                          
                else:
                    tarr = all_tarr
                    zarr = all_zarr
                    SFR = all_SFR[k]
                                            
                # Avoid diff-ing on each iteration if we can
                if native_sampling:
                    dt = all_dt[0:imax+1]
                    dz = all_dz[0:imax+1]
                else:
                    dt = np.diff(tarr[0:imax+1])
                    dz = np.diff(zarr[0:imax+1])
                
                # In order to get ages, need to invoke current redshift.
                _ages = (tnow - tarr[0:imax+1]) / 1e6
                # Hack out elements beyond current observed redshift 
                ages = _ages[0:inow+1]
                
                ## 
                # Optional: obscuration
                ##
                if self.pf['pop_fobsc'] in [0, None]:
                    corr = 1.
                else:
                    _M = self.histories['Mh'][k]
                    
                    # Means obscuration refers to fractional dimming of individual 
                    # objects
                    if self.pf['pop_fobsc_by'] == 'lum':
                        fobsc = self.guide.fobsc(z=znow, Mh=_M[iz])
                        corr = (1. - fobsc)
                    else:
                        raise NotImplemented('help')
                             
                #print(k, z, tnow / 1e7, ages / 1e7)
                #raw_input('<enter>')
                
                if len(ages) == 0:
                    print('ages has no elements!', k)
                    continue
                                    
                # Need to be careful about interpolating within last
                # dynamical time.
                if self.pf['pop_enrichment']:
        
                   #L_per_msun = self.L_of_Z_t(wave)(np.log10(ages), 
                   #    np.log10(Z[0:inow+1]))
                        
                    # spline wants output to be 2-D.
                    # we just have a time series of (ages, Z)
                    # sooo....
                    
                    logA = np.log10(ages)
                    logZ = np.log10(Z[0:inow+1])
                    
                    spl = self.L_of_Z_t(wave)
                    L_per_msun = [spl(logA[w], logZ[w]).squeeze() \
                        for w in range(logA.size)]
                    L_per_msun = np.array(L_per_msun)
                        
                    
                    #\
                    ##    [self.L_of_Z_t(wave)(np.log10(age), np.log10(Z)) \
                    #        for age in ages]
                    #L_per_msun = np.array(L_per_msun)        
                else:        
                    L_per_msun = np.interp(ages, times, 
                        self.src.L_per_SFR_of_t(wave))
                        
                        
                L_per_msun *= corr
                        
                #_w = np.ones_like(L_per_msun)
                
                if not native_sampling:
                    raise NotImplemented('Do we need to modify weight for last time bin?')
                
                #_w[-1] = (tarr[inow+1] - tnow) / dt[-1]
                #print(k, dt[0:inow-1], _w, L_per_msun, self.histories['SFR'][k,0:inow])
        
                # Fix last chunk of weights?
        
                #L[k] = np.trapz(L_per_msun * SFR[0:inow+1],
                #    dx=dt[0:inow])
                L[k] = np.sum(L_per_msun * SFR[0:inow+1] * dt[0:inow+1])
                            
        return L
        
    def _cache_lf(self, z, x):
        if not hasattr(self, '_cache_lf_'):
            self._cache_lf_ = {}
            
        if z in self._cache_lf_:            
            _x, _phi = self._cache_lf_[z]    
            
            if type(x) != np.ndarray:
                k = np.argmin(np.abs(x - _x))
                if abs(x - _x[k]) < 1e-3:
                    return _phi[k]
                else:
                    return 10**np.interp(x, _x, np.log10(_phi))
            elif np.allclose(_x, x):
                return _phi
            else:
                return 10**np.interp(x, _x, np.log10(_phi))
                
        return None
        
    def _cache_smf(self, z, bins):
        if not hasattr(self, '_cache_smf_'):
            self._cache_smf_ = {}
    
        if z in self._cache_smf_:
            _x, _phi = self._cache_smf_[z]    
                
            if bins is None:
                return _phi        
            elif type(bins) != np.ndarray:
                k = np.argmin(np.abs(bins - _x))
                if abs(bins - _x[k]) < 1e-3:
                    return _phi[k]
                else:
                    return 10**np.interp(bins, _x, np.log10(_phi))
            elif _x.size == bins.size:
                if np.allclose(_x, bins):
                    return _phi
            
            return 10**np.interp(bins, _x, np.log10(_phi))
            
        return None    
        
    def LuminosityFunction(self, z, x, mags=True, wave=1600., band=None):
        """
        Compute the luminosity function from discrete histories.
        """
        
        cached_result = self._cache_lf(z, x)
        if cached_result is not None:
            return cached_result
                        
        # Care required!
        if self.pf['pop_aging']:   
            
            assert self.pf['pop_ssp']
                     
            if self.is_deterministic:
                iz = np.argmin(np.abs(z - self.tab_z))         
                     
                zarr = self.tab_z[iz:]
                
                # Array of times corresponding to all z' > z.
                tarr = self.tab_t[iz:]
                
                # Ages of stellar populations formed at each z' > z
                ages = (tarr[0] - tarr) / 1e6
                                
                dz = self.tab_dz
                
                # Eventually shouldn't be necessary
                #assert np.allclose(dz, self.pf['sam_dz']), \
                #    "dz={}, sam_dz={}".format(dz, self.pf['sam_dz'])
                
                # in years
                dt = dz * self.cosm.dtdz(zarr) / s_per_yr
                
                # Is this all just an exercise in careful binning?
                # Also, within a bin, if we assume a constant SFR, there could
                # be a non-negligible age gradient....ugh.
                                            
                
                if self.pf['pop_enrichment']:
                    Z = self.histories['Z']
                    
                    print('hey', ages.shape, Z.shape)
                    L_per_msun = self.L_of_Z_t(wave)(np.log10(ages),
                        np.log10(Z))
                    
                    
                else:    
                    L_per_msun = np.interp(ages, self.src.times, 
                        self.src.L_per_SFR_of_t(wave))
                
                # Note: for SSP, this is in [erg/s/Msun]    
                
                # Conceptually, at this observed redshift, for all sources
                # with a unique identity (in simplest case: formation redshift)
                # we must sum over the ages present in those galaxies.
                
                # Also: must use burst model for star formation, which leads
                # to complication with fact that we defined a set of SFHs...
                # maybe only once we add some noise.
                
                # Shape = formation redshifts or ID
                L = np.zeros_like(self.histories['SFR'][:,0])
                for k in range(self.histories['SFR'].shape[0]):
                    # For each unique object (or object bin), sum emission
                    # over past episodes of star formation.

                    L[k] = np.sum(L_per_msun * self.histories['SFR'][k,iz:] * dt)
                    #L[k] = np.trapz(L_per_msun * self.histories['SFR'][k,iz:],
                    #    dx=dt[0:-1])
                        
            else:
                iz = np.argmin(np.abs(z - self.tab_z))   
                # In this case, the time-stepping is different for each 
                # trajectory. 
                #tnow = self.cosm.t_of_z(z) / s_per_yr
                znow = self.tab_z[iz]
                tnow = self.tab_t[iz]
                
                # In this case, can do some stuff up-front.
                if self.pf['pop_update_dt'].startswith('native'):
                    native_sampling = True
                    all_tarr = self.histories['tobs'][-1::-1]
                    all_zarr = self.histories['zobs'][0][-1::-1]
                    all_dt = np.diff(all_tarr)
                    all_dz = np.diff(all_zarr)
                    all_SFR = self.histories['SFR'][:,-1::-1]
                    imax = len(all_tarr)
                    inow = np.argmin(np.abs(all_tarr - tnow))
                    if all_tarr[inow] < tnow:
                        inow += 1
                else:
                    native_sampling = False                                    
                    
                times = self.src.times    
                    
                # Loop over all objects.
                L = np.zeros_like(self.histories['zobs'][:,0])
                #corr = np.ones_like(L)
                for k in range(self.histories['zobs'].shape[0]):
                    
                    # This galaxy formed after redshift of interest.
                    if not native_sampling:
                        if np.all(self.histories['zobs'][k] < z):
                            print('should this only happen for non-native dt?')
                            continue
                    
                    # In ascending time, descending redshift.
                    # First redshift element is zform, arrays are filled until
                    # object(s) reach z=0 (i.e., not necessarily the last element
                    # of the array).
                    
                    if not native_sampling:
                        tarr = self.histories['tobs'][k] # [yr]
                        zarr = self.histories['zobs'][k]
                        SFR  = self.histories['SFR'][k]
                        Z = np.maximum(self.histories['Z'][k], 1e-3)
                        
                        if not np.all(np.diff(tarr) > 0):
                            tarr = tarr[-1::-1]
                            zarr = zarr[-1::-1]
                            SFR  = SFR[-1::-1]   
                            
                        _imax = np.argwhere(tarr == 0)
                        if len(_imax) > 0:
                            imax = min(_imax[0]) - 1
                        else:
                            imax = len(tarr)    
                            
                        inow = np.argmin(np.abs(tarr - tnow))
                        if tarr[inow] < tnow:
                            inow += 1    
                              
                    else:
                        tarr = all_tarr
                        zarr = all_zarr
                        SFR = all_SFR[k]
                                                
                    # Avoid diff-ing on each iteration if we can
                    if native_sampling:
                        dt = all_dt[0:imax+1]
                        dz = all_dz[0:imax+1]
                    else:
                        dt = np.diff(tarr[0:imax+1])
                        dz = np.diff(zarr[0:imax+1])
                    
                    # In order to get ages, need to invoke current redshift.
                    _ages = (tnow - tarr[0:imax+1]) / 1e6
                    # Hack out elements beyond current observed redshift 
                    ages = _ages[0:inow+1]
                    
                    ## 
                    # Optional: obscuration
                    ##
                    if self.pf['pop_fobsc'] in [0, None]:
                        corr = 1.
                    else:
                        _M = self.histories['Mh'][k]
                        
                        # Means obscuration refers to fractional dimming of individual 
                        # objects
                        if self.pf['pop_fobsc_by'] == 'lum':
                            fobsc = self.guide.fobsc(z=znow, Mh=_M[iz])
                            corr = (1. - fobsc)
                        else:
                            raise NotImplemented('help')
                                 
                    #print(k, z, tnow / 1e7, ages / 1e7)
                    #raw_input('<enter>')
                    
                    if len(ages) == 0:
                        print('ages has no elements!', k)
                        continue
                                        
                    # Need to be careful about interpolating within last
                    # dynamical time.
                    if self.pf['pop_enrichment']:

                       #L_per_msun = self.L_of_Z_t(wave)(np.log10(ages), 
                       #    np.log10(Z[0:inow+1]))
                            
                        # spline wants output to be 2-D.
                        # we just have a time series of (ages, Z)
                        # sooo....
                        
                        logA = np.log10(ages)
                        logZ = np.log10(Z[0:inow+1])
                        
                        spl = self.L_of_Z_t(wave)
                        L_per_msun = [spl(logA[w], logZ[w]).squeeze() \
                            for w in range(logA.size)]
                        L_per_msun = np.array(L_per_msun)
                            
                        
                        #\
                        ##    [self.L_of_Z_t(wave)(np.log10(age), np.log10(Z)) \
                        #        for age in ages]
                        #L_per_msun = np.array(L_per_msun)        
                    else:        
                        L_per_msun = np.interp(ages, times, 
                            self.src.L_per_SFR_of_t(wave))
                            
                            
                    L_per_msun *= corr
                            
                    #_w = np.ones_like(L_per_msun)
                    
                    if not native_sampling:
                        raise NotImplemented('Do we need to modify weight for last time bin?')
                    
                    #_w[-1] = (tarr[inow+1] - tnow) / dt[-1]
                    #print(k, dt[0:inow-1], _w, L_per_msun, self.histories['SFR'][k,0:inow])

                    # Fix last chunk of weights?

                    #L[k] = np.trapz(L_per_msun * SFR[0:inow+1],
                    #    dx=dt[0:inow])
                    L[k] = np.sum(L_per_msun * SFR[0:inow+1] * dt[0:inow+1])

                self._L = L
                        
            # First dimension is formation redshift, second is redshift/time.
            # As long as we're not destroying halos, just take last timestep.
            w = self.histories['nh'][:,0]

        else:
            
            assert not self.pf['pop_ssp'], \
                "Should not have pop_ssp==True if pop_aging==False."
            
            # All star formation rates at this redshift, 'marginalizing' over
            # formation redshift.
            iz = np.argmin(np.abs(z - self.tab_z))
            sfr = self.histories['SFR'][:,iz]

            # Note: if we knew ahead of time that this was a Cohort population,
            # we could replace ':' with 'iz:' in this slice, since we need
            # not consider objects with formation redshifts < current redshift.
            L_per_sfr = self.src.L_per_sfr(wave)
            L = L_per_sfr * sfr

            # First dimension is formation redshift, second is redshift
            # at which we've got some SFR.
            w = self.histories['nh'][:,iz]

        MAB = self.magsys.L_to_MAB(L, z=z)

        #w *= 1. / np.diff(MAB)

        # Need to modify weights, since the mapping from Mh -> L -> mag
        # is not linear.
        #w *= np.diff(self.histories['Mh'][:,iz]) / np.diff(L)

        # Should assert that max(MAB) is < max(MAB)
        
        # If L=0, MAB->inf. Hack these elements off if they exist.
        # This should be a clean cut, i.e., there shouldn't be random
        # spikes where L==0, all L==0 elements should be a contiguous chunk.
        Misok = L > 0
        
        if type(x) != np.ndarray:
            _x = np.arange(-28, -5.8, 0.2)
        else:
            _x = x    
                                        
        hist, bin_edges = np.histogram(MAB[Misok==1], 
            weights=w[Misok==1], 
            bins=bin_c2e(_x), density=True)
        
        bin_c = bin_e2c(bin_edges)
                    
        N = np.sum(w)
        
        phi = hist * N
        
        self._cache_lf_[z] = bin_c, phi
        
        return self._cache_lf(z, x)
        
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
                        
        if type(self.pf['pop_histories']) is str:
            
            if self.pf['pop_histories'].endswith('.pkl'):
                f = open(self.pf['pop_histories'], 'rb')
                prefix = self.pf['pop_histories'].split('.pkl')[0]
            else:
                f = open(self.pf['pop_histories']+'.pkl', 'rb')
                prefix = self.pf['pop_histories']
            
            zall, traj_all = pickle.load(f)
            f.close()
            
            hist = traj_all
            hist['zform'] = zall
            hist['zobs'] = np.array([zall] * hist['nh'].shape[0])
            
            #with open(prefix+'.parameters.pkl', 'rb') as f:
            #    pars = pickle.load(f)
            
            ##
            # CHECK FOR MATCH IN PARAMETERS THAT MATTER.
            # IF SFR PARAMETERS ARE DIFFERENT, cleave off SFHs to force re-run.
            ##
            #mars_ok = 1
            #for par in pars_affect_mars:
            #
            #    ok = pars[par] == self.pf[par]
            #    if not ok:
            #        print("Mismatch in saved histories: {}".format(par))
            #
            #    mars_ok *= ok
            #    
            #sfhs_ok = 1
            #for par in pars_affect_sfhs:
            #
            #    ok = pars[par] == self.pf[par]
            #    if not ok:
            #        print("Mismatch in saved histories: {}".format(par))
            #
            #    sfhs_ok *= ok    
            #
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
        
        