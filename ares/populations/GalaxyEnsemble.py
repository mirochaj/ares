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
        z2d = np.array([zall] * nh.shape[0])
        histories = {'z2d': z2d, 'Mh': Mh, 
            'MAR': mar, 'nh': nh}
            
        # Add in formation redshifts to match shape (useful after thinning)
        histories['zthin'] = self.tile(zall, thin)
        
        histories['z'] = zall
        histories['t'] = np.array(map(self.cosm.t_of_z, zall)) / s_per_myr
                        
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
        
    def _gen_stars(self, z, Mh, Mg, E_SN=0.0):
        """
        Take draws from cluster mass function until stopping criterion met.
        
        Return the amount of mass formed in this burst.
        """
        
        E_h = self.halos.BindingEnergy(Mh, z)
        
        # Statistical approach from here on out.
        Ms = 0.0
        N_sn = 0
        
        fstar_gmc = self.pf['pop_fstar_cloud']
        
        if self.pf['pop_delay_feedback']:
            Efb = E_SN * 1.
            print('Preemptive feedback', Efb >= E_h)
        else:
            Efb = 0.0
        
        Mavg = 1e-1
        fsn = 1e-2
        NSN_per_M = fsn / Mavg
        
        # Form clusters until we use all the gas or blow it all out.    
        while Efb < E_h:

            r = np.random.rand()
            Mc = np.interp(r, self.tab_cdf, self.tab_Mcl)

            # Poisson-ify the SN feedback
            
            
            # Expected number of SNe if we were forming lots of clusters.
            lam = Mc * NSN_per_M

            Nsn = np.random.poisson(lam)
                                    
            # May want to hang on to few bins worth of Ms to track
            # SNe, HMXBs, etc.
            
            #gas_avail = (Mr * fc_r[i] + Ma * fc_i[i]) #* fstar
            gas_limited = False
            if Ms + Mc >= Mg * fstar_gmc:                
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
                Efb += 1e51 * Nsn * 1.0

        return Ms, N_sn
        
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
        nh = halo['nh'][-1::-1]
        
        # Short-hand
        fb = self.cosm.fbar_over_fcdm
        Mg_s = fb * Mh_s
        Nt = len(t)
        
        assert np.all(np.diff(t) >= 0)

        zform = max(z[Mh_s>0])

        SFR = np.zeros_like(Mh_s)
        Ms  = np.zeros_like(Mh_s)
        Msc = np.zeros_like(Mh_s)
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
            
            Mavg = 1e-1
            fsn = 1e-2
            NSN_per_M = fsn / Mavg
            
            ##
            # Override switch to smooth inflow-driven star formation model.s
            ##
            if E_h > (1e51 * self.pf['pop_force_equilibrium']):
                # Assume 1e51 * SNR * dt = 1e51 * SFR * SN/Mstell * dt = E_h
                # SFR = E_h / 1e51 / (SN/Ms) / dt
                SFR[i]  = E_h / 1e51 / NSN_per_M / dt
                Ms[i+1] = 0.5 * (SFR[i] + SFR[i-1]) * dt
                Mg[i+1] = Mg[i] + Macc - Ms[i+1]
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
            Msc[i+1] = Msc[i] + Mnew
        
        keep = np.ones_like(z)#np.logical_and(z > zobs, z <= zform)
        
        data = \
         { 
          'SFR': SFR[keep==1], 
          'MAR': MAR_s[keep==1],
          'Mg': Mg[keep==1], 
          'Ms': Msc[keep==1], 
          'Mh': Mh_s[keep==1], 
          'nh': nh[keep==1],
          'z': z[keep==1], 
          't': t[keep==1],
          'zthin': halo['zthin'][-1::-1],
         }       
                
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
        if self.pf['pop_stochastic']:
            
            fields = ['SFR', 'MAR', 'Mg', 'Ms', 'Mh', 'nh']
            num = halos['Mh'].shape[0]
            
            hist = {key:np.zeros_like(halos['Mh']) for key in fields}
            
            for i in range(num):                
                #halo = {key:halos[key][i] for key in keys}
                halo = {'t': halos['t'], 'z': halos['z'], 'zthin': halos['zthin'],
                    'Mh': halos['Mh'][i], 'nh': halos['nh'][i]}
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
 
        # Setup redshift and time arrays.
        #if '/' in self.pf['pop_update_dt']:
        #    _pre, _step = self.pf['pop_update_dt'].split('/')
        #    step = -int(_step)
        #else:
        #    step = -1
            
        #zall = zarr[-1::step]
        
        #assert np.allclose(zarr, self.tab_z)
        
        #tall = self.tab_t[-1::step]
        #zobs = halos['zobs']
        ##tobs = np.array([tall] * halos['nh'].shape[0])
        #tobs = tall#self.cosm.t_of_z(zobs) / s_per_yr

        # In this model we only use 'thin' to set size of arrays
                
        # These will be (zform, zarr).
        # We need to reverse the ordering of the second dimension
        # to be in ascending *time*. We'll switch it back at the end.
        #Nz = len(zall)
        #_Mh = halos['Mh'][:,-1::step]
        #_MAR = halos['MAR'][:,-1::step]
        #_SFE = self.guide.SFE(z=zall, Mh=_Mh)#halos['SFE'][:,-1::step]
        #shape = halos['nh'].shape
        #
        #fbar = self.cosm.fbar_over_fcdm 
        
        # Flip arrays to be in ascending time.
        z = halos['z'][-1::-1]
        z2d = halos['z2d'][:,-1::-1]
        t = halos['t'][-1::-1]
        Mh = halos['Mh'][:,-1::-1]
        nh = halos['nh'][:,-1::-1]
        MAR = halos['MAR'][:,-1::-1]
        SFE = self.guide.SFE(z=z2d, Mh=Mh)
        
        # Short-hand
        fb = self.cosm.fbar_over_fcdm
        
        # 't' is in Myr
        dt = np.abs(np.diff(t)) * 1e6
        
        # Integrate (crudely) mass accretion rates
        #_Mint = cumtrapz(_MAR[:,:], dx=dt, axis=1)
        _MAR_c = 0.5 * (np.roll(MAR, -1, axis=1) + MAR)
        _Mint = np.cumsum(_MAR_c[:,1:] * dt, axis=1)
                        
        # Increment relative to initial masses. Need to put in 
        # a set of zeros for first element to get shape right.
        _fill = np.zeros((MAR.shape[0], 1))
        #Mh = Mh0 + np.concatenate((_fill, _Mint), axis=1)
        
        #Mh = _Mh
        
        _SFE_c = 0.5 * (np.roll(SFE, -1, axis=1) + SFE)
        SFR = MAR * SFE * fb
        SFR_c = _MAR_c * _SFE_c * fb
        
        MGR = MAR * fb
        MGR_c = _MAR_c * fb
        
        # Stellar mass
        fml = (1. - self.pf['pop_mass_yield'])
        Ms0 = np.zeros((SFR.shape[0], 1))
        Msj = np.cumsum(SFR_c[:,1:] * dt * fml, axis=1)
        Ms = np.concatenate((Ms0, Msj), axis=1)
        
        # Metal mass
        fZy = self.pf['pop_mass_yield'] * self.pf['pop_metal_yield']
        MZ0 = np.zeros((SFR.shape[0], 1))
        MZj = np.cumsum(SFR_c[:,1:] * dt * fZy, axis=1)
        #Msj = cumtrapz(SFR[:,:], dx=dt, axis=1)
        #Msj = 0.5 * cumtrapz(SFR * tall, x=np.log(tall), axis=1)
        MZ = np.concatenate((MZ0, MZj), axis=1)
        
        # Gas mass
        Mh0 = self.guide.Mmin(z)
        Mg0 = Mh0 * fb
        Mg = Mg0 + np.concatenate((_fill, _Mint * fb), axis=1)
        #Mgj = np.cumsum(MGR_c[:,1:] * dt, axis=1)
        #Msj = cumtrapz(SFR[:,:], dx=dt, axis=1)
        #Msj = 0.5 * cumtrapz(SFR * tall, x=np.log(tall), axis=1)
        #Mg = np.concatenate((Mg0, Mgj), axis=1)
                            
        results = \
        {
         'nh': nh,#[:,-1::-1],
         'Mh': Mh,#[:,-1::-1],
         't': t,#[-1::-1],
         'z': z,#[-1::-1],
         'zthin': halos['zthin'][-1::-1],
         'z2d': z2d,
         'SFR': SFR,#[:,-1::-1],
         'Mg': Mg,#[:,-1::-1],
         'Ms': Ms,#[:,-1::-1],
         'MZ': MZ,#[:,-1::-1],
         'Mh': Mh,#[:,-1::-1],
         #'Z': MZ[:,-1::-1] / Mg[:,-1::-1],
        }
                
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
        Yield spectrum for a single galaxy.
        
        Parameters
        ----------
        hist : dict
            Dictionary containing the trajectory of a single object. Most
            importantly, must contain 'SFR', as a function of *increasing
            time*, stored in the key 't'.
            
        For example, hist could be the output of a call to _gen_galaxy_history.
            
        """
                
        if zobs is None:
            izobs = None
        else:
            izobs = np.argmin(np.abs(zobs - hist['z']))
        
        # Must be supplied in increasing time order, decreasing redshift.
        assert np.all(np.diff(hist['t']) >= 0)
                        
        #if not np.any(hist['SFR'] > 0):
        #    print('nipped this in the bud')
        #    return hist['t'], hist['z'], np.zeros_like(hist['z'])
        #
        izform = 0#min(np.argwhere(hist['Mh'] > 0))[0]
                                
        ##
        # First. Simple case without stellar population aging.
        ##
        if not self.pf['pop_aging']:
            assert not self.pf['pop_ssp'], \
                "Should not have pop_ssp==True if pop_aging==False."
            
            L_per_sfr = self.src.L_per_sfr(wave)
            return L_per_sfr * hist['SFR'][izform:izobs+1]
        
        ##
        # Second. Harder case where aging is allowed.
        ##          
        assert self.pf['pop_ssp']
                
        # SFH        
        SFR  = hist['SFR'][izform:izobs+1]
        tarr = hist['t'][izform:izobs+1] # in Myr
        zarr = hist['z'][izform:izobs+1]
        dt = np.diff(tarr) * 1e6
        
        #if SFR[np.argmin(hist['z'][izform:izobs+1] - 10.)] > 1e-2:
        #    import matplotlib.pyplot as pl
        #    pl.figure(10)
        #    pl.semilogy(zarr, SFR)
        #    raw_input('<enter>')
        
        
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
        
        # These are kept in ascending redshift just to make life difficult.
        raw = self.histories
                        
        keys = raw.keys()
        Nt = raw['t'].size
        Nh = raw['Mh'].shape[0]
        
        izobs = np.argmin(np.abs(raw['z'] - z))
        
        Lt = np.zeros((Nh, izobs+1))
        corr = np.ones_like(Lt)
        for i in range(Nh):
                        
            if not np.any(raw['Mh'][i] > 0):
                print('hey', i)
                continue
                
            hist = {'t': raw['t'], 'z': raw['z'],
                'SFR': raw['SFR'][i], 'Mh': raw['Mh'][i]}

            izform = 0#min(np.argwhere(raw['Mh'][i][-1::-1] > 0))[0]
            
            # Must supply in time-ascending order
            zarr, tarr, _L = self.SpectralSynthesis(hist, 
                zobs=z, wave=wave)

            Lt[i,izform+1:] = _L
            
            ## 
            # Optional: obscuration
            ##
            if self.pf['pop_fobsc'] in [0, 0.0, None]:
                pass
            else:
                _M = raw['Mh'][i,izform:izobs+1]
                
                # Means obscuration refers to fractional dimming of individual 
                # objects
                if self.pf['pop_fobsc_by'] == 'lum':
                    fobsc = self.guide.fobsc(z=zarr, Mh=_M)                    
                    corr[i] = (1. - fobsc)
                else:
                    raise NotImplemented('help')
            
        # Grab number of halos from last timestep.
        w = raw['nh'][:,-1]
                        
        # Just hack out the luminosity *now*.
        L = Lt[:,-1] * corr[:,-1]
        
        MAB = self.magsys.L_to_MAB(L, z=z)
        
        # Need to modify weights, since the mapping from Mh -> L -> mag
        # is not linear.
        #w *= np.diff(self.histories['Mh'][:,iz]) / np.diff(L)
        
        # Should assert that max(MAB) is < max(MAB)
        
        # If L=0, MAB->inf. Hack these elements off if they exist.
        # This should be a clean cut, i.e., there shouldn't be random
        # spikes where L==0, all L==0 elements should be a contiguous chunk.
        Misok = L > 0
        
        if type(x) != np.ndarray:
            _x = np.arange(-28, 0.0, 0.2)
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
        
        