"""

GalaxyEnsemble.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jul  6 11:02:24 PDT 2016

Description: 

"""

import numpy as np
from .Halo import HaloPopulation
from .GalaxyCohort import GalaxyCohort
from ..util.Stats import bin_e2c, bin_c2e
from ..util import ProgressBar
from ..physics.Constants import rhodot_cgs, s_per_yr, s_per_myr
#from scipy.interpolate import RectBivariateSpline

class GalaxyEnsemble(HaloPopulation):
    
    def __init__(self, **kwargs):
        # May not actually need this...
        HaloPopulation.__init__(self, **kwargs)
        
    def __dict__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        
        raise NotImplemented('help!')
        
    @property
    def dust(self):
        if not hasattr(self, '_dust'):
            self._dust = DustCorrection(**self.pf)
        return self._dust    
    
    @property
    def tab_z(self):
        if not hasattr(self, '_tab_z'):
            h = self.histories
        return self._tab_z
            
    @tab_z.setter
    def tab_z(self, value):
        self._tab_z = value
        
    def SFRD(self, z):
        """
        Will convert to internal cgs units.
        """
        
        iz = np.argmin(np.abs(z - self.tab_z))
        sfr = self.histories['SFR'][:,iz]
        w = self.histories['w'][:,iz]
        
        # Really this is the number of galaxies that formed in a given
        # differential redshift slice.
        
        #Mh = self.histories['Mh'][:,iz]
        
        #dw = np.diff(_w) / np.diff(Mh)
        
        # The 'bins' in the first dimension have some width...
        
        return np.sum(sfr * w) / rhodot_cgs
        #return np.trapz(sfr[0:-1] * dw, dx=np.diff(Mh)) / rhodot_cgs

    def tile(self, arr, thin, renorm=False):
        
        if arr is None:
            return None
        
        if thin == 0:
            return arr.copy()

        assert thin % 1 == 0

        new = np.tile(arr, (int(thin), 1)) 
        
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
        dt = 10.
        t = np.arange(30., 2000+dt)[-1::-1]
        z = self.cosm.z_of_t(t * s_per_myr)
        N = z.size
        
        # In this case we need to thin before generating SFHs?
        
        if thin == 0:
            thin = 1
            
        sfr0_func = lambda zz: guide.SFR(z=zz, Mh=guide.Mmin(zz))
        
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

        raw = {'z': z, 'SFR': sfr, 'nh': nh, 'w': nh, 
            'Mh': Mh, 'MAR': None, 'SFE': None}

        return zall, raw
        
    def _gen_kelson_history(self, sfr0, t, sigma=0.3):
        sfr = np.zeros_like(t)
        sfr[0] = sfr0
        for i in range(1, t.size):
            sfr[i] = sfr[i-1] * np.random.lognormal(mean=0., sigma=sigma)
        
        return np.array(sfr)
        
    def _gen_deterministic_history(self):
        pass
        
    def _gen_SAMlike_history(self):
        pass
                
    @property
    def histories(self):
        if not hasattr(self, '_histories'):
            
            hist = self.pf['pop_histories']
            guide = self.pf['pop_guide_pop']
            thin = self.pf['pop_thin_hist']
            
            sigma_sfr = self.pf['pop_scatter_sfr']
            sigma_sfe = self.pf['pop_scatter_sfe']
            sigma_mar = self.pf['pop_scatter_mar']
            sigma_env = self.pf['pop_scatter_env']
                        
            # Just read in histories in this case.
            if type(hist) is dict:
            
                raw = hist
                
                zall = raw['z']
                self.tab_z = raw['z']
                self._histories = raw
                
                assert ('SFR' in raw) or ('MAR' in raw)
                    
            # In this case, some building is required. 
            elif hist == 'kelson':
                
                # If we're thinning, will update SFRs later.
                zall, raw = self.gen_kelson(guide, thin)
                
                self.tab_z = zall
                self._histories = raw
                return raw
                
            else:

                assert isinstance(guide, GalaxyCohort)

                # Is this going to be MCMC-able? No...
                # How can we enable scatter to be free parameters without
                # running into circular import?
                # Well, as long as the guide pop doesn't change, we can 
                # parameterize the scatter.

                zall, raw = guide.Trajectories()
                

            have_sfr = False
            if ('SFR' in raw) and (sigma_sfe == sigma_mar == 0):
                sfr_raw = raw['SFR']
                have_sfr = True
            else:
                assert 'MAR' in raw
            
            if 'MAR' in raw:
                mar_raw = raw['MAR']
                
            ##
            # May need to construct SFRs from MARs
            ##
            if not have_sfr:
                
                
                # Need to decide whether to use SFE from guide population
                # or PQ or to generate from some distribution.
                
                # zall is 1-D, Mh is 2-D (zform, zobs)
                sfe_raw = guide.SFE(z=zall, Mh=raw['Mh'])
            else:    
                sfe_raw = None
                
                #if 'MAR' in raw and hist != 'kelson':
                #    raise KeyError('Have MAR and SFR. Whaddya wanna do?')
                
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
            
            ##
            # Allow scatter in things
            ##            
            # Just adding scatter in SFR
            if have_sfr:
                sfr = self.tile(sfr_raw, thin)
            
            if sigma_sfr > 0:
                assert have_sfr
                assert not (self.pf['pop_scatter_sfe'] or self.pf['pop_scatter_mar'])                
                sfr += self.noise_lognormal(sfr, sigma_sfr)
            
            # Can add SFE scatter too    
            sfe = self.tile(sfe_raw, thin)
            if sigma_sfe > 0:
                sfe += self.noise_lognormal(sfe, sigma_sfe)
                
            mar = self.tile(mar_raw, thin)
            
            if sigma_env > 0:
                mar *= (1. + self.noise_normal(mar, sigma_env))
                
            if sigma_mar > 0:
                mar += self.noise_lognormal(mar, sigma_mar)
                sfr = sfe * mar * self.cosm.fbar_over_fcdm
            else:
                if not have_sfr:
                    sfr = sfe * mar * self.cosm.fbar_over_fcdm 
            
            
            
            # Things to add: metal-enrichment history (MEH)
            #              : ....anything else?
            
            # Artificial SF shutdown option.
            if self.pf['pop_quench']:
                
                k = np.argmin(np.abs(zall - 8.))
                
                for i, hist in enumerate(sfr):
                    if Mh[i,k] >= guide.Mmin(8.):
                        continue
                    
                    sfr[i,0:k] = 0.0
                       
                   #print('Quenching zf={} at z<={}'.format(zall[i], zall[j]))    

            # SFR = (zform, time (but really redshift))
            # So, halo identity is wrapped up in axis=0
            # In Cohort, formation time defines initial mass and trajectory (in full)
            histories = {'z': zall, 'w': nh, 'SFR': sfr, 'Mh': Mh,
                'MAR': mar, 'SFE': sfe}
                            
            self.tab_z = zall
            self._histories = histories
    
                            
        return self._histories
        
    def get_Ms(self, z):
        iz = np.argmin(np.abs(z - self.tab_z))
        
        # Problem with the redshift array? Too short.
        
        zarr = self.tab_z[iz:]
        # These are bin centers
        #t = self.cosm.t_of_z(zarr) / s_per_yr
        
        dz = np.diff(self.tab_z)[0]
        dt = self.cosm.dtdz(zarr) * dz / s_per_yr 
        
        #dz = np.diff(self.tab_z[iz:])
        #dt = np.diff(t[iz:]) * dz / s_per_yr
        
        Nzf = self.histories['SFR'].shape[0]
        dt_r = np.reshape(np.tile(dt, Nzf), (Nzf, len(zarr)))
        
        # This is (zform, z'). Can neglect zform < z'.
        return np.sum(self.histories['SFR'][:,iz:] * dt_r, axis=1)
        
    def StellarMassFunction(self, z):
        """
        Could do a cumulative sum to get all stellar masses in one pass. 
        
        For now, just do for the redshift requested.
        """
        
        iz = np.argmin(np.abs(z - self.tab_z))
        
        Ms = self.get_Ms(z)

        return Ms, self.histories['w'][:,iz]
        
    def SMHM(self, z):
        iz = np.argmin(np.abs(z - self.tab_z))
        
        return self.get_Ms(z), self.histories['Mh'][:,iz]
        
    def LuminosityFunction(self, z, x, mags=True, wave=1600., band=None):
        """
        Compute the luminosity function from discrete histories.
        """
    
        iz = np.argmin(np.abs(z - self.tab_z))
        
        if self.pf['pop_enrichment']: 
            raise NotImplemented('help!')
        
        # Care required!
        if self.pf['pop_aging']:   
            
            assert self.pf['pop_ssp']
                     
            zarr = self.tab_z[iz:]
            
            # Array of times corresponding to all z' > z.
            tarr = self.cosm.t_of_z(zarr) / s_per_myr
            
            # Ages of stellar populations formed at each z' > z
            ages = tarr[0] - tarr
            
            dz = np.diff(self.tab_z)[0]
            
            # Eventually shouldn't be necessary
            #assert np.allclose(dz, self.pf['sam_dz']), \
            #    "dz={}, sam_dz={}".format(dz, self.pf['sam_dz'])

            # in years
            dt = dz * self.cosm.dtdz(zarr) / s_per_yr
            
            # Is this all just an exercise in careful binning?
            # Also, within a bin, if we assume a constant SFR, there could
            # be a non-negligible age gradient....ugh.
            
                        
            # Interpolate to find luminosity as function of age for
            # all ages we've got so far.
            il = np.argmin(np.abs(wave - self.src.wavelengths))
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
                
                L[k] = np.trapz(L_per_msun * self.histories['SFR'][k,iz:],
                    dx=dt[0:-1])
                                                
            # First dimension is formation redshift, second is redshift
            # at which we've got some SFR.
            w = self.histories['w'][:,iz]

        else:
            # All star formation rates at this redshift, 'marginalizing' over
            # formation redshift.
            sfr = self.histories['SFR'][:,iz]

            # Note: if we knew ahead of time that this was a Cohort population,
            # we could replace ':' with 'iz:' in this slice, since we need
            # not consider objects with formation redshifts < current redshift.
            L_per_sfr = self.src.L_per_sfr(wave)
            L = L_per_sfr * sfr

            # First dimension is formation redshift, second is redshift
            # at which we've got some SFR.
            w = self.histories['w'][:,iz]

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
                    
        hist, bin_edges = np.histogram(MAB[Misok==1], 
            weights=w[Misok==1], 
            bins=bin_c2e(x), density=True)
        
        N = np.sum(w)
            
        return hist * N
        
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
        
        