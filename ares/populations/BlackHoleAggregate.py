"""

BlackHoleAggregate.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Sat Mar 17 13:38:58 PDT 2018

Description: 

"""

import numpy as np
from scipy.integrate import ode
from .Halo import HaloPopulation
from ..util.Math import interp1d
from ..physics.Constants import G, g_per_msun, m_p, sigma_T, c, rhodot_cgs, \
    rho_cgs, s_per_myr, t_edd
 

class BlackHoleAggregate(HaloPopulation):
    def __init__(self, **kwargs):
        """
        Initializes a GalaxyPopulation object (duh).
        """

        # This is basically just initializing an instance of the cosmology
        # class. Also creates the parameter file attribute ``pf``.
        HaloPopulation.__init__(self, **kwargs)
        
    @property
    def _frd(self):
        if not hasattr(self, '_frd_'):
            pass

        return self._frd_    
    
    @_frd.setter
    def _frd(self, value):
        self._frd_ = value
        
    def _frd_func(self, z):
        # This is a cheat so that the FRD spline isn't constructed
        # until CALLED. Used only for linking.                        
        return self.FRD(z)
        
    def FRD(self, z):
        """
        Compute BH formation rate density.
        
        Units = cgs 
        
        A bit odd to have this in mass units (would rather #/time/vol) but
        for the fcoll model one doesn't need to invoke a BH mass, hence
        the difference between the linked FRD model and the fcoll model below.
        
        """
        
        on = self.on(z)
        if not np.any(on):
            return z * on
        
        # SFRD given by some function
        if self.is_link_sfrd:
            # _frd is in # / cm^3 / s, so we convert to g / cm^3 / s
            return self._frd(z) * on * self.pf['pop_mass'] * g_per_msun \
                * self.pf['pop_bh_seed_eff']
        
        # Otherwise, use dfcolldt model (all we know right now).
        bhfrd = self.pf['pop_bh_seed_eff'] * self.cosm.rho_b_z0 * self.dfcolldt(z)

        return bhfrd
        
    @property
    def Ledd_1Msun(self):
        # Multiply by BH mass density to get luminosity in erg/s
        return self.pf['pop_eta'] * 4.0 * np.pi * G * g_per_msun * m_p \
            * c / sigma_T
    
    def _BHGRD(self, z, rho_bh):
        """
        rho_bh in Msun / cMpc^3 / yr.
        """

        # Convert to Msun / cMpc^3
        new = self.FRD(z) * rhodot_cgs
        
        # Currently in Msun / cMpc^3 / sec
        old = self.pf['pop_fduty'] \
            * rho_bh[0] * 4.0 * np.pi * G * m_p \
                / sigma_T / c / self.pf['pop_eta']
                                        
        # In Msun / cMpc^3 / dz
        return -np.array([new + old]) * self.cosm.dtdz(z)
        
    @property
    def _BHMD(self):
        if not hasattr(self, '_BHMD_'):
            
            z0 = min(self.halos.tab_z.max(), self.zform)
            zf = max(float(self.halos.tab_z.min()), self.pf['final_redshift'])
            zf = max(zf, self.zdead)
            
            if self.pf['sam_dz'] is not None:
                dz = self.pf['sam_dz'] * np.ones_like(self.halos.tab_z)
                zfreq = int(round(self.pf['sam_dz'] / np.diff(self.halos.tab_z)[0], 0))
            else:
                dz = np.diff(self.halos.tab_z)
                zfreq = 1
   
            # Initialize solver
            solver = ode(self._BHGRD).set_integrator('lsoda', nsteps=1e4, 
                atol=self.pf['sam_atol'], rtol=self.pf['sam_rtol'],
                with_jacobian=False)
                
            in_range = np.logical_and(self.halos.tab_z >= zf, 
                self.halos.tab_z <= z0)
            zarr = self.halos.tab_z[in_range][::zfreq]
            Nz = zarr.size

            # y in units of Msun / cMpc^3 
            #Mh0 = #self.halos.Mmin(z0) 
            #if self.is_link_sfrd:
            #    rho_bh_0 = 1e-10
            #else:        
            #    rho_bh_0 = self.halos.fcoll_2d(z0, 5.) * self.pf['pop_bh_seed_eff'] \
            #        * self.cosm.rho_b_z0 * rho_cgs
                    
            solver.set_initial_value(np.array([0.0]), z0)


            zflip = zarr[-1::-1]
            
            rho_bh = []
            redshifts = []
            for i in range(Nz):
                
                if i == Nz - 1:
                    break
                                                                
                redshifts.append(zflip[i])
                rho_bh.append(solver.y[0])
                
                z = redshifts[-1]
                                            
                solver.integrate(solver.t-dz[i])
                
            z = np.array(redshifts)[-1::-1]
            
            # Convert back to cgs (internal units)
            rho_bh = np.array(rho_bh)[-1::-1] / rho_cgs
            
            self._z = z
            self._rhobh = rho_bh
            
            tmp = interp1d(z, rho_bh, 
                kind=self.pf['pop_interp_sfrd'],
                bounds_error=False, fill_value=0.0)

            self._BHMD_ = lambda z: tmp(z)
                
        return self._BHMD_       
        
    def BHMD(self, z):
        """
        Compute the BH mass density.
        """
        
        return self._BHMD(z)

    def ARD(self, z):
        """
        Compute the BH accretion rate density.
        """
        
        tacc = self.pf['pop_eta'] * t_edd / self.pf['pop_fduty']
        return self.BHMD(z) / tacc

    def Emissivity(self, z, E=None, Emin=None, Emax=None):
        """
        Compute the emissivity of this population as a function of redshift
        and rest-frame photon energy [eV].
    
        ..note:: If `E` is not supplied, this is a luminosity density in the
            (Emin, Emax) band.
    
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
    
        if self.pf['pop_sed_model'] and (Emin is not None) and (Emax is not None):
            if (Emin > self.pf['pop_Emax']):
                return 0.0
            if (Emax < self.pf['pop_Emin']):
                return 0.0    
    
        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        rhoL = on * self.Ledd_1Msun * self.BHMD(z) / g_per_msun
                
        ## Convert from reference band to arbitrary band
        rhoL *= self._convert_band(Emin, Emax)
        #if (Emax is None) or (Emin is None):
        #    pass
        #elif Emax > 13.6 and Emin < self.pf['pop_Emin_xray']:
        #    rhoL *= self.pf['pop_fesc']
        #elif Emax <= 13.6:
        #    rhoL *= self.pf['pop_fesc_LW']    
    
        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL
    
    def NumberEmissivity(self, z, E=None, Emin=None, Emax=None):
        return self.Emissivity(z, E=E, Emin=Emin, Emax=Emax) / (E * erg_per_ev)
    
    def LuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the luminosity density in the (Emin, Emax) band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Luminosity density in erg / s / c-cm**3.
    
        """
    
        return self.Emissivity(z, Emin=Emin, Emax=Emax)
    
    def PhotonLuminosityDensity(self, z, Emin=None, Emax=None):
        """
        Return the photon luminosity density in the (Emin, Emax) band.
    
        Parameters
        ----------
        z : int, flot
            Redshift of interest.
    
        Returns
        -------
        Photon luminosity density in photons / s / c-cm**3.
    
        """
    
        rhoL = self.LuminosityDensity(z, Emin, Emax)
        eV_per_phot = self._get_energy_per_photon(Emin, Emax)
    
        return rhoL / (eV_per_phot * erg_per_ev)    
    