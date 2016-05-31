"""

GalaxyMZ.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan 13 09:49:00 PST 2016

Description: 

"""

import re
import numpy as np
from ..util import read_lit
from types import FunctionType
from ..util.ParameterFile import par_info
from .GalaxyAggregate import GalaxyAggregate
from scipy.optimize import fsolve, curve_fit
from ..phenom.DustCorrection import DustCorrection
from scipy.integrate import quad, simps, cumtrapz, ode
from ..physics.RateCoefficients import RateCoefficients
from scipy.interpolate import interp1d, RectBivariateSpline
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..phenom.HaloProperty import ParameterizedHaloProperty, \
    Mh_dep_parameters
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc, G, m_p, \
    k_B, h_p, erg_per_ev, ev_per_hz

try:
    from scipy.misc import derivative
except ImportError:
    pass
    
z0 = 9. # arbitrary
    
class GalaxyPopulation(GalaxyAggregate,DustCorrection):
    
    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
    
        return self._magsys
        
    def __getattr__(self, name):
        """
        This gets called anytime we try to fetch an attribute that doesn't
        exist (yet). Right, now this is only used for L1500, Nion, Nlw.
        """
            
        # Indicates that this attribute is being accessed from within a 
        # property. Don't want to override that behavior!
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')
        
        full_name = 'pop_' + name
                
        # Now, possibly make an attribute
        if name not in self.__dict__.keys(): 
            
            try:
                is_php = self.pf[full_name][0:3] == 'php'
            except (IndexError, TypeError):
                is_php = False
    
            if self.sed_tab and (not is_php):
                result = lambda z, M: self.src.__getattribute__(name) \
                    / self.pf['pop_fstar_boost']
            elif type(self.pf[full_name]) in [float, np.float64]:
                result = lambda z, M: self.pf[full_name] \
                        / self.pf['pop_fstar_boost']
            elif is_php:
                tmp = self.get_php_pars(self.pf[full_name]) 
                
                # Correct values that are strings:
                if self.sed_tab:
                    pars = {}
                    for par in tmp:
                        if tmp[par] == 'from_sed':
                            pars[par] = self.src.__getattribute__(name) \
                                / self.pf['pop_fstar_boost']
                        else:
                            pars[par] = tmp[par]            
                else:
                    pars = tmp            
                    
                inst = ParameterizedHaloProperty(**pars)
                result = lambda z, M: inst.__call__(z, M) \
                        / self.pf['pop_fstar_boost']          
        
            else:
                raise TypeError('dunno how to handle this')
                
            self.__dict__[name] = result
    
        return self.__dict__[name]

    def N_per_Msun(self, Emin, Emax):
        """
        Compute photon luminosity in band of interest per unit SFR for 
        all halos.
        
        Returns
        -------
        In units of photons/s/Msun.
        
        """
        if not hasattr(self, '_N_per_Msun'):
            self._N_per_Msun = {}

        # If we've already figured it out, just return    
        if (Emin, Emax) in self._N_per_Msun:    
            return self._N_per_Msun[(Emin, Emax)]

        # Otherwise, calculate what it should be
        if (Emin, Emax) == (13.6, 24.6):

            # Should be based on energy at this point, not photon number
            self._N_per_Msun[(Emin, Emax)] = self.Nion(None, self.halos.M) \
                * self.cosm.b_per_g * g_per_msun #/ s_per_yr
        elif (Emin, Emax) == (10.2, 13.6):
            self._N_per_Msun[(Emin, Emax)] = self.Nlw(None, self.halos.M) \
                * self.cosm.b_per_g * g_per_msun #/ s_per_yr
    
        return self._N_per_Msun[(Emin, Emax)]
    
    #def rho_L(self, z, Emin, Emax):
    #    """
    #    Compute the luminosity density in some bandpass at some redshift.
    #    
    #    Returns
    #    -------
    #    Luminosity density in units of erg / s / (comoving cm)**3.
    #    """
    #    
    #    if not hasattr(self, '_rho_L'):
    #        self._rho_L = {}
    #    
    #    # If we've already figured it out, just return    
    #    if (Emin, Emax) in self._rho_L:    
    #        return self._rho_L[(Emin, Emax)]    
    #        
    #    tab = np.zeros(self.halos.Nz)
    #    
    #    for i, z in enumerate(self.halos.z):
    #        integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
    #            * self.L_per_SFR(Emin=Emin, Emax=Emax)
    #    
    #        tot = np.trapz(integrand, x=self.halos.lnM)
    #        cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
    #        
    #        self._sfrd_tab[i] = tot - \
    #            np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
    #        
    #    self._sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3    
    #
    #    self._rho_L[(Emin, Emax)] = interp1d(self.halos.z, tab, kind='cubic')
    
    def rho_N(self, z, Emin, Emax):
        """
        Compute the photon luminosity density in some bandpass at some redshift.
        
        Returns
        -------
        Luminosity density in units of photons / s / (comoving cm)**3.
        """
        
        if not hasattr(self, '_rho_N'):
            self._rho_N = {}
        
        # If we've already figured it out, just return    
        if (Emin, Emax) in self._rho_N:    
            return self._rho_N[Emin, Emax](z)
            
        tab = np.zeros(self.halos.Nz)
        
        # For all halos
        N_per_Msun = self.N_per_Msun(Emin=Emin, Emax=Emax)
        
        if (Emin, Emax) == (13.6, 24.6):
            fesc = self.fesc(None, self.halos.M)
        else:
            fesc = 1.
        
        for i, z in enumerate(self.halos.z):
            integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
                * N_per_Msun * fesc * self.fduty(z, self.halos.M)
        
            tot = np.trapz(integrand, x=self.halos.lnM)
            cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
            
            tab[i] = tot - \
                np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
            
        tab *= 1. / s_per_yr / cm_per_mpc**3    
        
        self._rho_N[(Emin, Emax)] = interp1d(self.halos.z, tab, kind='cubic')

        return self._rho_N[(Emin, Emax)](z)
        
    def _sfrd_func(self, z):
        # This is a cheat so that the SFRD spline isn't constructed
        # until CALLED. Used only for tunneling (see `pop_tunnel` parameter). 
        return self.SFRD(z)
        
    @property   
    def SFRD(self):
        """
        Compute star-formation rate density (SFRD).
        """
        
        if not hasattr(self, '_SFRD'):
            self._SFRD = interp1d(self.halos.z, self.sfrd_tab, kind='cubic')

        return self._SFRD
        
    @SFRD.setter
    def SFRD(self, value):
        self._SFRD = value    
    
    @property   
    def SMD(self):
        """
        Compute stellar mass density (SMD).
        """
    
        if not hasattr(self, '_SMD'):
            dtdz = np.array(map(self.cosm.dtdz, self.halos.z))
            self._smd_tab = cumtrapz(self.sfrd_tab[-1::-1] * dtdz[-1::-1], 
                dx=np.abs(np.diff(self.halos.z[-1::-1])), initial=0.)[-1::-1]
            self._SMD = interp1d(self.halos.z, self._smd_tab, kind='cubic')
    
        return self._SMD
    
    @property
    def MAR(self):
        """
        Mass accretion rate onto halos of mass M at redshift z.
    
        ..note:: This is the *DM* mass accretion rate. To obtain the baryonic 
            accretion rate, multiply by Cosmology.fbaryon.
            
        """
        if not hasattr(self, '_MAR'):
            if self.pf['pop_MAR'] is None:
                self._MAR = None
            elif type(self.pf['pop_MAR']) is FunctionType:
                self._MAR = self.pf['pop_MAR']
            elif self.pf['pop_MAR'] == 'pl':
                raise NotImplemented('do this')
            elif self.pf['pop_MAR'] == 'hmf':
                self._MAR = self.halos.MAR_func
            else:
                self._MAR = read_lit(self.pf['pop_MAR']).MAR

        return self._MAR
        
    def iMAR(self, z, source=None):
        """
        The integrated mass (*matter*, i.e., baryons + CDM) accretion rate.
        
        Parameters
        ----------
        z : int, float
            Redshift
        source : str
            Can be a litdata module, e.g., 'mcbride2009'.
            If None, will compute from HMF.
            
        Returns
        -------
        Integrated DM mass accretion rate in units of Msun/yr/cMpc**3.
            
        """    

        if source is not None:
            src = read_lit(source)

            i = np.argmin(np.abs(z - self.halos.z))

            # Integrand: convert MAR from DM MAR to total matter MAR
            integ = self.halos.dndlnm[i] \
                * src.MAR(z, self.halos.M) / self.cosm.fcdm

            Mmin = np.interp(z, self.halos.z, self.Mmin)
            j1 = np.argmin(np.abs(Mmin - self.halos.M))
            if Mmin > self.halos.M[j1]:
                j1 -= 1

            p0 = simps(integ[j1-1:], x=self.halos.lnM[j1-1:])
            p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
            p2 = simps(integ[j1+1:], x=self.halos.lnM[j1+1:])
            p3 = simps(integ[j1+2:], x=self.halos.lnM[j1+2:])

            interp = interp1d(self.halos.lnM[j1-1:j1+3], [p0,p1,p2,p3])

            return interp(np.log(Mmin))
        else:
            return super(GalaxyPopulation, self).iMAR(z)

    def cMAR(self, z, source=None):
        """
        Compute cumulative mass accretion rate, i.e., integrated MAR in 
        halos Mmin<M'<M.
        
        Parameters
        ----------
        z : int, float
        """
        
        if source is not None:        
            src = read_lit(source)
            MAR = src.MAR(z, self.halos.M)    
        else:
            MAR = super(GalaxyPopulation, self).MAR_via_AM(z)
                    
        # Grab redshift
        k = np.argmin(np.abs(z - self.halos.z))

        integ = self.halos.dndlnm[k] * MAR / self.cosm.fcdm

        Mmin = np.interp(z, self.halos.z, self.Mmin)
        j1 = np.argmin(np.abs(Mmin - self.halos.M))
        if Mmin > self.halos.M[j1]:
            j1 -= 1    

        incremental_Macc = cumtrapz(integ[j1:], x=self.halos.lnM[j1:],
            initial=0.0)

        return self.halos.M[j1:], incremental_Macc

    @property
    def eta(self):
        """
        Correction factor for MAR.
    
        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}

        """

        # Prepare to compute eta
        if not hasattr(self, '_eta'):
        
            if self.pf['pop_MAR_conserve_norm']:
                
                _rhs = np.zeros_like(self.halos.z)
                _lhs = np.zeros_like(self.halos.z)
                self._eta = np.ones_like(self.halos.z)
                
                for i, z in enumerate(self.halos.z):
                
                    # eta = rhs / lhs
                
                    Mmin = self.Mmin[i]
                
                    # My Eq. 3
                    rhs = self.cosm.rho_cdm_z0 * self.dfcolldt(z)
                    rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3
                
                    # Accretion onto all halos (of mass M) at this redshift
                    # This is *matter*, not *baryons*
                    MAR = self.MAR(z, self.halos.M)
                
                    # Find Mmin in self.halos.M
                    j1 = np.argmin(np.abs(Mmin - self.halos.M))
                    if Mmin > self.halos.M[j1]:
                        j1 -= 1
                
                    integ = self.halos.dndlnm[i] * MAR
                        
                    p0 = simps(integ[j1-1:], x=self.halos.lnM[j1-1:])
                    p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
                    p2 = simps(integ[j1+1:], x=self.halos.lnM[j1+1:])
                    p3 = simps(integ[j1+2:], x=self.halos.lnM[j1+2:])
                
                    interp = interp1d(self.halos.lnM[j1-1:j1+3], [p0,p1,p2,p3])

                    lhs = interp(np.log(Mmin))
                    
                    _lhs[i] = lhs
                    _rhs[i] = rhs

                    self._eta[i] = rhs / lhs
                        
            else:
                self._eta = np.ones_like(self.halos.z)
    
        return self._eta
                
    def metallicity_in_PR(self, z, M):
        return 1e-2 * (M / 1e11)**0.48 #* 10**(-0.15 * z)
                
    @property
    def cooling_function(self):
        if not hasattr(self, '_Lambda'):
            #rc = RateCoefficients()
            #cool_ci = lambda T: rc.CollisionalIonizationCoolingRate(0, T)
            #cool_re = lambda T: rc.RadiativeRecombinationRate(0, T)
            #cool_ex = lambda T: rc.CollisionalExcitationCoolingRate(0, T)
            #self._Lambda = lambda T: cool_ci(T) + cool_re(T) + cool_ex(T)
            #M = lambda z, T: self.halos.VirialMass(T, z)
            Z = lambda z, M: 1e-2#self.metallicity_in_PR(z, M)
            self._Lambda = lambda T, z: 1.8e-22 * (1e6 / T) * 1e-2#Z(z, M(z, T))
            
        return self._Lambda
                
    def SFR(self, z, M, mu=0.6):
        """
        Star formation rate at redshift z in a halo of mass M.
        
        ..note:: Units should be solar masses per year at this point.
        """
        
        return self.pSFR(z, M, mu) * self.SFE(z, M)

    def pSFR(self, z, M, mu=0.6):
        """
        The product of this number and the SFE gives you the SFR.
        
        pre-SFR factor, hence, "pSFR"        
        """
        if self.model == 'sfe':
            eta = np.interp(z, self.halos.z, self.eta)
            return self.cosm.fbar_over_fcdm * self.MAR(z, M) * eta
        elif self.model == 'tdyn':
            return self.cosm.fbaryon * M / self.tdyn(z, M)    
        elif self.model == 'precip':
            T = self.halos.VirialTemperature(M, z, mu)
            cool = self.cooling_function(T, z)
            pre_factor = 3. * np.pi * G * mu * m_p * k_B * T / 50. / cool                        
            return pre_factor * M * s_per_yr
        else:
            raise NotImplemented('Unrecognized model: %s' % self.model)
    
    @property
    def scalable_rhoL(self):
        """
        Can we just determine a luminosity density by scaling the SFRD?
        
        The answer will be "no" for any population with halo-mass-dependent
        values for photon yields (per SFR), escape fractions, or spectra.
        """
        
        if not hasattr(self, '_scalable_rhoL'):
            self._scalable_rhoL = True
            for par in Mh_dep_parameters:
                if type(self.pf[par]) is str:
                    self._scalable_rhoL = False
                    break
                    
                for i in range(self.pf.Nphps):
                    pn = '%s[%i]' % (par,i)
                    if pn not in self.pf:
                        continue

                    if type(self.pf[pn]) is str:
                        self._scalable_rhoL = False
                        break

        return self._scalable_rhoL
            
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

        # This assumes we're interested in the (EminNorm, EmaxNorm) band
        if self.scalable_rhoL:
            rhoL = super(GalaxyPopulation, self).Emissivity(z, E, Emin, Emax)
        else:
            raise NotImplemented('can\'t yet have Mh-dep SEDs (parametric)')

        if E is not None:
            return rhoL * self.src.Spectrum(E)
        else:
            return rhoL
            
    def StellarMassFunction(self, z, sSFR, mags=True):
        """
        Name says it all.
        
        Parameters
        ----------
        z : int, float
            Redshift.
        sSFR : int, float, np.ndarray
            Specific star-formation rate
        
        Returns
        -------   
        Tuple containing (stellar masses, # density).
            
        """
        
        assert mags

        mags, phi = self.phi_of_M(z)
    
        # mags corresponds to halos.M
        
        return self.SFR(z, self.halos.M) * sSFR, phi
    
    def LuminosityFunction(self, z, x, mags=True):
        """
        Reconstructed luminosity function.
        
        ..note:: This is number per [abcissa]. No dust correction has
            been applied.
                
        Parameters
        ----------
        z : int, float
            Redshift. Will interpolate between values in halos.z if necessary.
        mags : bool
            If True, x-values will be in absolute (AB) magnitudes
        Returns
        -------
        Magnitudes (or luminosities) and number density.

        """

        if mags:
            x_phi, phi = self.phi_of_M(z)
            phi_of_x = 10**np.interp(x, x_phi[-1::-1], np.log10(phi)[-1::-1])
        else:
            
            x_phi, phi = self.phi_of_L(z)
            
            # Setup interpolant
            interp = interp1d(np.log10(x_phi), np.log10(phi), kind='linear',
                bounds_error=False, fill_value=-np.inf)
            
            phi_of_x = 10**interp(np.log10(x))
                                                                
        return phi_of_x

    def Lh(self, z):
        return self.SFR(z, self.halos.M) * self.L1500_per_sfr(z, self.halos.M)

    def phi_of_L(self, z):

        if not hasattr(self, '_phi_of_L'):
            self._phi_of_L = {}
        else:
            if z in self._phi_of_L:
                return self._phi_of_L[z]

        Lh = self.Lh(z)
        
        dMh_dLh = np.diff(self.halos.M) / np.diff(Lh)
        dndm = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)

        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self.Mmin)

        above_Mmin = self.halos.M >= Mmin
        below_Mmax = self.halos.M <= self.pf['pop_lf_Mmax']
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)

        phi_of_L = dndm(z) * self.fduty(z, self.halos.M[0:-1]) * dMh_dLh
        
        lum = np.ma.array(Lh[:-1], mask=mask)
        phi = np.ma.array(phi_of_L, mask=mask)
        
        phi[mask == True] = 0.

        self._phi_of_L[z] = lum, phi

        return self._phi_of_L[z]

    def phi_of_M(self, z):
        if not hasattr(self, '_phi_of_M'):
            self._phi_of_M = {}
        else:
            if z in self._phi_of_M:
                return self._phi_of_M[z]

        Lh, phi_of_L = self.phi_of_L(z)

        MAB = self.magsys.L_to_MAB(Lh, z=z)

        phi_of_M = phi_of_L[0:-1] * np.abs(np.diff(Lh) / np.diff(MAB))

        self._phi_of_M[z] = MAB[0:-1], phi_of_M

        return self._phi_of_M[z]

    def MUV_max(self, z): 
        """
        Compute the magnitude corresponding to the Tmin threshold.
        """   

        i_z = np.argmin(np.abs(z - self.halos.z))

        Mmin = np.interp(z, self.halos.z, self.Mmin)
        Lmin = np.interp(Mmin, self.halos.M, self.Lh(z))

        MAB = self.magsys.L_to_MAB(Lmin, z=z)

        return MAB

    def Mh_of_MUV(self, z, MUV):
        
        # MAB corresponds to self.halos.M
        MAB, phi = self.phi_of_M(z)

        return np.interp(MUV, MAB[-1::-1], self.halos.M[-1:1:-1])
        

    def lf_from_pars(self, z, pars):
        for i, par in enumerate(pars):
            self.pf['php_Mfun_par%i' % i] = par

        return self.phi_of_M(z)

    @property
    def Mmin(self):
        if not hasattr(self, '_Mmin'):
            # First, compute threshold mass vs. redshift
            if self.pf['pop_Mmin'] is not None:
                self._Mmin = self.pf['pop_Mmin'] * np.ones(self.halos.Nz)
            else:
                Mvir = lambda z: self.halos.VirialMass(self.pf['pop_Tmin'], 
                    z, mu=self.pf['mu'])
                self._Mmin = np.array(map(Mvir, self.halos.z))

        return self._Mmin    

    @property
    def sfr_tab(self):
        """
        SFR as a function of redshift and halo mass.

            ..note:: Units are Msun/yr.
    
        """
        if not hasattr(self, '_sfr_tab'):
            self._sfr_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            for i, z in enumerate(self.halos.z):
                self._sfr_tab[i] = self.eta[i] * self.MAR(z, self.halos.M) \
                    * self.cosm.fbar_over_fcdm * self.SFE(z, self.halos.M)
    
                mask = self.halos.M >= self.Mmin[i]
                self._sfr_tab[i] *= mask
    
        return self._sfr_tab
                
    @property
    def sfrd_tab(self):
        """
        SFRD as a function of redshift.
    
            ..note:: Units are g/s/cm^3 (comoving).
    
        """
        if not hasattr(self, '_sfrd_tab'):
            self._sfrd_tab = np.zeros(self.halos.Nz)
            
            for i, z in enumerate(self.halos.z):
                integrand = self.sfr_tab[i] * self.halos.dndlnm[i] \
                    * self.fduty(z, self.halos.M)
 
                tot = np.trapz(integrand, x=self.halos.lnM)
                cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
                
                self._sfrd_tab[i] = tot - \
                    np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
                
            self._sfrd_tab *= g_per_msun / s_per_yr / cm_per_mpc**3

        return self._sfrd_tab
    
    @property
    def LLyC_tab(self):
        """
        Number of LyC photons emitted per unit SFR in halos of mass M.
        """
        if not hasattr(self, '_LLyC_tab'):
            M = self.halos.M
            fesc = self.fesc(None, M)
            
            dnu = (24.6 - 13.6) / ev_per_hz
            #nrg_per_phot = 25. * erg_per_ev

            Nion_per_L1500 = self.Nion(None, M) / (1. / dnu)
            
            self._LLyC_tab = np.zeros([self.halos.Nz, self.halos.Nm])
            
            for i, z in enumerate(self.halos.z):
                self._LLyC_tab[i] = self.L1500_tab[i] * Nion_per_L1500 \
                    * fesc
            
                mask = self.halos.M >= self.Mmin[i]
                self._LLyC_tab[i] *= mask
            
        return self._LLyC_tab
                
    @property
    def LLW_tab(self):
        if not hasattr(self, '_LLW_tab'):
            M = self.halos.M
    
            dnu = (13.6 - 10.2) / ev_per_hz
            #nrg_per_phot = 25. * erg_per_ev
    
            Nlw_per_L1500 = self.Nlw(None, M) / (1. / dnu)
    
            self._LLW_tab = np.zeros([self.halos.Nz, self.halos.Nm])
    
            for i, z in enumerate(self.halos.z):
                self._LLW_tab[i] = self.L1500_tab[i] * Nlw_per_L1500
    
                mask = self.halos.M >= self.Mmin[i]
                self._LLW_tab[i] *= mask

        return self._LLW_tab

    def SFE(self, z, M):
        """
        Compute the star-formation efficiency.
    
        If outside the bounds, must extrapolate.
        """
        return self.fstar(z, M)

    def fstar_no_boost(self, z, M, coeff):
        """
        Only used in AbundanceMatching routine. Kind of a cludge.
        """
        if not hasattr(self, '_fstar'):
            tmp = self.fstar

        return self._fstar_inst._call(z, M, coeff)

    @property
    def fstar(self):
        if not hasattr(self, '_fstar'):
            
            if self.pf['pop_calib_rhoL1500'] is not None:
                boost = self.pf['pop_calib_rhoL1500'] / self.L1500_per_sfr(None, None)
                assert self.pf['pop_fstar_boost'] == 1
            else:
                boost = 1. / self.pf['pop_fstar_boost']
            
            if type(self.pf['pop_fstar']) in [float, np.float64]:
                self._fstar = lambda z, M: self.pf['pop_fstar'] * boost
            elif self.pf['pop_fstar'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_fstar'])
                self._fstar_inst = ParameterizedHaloProperty(**pars)
                
                self._fstar = lambda z, M: self._fstar_inst.__call__(z, M) \
                        * boost
            else:
                raise ValueError('Unrecognized data type for pop_fstar!')  
                
        return self._fstar
    
    @property
    def fduty(self):
        if not hasattr(self, '_fduty'):
            if type(self.pf['pop_fduty']) in [float, np.float64]:
                self._fduty = lambda z, M: self.pf['pop_fduty']
            elif self.pf['pop_fstar'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_fduty'])
                self._fduty = ParameterizedHaloProperty(**pars)
            else:
                raise ValueError('Unrecognized data type for pop_fstar!')  
    
        return self._fduty   
    
    @property    
    def fesc(self):
        if not hasattr(self, '_fesc'):
            if type(self.pf['pop_fesc']) in [float, np.float64]:
                self._fesc = lambda z, M: self.pf['pop_fesc']
            elif self.pf['pop_fesc'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_fesc'])    
                self._fesc = ParameterizedHaloProperty(**pars)
            else:
                raise ValueError('Unrecognized data type for pop_fesc!')  
    
        return self._fesc

    @property    
    def tdyn(self):
        if not hasattr(self, '_tdyn'):
            if type(self.pf['pop_tdyn']) in [float, np.float64]:
                self._tdyn = lambda z, M: self.pf['pop_tdyn']
            elif self.pf['pop_tdyn'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_tdyn'])    
                self._tdyn = ParameterizedHaloProperty(**pars)
            else:
                raise ValueError('Unrecognized data type for pop_fesc!')  
    
        return self._tdyn  
    
    def get_php_pars(self, par):
        """
        Find ParameterizedHaloProperty's for this parameter.
        
        ..note:: par isn't the name of the parameter, it is the value. Usually,
            it's something like 'php[0]'.
            
        For example, if in the parameter file, you set:
        
            'pop_fesc{0}'='php[1]'
            
        This routine runs off and finds all parameters that look like:
        
            'php_*par?{0}[1]'
            
        Returns
        -------
        Dictionary of parameters to be used to initialize a new HaloProperty.
            
        """


        prefix, popid, phpid = par_info(par)

        pars = {}
        for key in self.pf:
            if (self.pf.Nphps != 1):
                if not re.search('\[%i\]' % phpid, key):
                    continue

            if key[0:3] != 'php':
                continue

            p, popid, phpid_ = par_info(key)    

            if (phpid is None) and (self.pf.Nphps == 1):
                pars[p] = self.pf['%s' % p]          

            # This means we probably have some parameters bracketed
            # and some not...should make it so this doesn't happen
            elif (phpid is not None) and (self.pf.Nphps == 1):
                try:
                    pars[p] = self.pf['%s[%i]' % (p, phpid)]   
                except KeyError:
                    # This means it's just default values
                    pars[p] = self.pf['%s' % p]   
            else:    
                pars[p] = self.pf['%s[%i]' % (p, phpid)]

        return pars
        
    def gamma_sfe(self, z, M):
        """
        This is a power-law index describing the relationship between the
        SFE and and halo mass.
        
        Parameters
        ----------
        z : int, float
            Redshift
        M : int, float
            Halo mass in [Msun]
            
        """
        
        fst = lambda MM: self.SFE(z, MM)
        
        return derivative(fst, M, dx=1e6) * M / fst(M)
            
    def alpha_lf(self, z, mag):
        """
        Slope in the luminosity function
        """
        
        logphi = lambda logL: np.log10(self.LuminosityFunction(z, 10**logL, mags=False))
        
        Mdc = mag - self.AUV(z, mag)
        L = self.magsys.MAB_to_L(mag=Mdc, z=z)
        
        return derivative(logphi, np.log10(L), dx=0.1)
        
    #def LuminosityDensity(self, z, Emin=None, Emax=None):
    #    """
    #    Return the integrated luminosity density in the (Emin, Emax) band.
    #    
    #    Parameters
    #    ----------
    #    z : int, flot
    #        Redshift of interest.
    #    
    #    Returns
    #    -------
    #    Luminosity density in erg / s / c-cm**3.
    #    
    #    """
    #    
    #    if self.scalable_rhoL:
    #        return self.Emissivity(z, Emin, Emax)
    #    else:
    #        return self.rho_L[(Emin, Emax)](z)
     
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
            
        # erg / s / cm**3
        if self.scalable_rhoL:
            rhoL = self.Emissivity(z, Emin, Emax)
            erg_per_phot = super(GalaxyPopulation, 
                self)._get_energy_per_photon(Emin, Emax) * erg_per_ev
            return rhoL / erg_per_phot
        else:
            return self.rho_N(z, Emin, Emax)
        

        
        
        
         
         
         
     