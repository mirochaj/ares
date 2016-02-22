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

    @property
    def L1500_per_SFR(self):
        """
        
        Returns
        -------
        In units of erg/s/Hz/(Msun/yr).
        
        """
        
        if not hasattr(self, '_L1500_per_SFR'):
            if self.sed_tab:
                self._L1500_per_SFR = lambda z, M: self.src.pop.L_per_sfr()
            elif type(self.pf['pop_L1500_per_sfr']) in [float, np.float64]:
                self._L1500_per_SFR = \
                    lambda z, M: self.pf['pop_L1500_per_sfr']
            elif self.pf['pop_L1500_per_sfr'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_L1500_per_sfr']) 
                self._L1500_per_SFR = ParameterizedHaloProperty(**pars)    
            else:
                raise TypeError('dunno how to handle this')
        
        return self._L1500_per_SFR

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
                * N_per_Msun * fesc
        
            tot = np.trapz(integrand, x=self.halos.lnM)
            cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
            
            tab[i] = tot - \
                np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
            
        tab *= 1. / s_per_yr / cm_per_mpc**3    
        
        self._rho_N[(Emin, Emax)] = interp1d(self.halos.z, tab, kind='cubic')

        return self._rho_N[(Emin, Emax)](z)
        
    @property   
    def SFRD(self):
        """
        Compute star-formation rate density (SFRD).
        """
        
        if not hasattr(self, '_SFRD'):
            self._SFRD = interp1d(self.halos.z, self.sfrd_tab, kind='cubic')
                
        return self._SFRD

    @property
    def Macc(self):
        """
        Mass accretion rate onto halos of mass M at redshift z.
    
        ..note:: This is the *matter* accretion rate. To obtain the baryonic 
            accretion rate, multiply by Cosmology.fbaryon.
            
        """
        if not hasattr(self, '_Macc'):
            if self.pf['pop_Macc'] is None:
                self._Macc = None
            elif type(self.pf['pop_Macc']) is FunctionType:
                self._Macc = self.pf['pop_Macc']
            elif self.pf['pop_Macc'] == 'pl':
                raise NotImplemented('do this')
            else:
                self._Macc = read_lit(self.pf['pop_Macc']).Macc

        return self._Macc
        
    @property
    def eta(self):
        """
        Correction factor for Macc.
    
        \eta(z) \int_{M_{\min}}^{\infty} \dot{M}_{\mathrm{acc}}(z,M) n(z,M) dM
            = \bar{\rho}_m^0 \frac{df_{\mathrm{coll}}}{dt}|_{M_{\min}}

        """

        # Prepare to compute eta
        if not hasattr(self, '_eta'):
    
            self._eta = np.zeros_like(self.halos.z)
    
            for i, z in enumerate(self.halos.z):
    
                # eta = rhs / lhs
    
                Mmin = self.Mmin[i]
    
                rhs = self.cosm.rho_m_z0 * self.dfcolldt(z)
                rhs *= (s_per_yr / g_per_msun) * cm_per_mpc**3
    
                # Accretion onto all halos (of mass M) at this redshift
                # This is *matter*, not *baryons*
                Macc = self.Macc(z, self.halos.M)
    
                # Find Mmin in self.halos.M
                j1 = np.argmin(np.abs(Mmin - self.halos.M))
                if Mmin > self.halos.M[j1]:
                    j1 -= 1
    
                integ = self.halos.dndlnm[i] * Macc
                    
                p0 = simps(integ[j1-1:], x=self.halos.lnM[j1-1:])
                p1 = simps(integ[j1:], x=self.halos.lnM[j1:])
                p2 = simps(integ[j1+1:], x=self.halos.lnM[j1+1:])
                p3 = simps(integ[j1+2:], x=self.halos.lnM[j1+2:])
    
                interp = interp1d(self.halos.lnM[j1-1:j1+3], [p0,p1,p2,p3])
    
                lhs = interp(np.log(Mmin))
    
                self._eta[i] = rhs / lhs
    
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
        if self.model == 'sfe':
            eta = np.interp(z, self.halos.z, self.eta)
            return self.cosm.fbaryon * self.Macc(z, M) * eta * self.SFE(z, M)
        elif self.model == 'tdyn':
            return self.pf['pop_fstar'] * self.cosm.fbaryon * M / self.tdyn(z, M)    
        elif self.model == 'precip':
            T = self.halos.VirialTemperature(M, z, mu)
            cool = self.cooling_function(T, z)
            pre_factor = 3. * np.pi * G * mu * m_p * k_B * T / 50. / cool
                        
            return pre_factor * M * s_per_yr * self.SFE(z, M)
        else:
            raise NotImplemented('Unrecognized model: %s' % self.model)
        
    def pSFR(self, z, M, mu=0.6):
        """
        The product of this number and the SFE gives you the SFR.
        
        pre-SFR factor, hence, "pSFR"        
        """
        if self.model == 'sfe':
            eta = np.interp(z, self.halos.z, self.eta)
            return self.cosm.fbaryon * self.Macc(z, M) * eta
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

    def phi_of_L(self, z):

        if not hasattr(self, '_phi_of_L'):
            self._phi_of_L = {}
        else:
            if z in self._phi_of_L:
                return self._phi_of_L[z]

        Lh = self.SFR(z, self.halos.M) * self.L1500_per_SFR(z, self.halos.M)
        
        dMh_dLh = np.diff(self.halos.M) / np.diff(Lh)
        dndm = interp1d(self.halos.z, self.halos.dndm[:,:-1], axis=0)

        # Only return stuff above Mmin
        Mmin = np.interp(z, self.halos.z, self.Mmin)

        above_Mmin = self.halos.M >= Mmin
        below_Mmax = self.halos.M <= self.pf['pop_lf_Mmax']
        ok = np.logical_and(above_Mmin, below_Mmax)[0:-1]
        mask = self.mask = np.logical_not(ok)

        phi_of_L = dndm(z) * dMh_dLh
        
        lum = np.ma.array(Lh[:-1], mask=mask)
        phi = np.ma.array(phi_of_L, mask=mask)

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
        
    def lf_from_pars(self, z, pars):
        for i, par in enumerate(pars):
            self.pf['php_Mfun_par%i' % i] = par
    
        return self.phi_of_M(z)
        
    def MAB_limit(self, z):
        """
        Magnitude corresponding to minimum halo mass in which stars form.
        """
        
        Lh_Mmin = self.L1600_limit(z)
        
        return self.magsys.L_to_MAB(Lh_Mmin, z=z)

    #@property
    #def LofM_tab(self):
    #    """
    #    Intrinsic luminosities corresponding to the supplied magnitudes.
    #    """
    #    if not hasattr(self, '_LofM_tab'):
    #        tab = self.fstar_tab
    #
    #    return self._LofM_tab            
    #
    #@property
    #def MofL_tab(self):
    #    """
    #    These are the halo masses determined via abundance matching that
    #    correspond to the M_UV's provided.
    #    """
    #    if not hasattr(self, '_MofL_tab'):
    #        tab = self.fstar_tab
    #
    #    return self._MofL_tab

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
                self._sfr_tab[i] = self.eta[i] * self.Macc(z, self.halos.M) \
                    * self.cosm.fbaryon * self.SFE(z, self.halos.M)
    
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
                integrand = self.sfr_tab[i] * self.halos.dndlnm[i]
 
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
        
        
        
    #@property
    #def f912_per_f1500(self):
    #    return 1.

    #@property
    #def L1500_tab(self):
    #    """
    #    Luminosity as a function of redshift and halo mass.
    #
    #        ..note:: Units are erg/s/Hz/(Msun / yr).
    #
    #    """
    #    if not hasattr(self, '_L1500_tab'):
    #        M = self.halos.M
    #        kappa = self.kappa_UV(None, M)
    #        
    #        self._L1500_tab = np.zeros([self.halos.Nz, self.halos.Nm])
    #        
    #        for i, z in enumerate(self.halos.z):
    #            self._L1500_tab[i] = self.sfr_tab[i,:] / kappa
    #        
    #            mask = self.halos.M >= self.Mmin[i]
    #            self._L1500_tab[i] *= mask
    #
    #    return self._L1500_tab
        
    #@property
    #def rhoL1500_tab(self):
    #    """
    #    Luminosity density at 1500A as a function of redshift.
    #
    #        ..note:: Units are erg/s/Hz (comoving).
    #
    #    """
    #    if not hasattr(self, '_rhoL1500_tab'):
    #        self._rhoL1500_tab = np.zeros(self.halos.Nz)
    #
    #        for i, z in enumerate(self.halos.z):
    #            integrand = self.L1500_tab[i] * self.halos.dndlnm[i]
    #
    #            tot = np.trapz(integrand, x=self.halos.lnM)
    #            cumtot = cumtrapz(integrand, x=self.halos.lnM, initial=0.0)
    #
    #            self._rhoL1500_tab[i] = tot - \
    #                np.interp(np.log(self.Mmin[i]), self.halos.lnM, cumtot)
    #    
    #        self._rhoL1500_tab /= cm_per_mpc**3
    #    
    #    return self._rhoL1500_tab    

    def SFE(self, z, M):
        """
        Compute the star-formation efficiency.
    
        If outside the bounds, must extrapolate.
        """
        return self.fstar(z, M)
    
    @property
    def fstar(self):
        if not hasattr(self, '_fstar'):
            if type(self.pf['pop_fstar']) in [float, np.float64]:
                self._fstar = lambda z, M: self.pf['pop_fstar']
            elif self.pf['pop_fstar'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_fstar'])
                self._fstar = ParameterizedHaloProperty(**pars)
            else:
                raise ValueError('Unrecognized data type for pop_fstar!')  
                
        return self._fstar
    
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
    def Nion(self):
        if not hasattr(self, '_Nion'):
            if type(self.pf['pop_Nion']) in [float, np.float64]:
                self._Nion = lambda z, M: self.pf['pop_Nion']
            elif self.pf['pop_Nion'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_Nion'])
                self._Nion = ParameterizedHaloProperty(**pars)
            else:
                raise ValueError('Unrecognized data type for pop_fesc!')  
    
        return self._Nion   
        
    @property
    def Nlw(self):
        if not hasattr(self, '_Nlw'):
            if type(self.pf['pop_Nlw']) in [float, np.float64]:
                self._Nlw = lambda z, M: self.pf['pop_Nlw']
            elif self.pf['pop_Nlw'][0:3] == 'php':
                pars = self.get_php_pars(self.pf['pop_Nlw'])
                self._Nlw = ParameterizedHaloProperty(**pars)
            else:
                raise ValueError('Unrecognized data type for pop_fesc!')  
    
        return self._Nlw 
    
    def get_php_pars(self, par):
        """
        Find ParameterizedHaloProperty's for this parameter.
        """
        
        prefix, popid, phpid = par_info(par)

        pars = {}
        for key in self.pf:
            if not re.search('\[%i\]' % phpid, key):
                continue
                
            if key[0:3] != 'php':
                continue
                
            p, popid, phpid_ = par_info(key)    
            
            if (phpid is None) and (self.pf.Nphps == 1):
                pars[p] = self.pf['%s' % p]
            else:    
                pars[p] = self.pf['%s[%i]' % (p, phpid)]
                        
        return pars
        
    @property
    def tdyn_inv(self):
        if not hasattr(self, '_tdyn_func'):
            self._tdyn_inv = ParameterizedHaloProperty(**self.pf)
        return self._tdyn_inv
    
    def tdyn(self, z, M):
        """
        Compute the dynamical time of a halo of mass M at redshift z.
        
        .. note :: Units are [s]^{-1}
        """
        return 1. / self.tdyn_inv(z, M)
    
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
        

        
        
        
         
         
         
     