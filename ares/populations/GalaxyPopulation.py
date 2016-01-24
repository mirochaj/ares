"""

GalaxyMZ.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Jan 13 09:49:00 PST 2016

Description: 

"""

import numpy as np
from ..util import read_lit
from types import FunctionType
from collections import namedtuple
from .GalaxyAggregate import GalaxyAggregate
from scipy.optimize import fsolve, curve_fit
from ..util.DustCorrection import DustCorrection
from scipy.integrate import quad, simps, cumtrapz, ode
from ..util.StarFormationEfficiency import ParameterizedSFE
from scipy.interpolate import interp1d, RectBivariateSpline
from ..util import ParameterFile, MagnitudeSystem, ProgressBar
from ..physics.Constants import s_per_yr, g_per_msun, cm_per_mpc

try:
    from scipy.misc import derivative
except ImportError:
    pass
    
z0 = 9. # arbitrary
    
class GalaxyPopulation(GalaxyAggregate,ParameterizedSFE,DustCorrection):

    @property
    def magsys(self):
        if not hasattr(self, '_magsys'):
            self._magsys = MagnitudeSystem(**self.pf)
    
        return self._magsys
        
    @property   
    def SFRD(self):
        """
        Compute star-formation rate density (SFRD).
        
        """
        
        if not hasattr(self, '_SFRD'):
            self._SFRD = interp1d(self.halos.z, self.sfrd_tab,
                kind='cubic')
                
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
    def kappa_UV(self):
        if not hasattr(self, '_kappa_UV'):
            if self.sed_tab:
                self._kappa_UV = self.src.pop.kappa_UV()
            else:
                self._kappa_UV = self.pf['pop_kappa_UV']
            
        return self._kappa_UV    
    
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
                
    def SFR(self, z, M):
        eta = np.interp(z, self.halos.z, self.eta)
        return self.cosm.fbaryon * self.Macc(z, M) * eta * self.SFE(z, M)
        
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

        Lh = self.SFR(z, self.halos.M) / self.kappa_UV
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

    def L1600_limit(self, z):
        eta = np.interp(z, self.halos.z, self.eta)
        Mmin = np.interp(z, self.halos.z, self.Mmin)

        #sfr_M_z = RectBivariateSpline(self.halos.z, self.halos.lnM, 
        #    np.log(self.sfr_tab))

        #Lh_Mmin = np.exp(sfr_M_z(z, np.log(Mmin))[0][0]) / self.kappa_UV   

        return self.cosm.fbaryon * self.Macc(z, Mmin) \
            * eta * self.SFE(z, Mmin) / self.kappa_UV
            
    def MAB_limit(self, z):
        """
        Magnitude corresponding to minimum halo mass in which stars form.
        """
        
        Lh_Mmin = self.L1600_limit(z)
        
        return self.magsys.L_to_MAB(Lh_Mmin, z=z)

    @property
    def LofM_tab(self):
        """
        Intrinsic luminosities corresponding to the supplied magnitudes.
        """
        if not hasattr(self, '_LofM_tab'):
            tab = self.fstar_tab

        return self._LofM_tab            

    @property
    def MofL_tab(self):
        """
        These are the halo masses determined via abundance matching that
        correspond to the M_UV's provided.
        """
        if not hasattr(self, '_MofL_tab'):
            tab = self.fstar_tab
    
        return self._MofL_tab

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
        SFR as a function of redshift and halo mass yielded by abundance match.

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
        SFRD as a function of redshift yielded by abundance match.
    
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
    def _apply_floor(self):
        if not hasattr(self, '_apply_floor_'):
            self._apply_floor_ = 1
        return self._apply_floor_
    
    @_apply_floor.setter
    def _apply_floor(self, value):
        self._apply_floor_ = value        
        
    @property
    def Mpars_of_z(self):
        if not hasattr(self, '_Mpars_of_z'):
            
            if self.zfunc == 'constant':
                f1 = lambda zz: self.pf['pop_sfe_Mfun_par0']
                f2 = lambda zz: self.pf['pop_sfe_Mfun_par1']
                f3 = lambda zz: self.pf['pop_sfe_Mfun_par2']
                self._Mpars_of_z = (f1, f2, f3)
            elif self.zfunc == 'linear_t':
                co1, co2 = self.pf['pop_sfe_Mfun_par0'], self.pf['pop_sfe_Mfun_par1']
                f1 = lambda zz: coeff1 + coeff2 * (1. + zz) / z0
                co3, co4 = self.pf['pop_sfe_Mfun_par0'], self.pf['pop_sfe_Mfun_par1']
                f2 = lambda zz: coeff3 + coeff4 * (1. + zz) / z0
                co5, co6 = self.pf['pop_sfe_Mfun_par0'], self.pf['pop_sfe_Mfun_par1']
                f3 = lambda zz: coeff5 + coeff6 * (1. + zz) / z0
                self._Mpars_of_z = (f1, f2, f3)
                
            elif self.zfunc == 'linear_z':
                self._Mpars_of_z = {}
                for i in range(3):
                    coeff = self.pf['pop_sfe_Mfun_par%i' % i]
                    func = lambda zz: coeff - 1.5 * (1. + zz) / z0
                    self._Mpars_of_z[i] = func

        return self._Mpars_of_z        
                        
    #ef fstar(self, z, M):  
    #   """
    #   Compute the halo-mass and redshift dependent star formation efficiency.
    #   
    #   Parameters
    #   ----------
    #   
    #   """  
    #   
    #   logM = np.log10(M)
    #   
    #   if self.Mfunc == 'lognormal':
    #       p = self.Mpars_of_z
    #       f = p[0](z) * np.exp(-(logM - p[1](z))**2 / 2. / p[2](z)**2)
    #   else:
    #       raise NotImplemented('sorry!')
    #           
    #   # Nothing stopping some of the above treatments from negative fstar 
    #   f = np.maximum(f, 0.0)
    #           
    #   ##
    #   # HANDLE LOW-MASS END
    #   ##        
    #   #if (self.Mext == 'floor'):
    #   #    f += self.Mext[1]
    #   #elif self.Mext == 'pl_floor' and self._apply_floor:
    #   #    self._apply_floor = 0
    #   #    to_add = self.Mext_pars[0] * 10**self._log_fstar(z, 1e10, *coeff)
    #   #    to_add *= (M / 1e10)**self.Mext_pars[1] * np.exp(-M / 1e11)
    #   #    f += to_add
    #   #    self._apply_floor = 1
    #       
    #   # Apply ceiling
    #   f = np.minimum(f, self.pf['pop_sfe_ceil'])
    #                       
    #   return f

    def SFE(self, z, M):
        """
        Compute the star-formation efficiency.
    
        If outside the bounds, must extrapolate.
        """
    
        return self.fstar(z, M)
    
        ## Otherwise, we fit the mass-to-light ratio
        #eta = np.interp(z, self.halos.z, self.eta)
        #
        #return self.Lh(z, M) * self.kappa_UV \
        #    / (self.cosm.fbaryon * self.Macc(z, M) * eta)        
    
    
    def Lh(self, z, M):
        return 10**self._log_Lh(z, M, *self.coeff) 
           
    def _log_Lh(self, z, M, *coeff): 
        if self.Mfunc == 'pl':
            return coeff[0] + coeff[1] * np.log10(M / 1e12)
        elif self.Mfunc == 'schechter':
            return coeff[0] + coeff[1] * np.log10(M / coeff[2]) - M / coeff[2]
        elif self.Mfunc == 'poly':
            return coeff[0] + coeff[1] * np.log10(M / 1e10) \
                + coeff[2] * (np.log10(M / 1e10))**2 
    
    @property
    def Npops(self):
        return self.pf.Npops
        
    @property
    def pop_id(self):
        # Pop ID number for HAM population
        if not hasattr(self, '_pop_id'):
            for i, pf in enumerate(self.pf.pfs):
                if pf['pop_model'] == 'ham':
                    break
            
            self._pop_id = i
        
        return self._pop_id
    
    @property
    def Mext(self):
        return self.pf.pfs[self.pop_id]['pop_sfe_Mext']

    @property
    def Mext_pars(self):
        return self.pf.pfs[self.pop_id]['pop_sfe_Mext_par1'], \
            self.pf.pfs[self.pop_id]['pop_sfe_Mext_par2']

    @property
    def zext(self):
        return self.pf.pfs[self.pop_id]['pop_sfe_zext'], \
              self.pf.pfs[self.pop_id]['pop_sfe_zext_par']

    #def Mpeak(self, z):
    #    """
    #    The mass at which the star formation efficiency peaks.
    #    """
    #    
    #    alpha = lambda MM: self.gamma_sfe(z, MM)
    #        
    #    i = np.argmin(np.abs(z - np.array(self.redshifts)))
    #    guess = self.MofL_tab[i][np.argmax(self.fstar_tab[i])]
    #
    #    return fsolve(alpha, x0=guess, maxfev=10000, full_output=False,
    #        xtol=1e-3)[0]
    #
    #def fpeak(self, z):
    #    return self.SFE(z, self.Mpeak(z))
    
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
        

            
            
            
            
            
    