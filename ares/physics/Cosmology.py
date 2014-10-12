""" 

Cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description:

"""

import numpy as np
from scipy.integrate import quad
from .Constants import c, G, km_per_mpc, m_H, m_He, sigma_SB

class Cosmology:
    def __init__(self, OmegaMatterNow=0.272, OmegaLambdaNow=0.728,
        OmegaBaryonNow=0.044, HubbleParameterNow=0.702, 
        HeliumAbundanceByNumber=0.08,
        CMBTemperatureNow=2.725, 
        approx_highz=False, SigmaEight=0.807, PrimordialIndex=0.96):
        """Initialize a Cosmology object.
        
        :param: OmegaMatterNow: Pretty self-explanatory.
        
        """
                
        self.OmegaMatterNow = OmegaMatterNow
        self.OmegaBaryonNow = OmegaBaryonNow
        self.OmegaLambdaNow = OmegaLambdaNow
        self.OmegaCDMNow = self.OmegaMatterNow - self.OmegaBaryonNow
        self.HubbleParameterNow = HubbleParameterNow * 100 / km_per_mpc
        self.CMBTemperatureNow = CMBTemperatureNow
        self.approx_highz = approx_highz
        self.SigmaEight = self.sigma8 = SigmaEight
        self.PrimordialIndex = PrimordialIndex
        
        self.CriticalDensityNow = (3 * self.HubbleParameterNow**2) \
            / (8 * np.pi * G)
        
        self.h70 = HubbleParameterNow
        
        self.y = HeliumAbundanceByNumber
        self.Y = 4. * self.y / (1. + 4. * self.y)
        
        self.X = 1. - self.Y
        
        self.g_per_baryon = m_H / (1. - self.Y) / (1. + self.y)
                
        self.zdec = 150. * (self.OmegaBaryonNow * self.h70**2 / 0.023)**0.4 - 1.

        self.Omh2 = self.OmegaBaryonNow * self.h70**2

        # Hydrogen, helium, electron, and baryon densities today (z = 0)
        self.rho_b_z0 = self.MeanBaryonDensity(0)
        self.rho_m_z0 = self.MeanMatterDensity(0)
        self.nH0 = (1. - self.Y) * self.rho_b_z0 / m_H
        self.nHe0 = self.y * self.nH0
        self.ne0 = self.nH0 + 2. * self.nHe0
        #self.n0 = self.nH0 + self.nHe0 + self.ne0
        
        self.nH = lambda z: self.nH0 * (1. + z)**3
        self.nHe = lambda z: self.nHe0 * (1. + z)**3
        
        self.delta_c0 = 1.686
        self.TcmbNow = self.CMBTemperatureNow
        
        self.pars = {'omega_lambda':self.OmegaLambdaNow,
         'omega_b':self.OmegaBaryonNow,
         'omega_M':self.OmegaMatterNow,
         'sigma_8':self.sigma8,
         'n': self.PrimordialIndex}
        
        
        
    def TimeToRedshiftConverter(self, t_i, t_f, z_i):
        """
        High redshift approximation under effect.
        """
        return ((1. + z_i)**(-3. / 2.) + (3. * self.HubbleParameterNow * 
            np.sqrt(self.OmegaMatterNow) * (t_f - t_i) / 2.))**(-2. / 3.) - 1.
        
    def LookbackTime(self, z_i, z_f):
        """
        Returns lookback time from z_i to z_f in seconds, where z_i < z_f.
        """
        return 2. * ((1. + z_i)**-1.5 - (1. + z_f)**-1.5) / \
            np.sqrt(self.OmegaMatterNow) / self.HubbleParameterNow / 3.    
        
    def TCMB(self, z):
        return self.CMBTemperatureNow * (1. + z)
        
    def UCMB(self, z):
        """ CMB energy density. """    
        return 4.0 * sigma_SB * self.TCMB(z)**4 / c
    
    def Tgas(self, z):
        """
        Gas kinetic temperature at z assuming only adiabatic cooling after zdec.
        """
        
        if z >= self.zdec:
            return self.TCMB(z)
        else:
            return self.TCMB(self.zdec) * (1. + z)**2 / (1. + self.zdec)**2

    def ScaleFactor(self, z):
        return 1. / (1. + z)
        
    def EvolutionFunction(self, z):
        return self.OmegaMatterNow * (1.0 + z)**3  + self.OmegaLambdaNow
        
    def HubbleParameter(self, z):
        if self.approx_highz:
            return self.HubbleParameterNow * np.sqrt(self.OmegaMatterNow) \
                * (1. + z)**1.5
        return self.HubbleParameterNow * np.sqrt(self.EvolutionFunction(z)) 
    
    def HubbleLength(self, z):
        return c / self.HubbleParameter(z)
    
    def HubbleTime(self, z):
        return 2. / 3. / self.HubbleParameter(z)
        
    def OmegaMatter(self, z):
        if self.approx_highz:
            return 1.0
        return self.OmegaMatterNow * (1. + z)**3 / self.EvolutionFunction(z)
    
    def OmegaLambda(self, z):
        if self.approx_highz:
            return 0.0
        
        return self.OmegaLambdaNow / self.EvolutionFunction(z)
    
    def MeanMatterDensity(self, z):
        return self.OmegaMatter(z) * self.CriticalDensity(z)
        
    def MeanBaryonDensity(self, z):
        return (self.OmegaBaryonNow / self.OmegaMatterNow) \
            * self.MeanMatterDensity(z)
    
    def MeanHydrogenNumberDensity(self, z):
        return (1. - self.Y) * self.MeanBaryonDensity(z) / m_H
        
    def MeanHeliumNumberDensity(self, z):
        return self.Y * self.MeanBaryonDensity(z) / m_He    
    
    def MeanBaryonNumberDensity(self, z):
        return self.MeanBaryonDensity(z) / (m_H * self.MeanHydrogenNumberDensity(z) + 
            4. * m_H * self.y * self.MeanHeliumNumberDensity(z))
    
    def CriticalDensity(self, z):
        return (3.0 * self.HubbleParameter(z)**2) / (8.0 * np.pi * G)
    
    def dtdz(self, z):
        return 1. / self.HubbleParameter(z) / (1. + z) 
    
    def LuminosityDistance(self, z):
        """
        Returns luminosity distance in Mpc.  Assumes we mean distance from us (z = 0).
        """
        
        return (1. + z) * self.ComovingRadialDistance(0., z)
        
    def ComovingRadialDistance(self, z0, z):
        """
        Return comoving distance between redshift z0 and z, z0 < z.
        """
        
        if self.approx_highz:
            return 2. * c * ((1. + z0)**-0.5 - (1. + z)**-0.5) \
                / self.HubbleParameterNow / self.sqrtOmegaMatterNow
                
        # Otherwise, do the integral - normalize to H0 for numerical reasons
        integrand = lambda z: self.HubbleParameterNow / self.HubbleParameter(z)
        return c * quad(integrand, z0, z)[0] / self.HubbleParameterNow        
            
    def ProperRadialDistance(self, z0, z):
        return self.ComovingRadialDistance(z0, z) / (1. + z0)    
        
    def ComovingLineElement(self, z):
        """
        Comoving differential line element at redshift z.
        """
        
        return c / self.HubbleParameter(z)
        
    def ProperLineElement(self, z):
        """
        Proper differential line element at redshift z (i.e. dl/dz).
        """
        
        return self.ComovingLineElement(z) / (1. + z)
        
    def dldz(self, z):
        """ Proper differential line element. """
        return self.ProperLineElement(z)        
    
    def CriticalDensityForCollapse(self, z):
        """
        Generally denoted (in LaTeX format) \Delta_c, fit from 
        Bryan & Norman (1998).
        """            
        d = self.OmegaMatter(z) - 1.
        return 18. * np.pi**2 + 82. * d - 39. * d**2
    
            
    