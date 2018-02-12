""" 

Cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description:

"""

import numpy as np
from scipy.misc import derivative
from scipy.optimize import fsolve
from scipy.integrate import quad, ode
from ..util.ReadData import _load_inits
from ..util.ParameterFile import ParameterFile
from .Constants import c, G, km_per_mpc, m_H, m_He, sigma_SB, g_per_msun, \
    cm_per_mpc

class Cosmology(object):
    def __init__(self, **kwargs):        
        self.pf = ParameterFile(**kwargs)
                
        self.omega_m_0 = self.pf['omega_m_0']
        self.omega_b_0 = self.pf['omega_b_0']
        self.omega_l_0 = self.pf['omega_l_0']
        self.omega_cdm_0 = self.omega_m_0 - self.omega_b_0

        self.hubble_0 = self.pf['hubble_0'] * 100 / km_per_mpc
        self.cmb_temp_0 = self.pf['cmb_temp_0']
        self.approx_highz = self.pf['approx_highz']
        self.approx_lowz = False
        self.sigma_8 = self.sigma8 = self.pf['sigma_8']
        self.primordial_index = self.pf['primordial_index']
        
        self.CriticalDensityNow = self.rho_crit_0 = \
            (3 * self.hubble_0**2) / (8 * np.pi * G)
        
        self.h70 = self.pf['hubble_0']
        
        self.mean_density0 = self.omega_m_0 * self.rho_crit_0 \
            * cm_per_mpc**3 / g_per_msun / self.h70**2
        
        if self.pf['helium_by_number'] is None:
            self.helium_by_mass = self.Y = self.pf['helium_by_mass']
            self.helium_by_number = self.y = 1. / (1. / self.Y - 1.) / 4.
        else:
            self.helium_by_number = self.y = self.pf['helium_by_number']
            self.Y = self.helium_by_mass = 4. * self.y / (1. + 4. * self.y)
        
        self.X = 1. - self.Y
        
        self.g_per_baryon = self.g_per_b = m_H / (1. - self.Y) / (1. + self.y)
        self.b_per_g = 1. / self.g_per_baryon
        self.baryon_per_Msun = self.b_per_msun = g_per_msun / self.g_per_baryon
         
        # Decoupling (gas from CMB) redshift       
        self.zdec = 150. * (self.omega_b_0 * self.h70**2 / 0.023)**0.4 - 1.

        # Matter/Lambda equality
        #if self.omega_l_0 > 0:
        self.a_eq = (self.omega_m_0 / self.omega_l_0)**(1./3.)
        self.z_eq = 1. / self.a_eq - 1.
        
        # Common
        self.Omh2 = self.omega_b_0 * self.h70**2

        # Hydrogen, helium, electron, and baryon densities today (z = 0)
        self.rho_b_z0 = self.MeanBaryonDensity(0)
        self.rho_m_z0 = self.MeanMatterDensity(0)
        self.rho_cdm_z0 = self.rho_m_z0 - self.rho_b_z0
        self.nH0 = (1. - self.Y) * self.rho_b_z0 / m_H
        self.nHe0 = self.y * self.nH0
        self.ne0 = self.nH0 + 2. * self.nHe0
        #self.n0 = self.nH0 + self.nHe0 + self.ne0
        
        self.nH = lambda z: self.nH0 * (1. + z)**3
        self.nHe = lambda z: self.nHe0 * (1. + z)**3
        
        self.delta_c0 = 1.686
        self.TcmbNow = self.cmb_temp_0
        
        self.fbaryon = self.omega_b_0 / self.omega_m_0
        self.fcdm = self.omega_cdm_0 / self.omega_m_0
        
        self.fbar_over_fcdm = self.fbaryon / self.fcdm
        
        # Used in hmf
        self.pars = {'omega_lambda':self.omega_l_0,
         'omega_b':self.omega_b_0,
         'omega_M':self.omega_m_0,
         'sigma_8':self.sigma8,
         'n': self.primordial_index}
                 
    @property
    def inits(self):
        if not hasattr(self, '_inits'):
            self._inits = _load_inits()
        return self._inits
        
    def TimeToRedshiftConverter(self, t_i, t_f, z_i):
        """
        High redshift approximation under effect.
        """
        return ((1. + z_i)**-1.5 + (3. * self.hubble_0 * 
            np.sqrt(self.omega_m_0) * (t_f - t_i) / 2.))**(-2. / 3.) - 1.
        
    def LookbackTime(self, z_i, z_f):
        """
        Returns lookback time from z_i to z_f in seconds, where z_i < z_f.
        """

        return self.t_of_z(z_i) - self.t_of_z(z_f)

    def TCMB(self, z):
        return self.cmb_temp_0 * (1. + z)
        
    def UCMB(self, z):
        """ CMB energy density. """
        return 4.0 * sigma_SB * self.TCMB(z)**4 / c
    
    def t_of_z(self, z):
        """
        Time-redshift relation for a matter + lambda Universe.
        
        References
        ----------
        Ryden, Equation 6.28
        
        Returns
        -------
        Time since Big Bang in seconds.
        
        """
        #if self.approx_highz:
        #    pass
        #elif self.approx_lowz:
        #    pass
            
        # Full calculation
        a = 1. / (1. + z)
        t = (2. / 3. / np.sqrt(1. - self.omega_m_0)) \
            * np.log((a / self.a_eq)**1.5 + np.sqrt(1. + (a / self.a_eq)**3.)) \
            / self.hubble_0

        return t
        
    def z_of_t(self, t):
        C = np.exp(1.5 * self.hubble_0 * t * np.sqrt(1. - self.omega_m_0))
        
        a = self.a_eq * (C**2 - 1.)**(2./3.) / (2. * C)**(2./3.)
        return (1. / a) - 1.
        
    def Tgas(self, z):
        """
        Gas kinetic temperature at z assuming only adiabatic cooling after zdec.
        
        .. note :: This is very approximate. Use RECFAST or CosmoRec for more
            precise solutions.
            
        """
        
        if self.pf['approx_thermal_history'] == 'piecewise':
            if z >= self.zdec:
                return self.TCMB(z)
            else:
                return self.TCMB(self.zdec) * (1. + z)**2 / (1. + self.zdec)**2
        elif self.pf['approx_thermal_history']:
            return np.interp(z, self.thermal_history['z'], 
                self.thermal_history['Tk'])
        elif not self.pf['approx_thermal_history']:
            return np.interp(z, self.inits['z'], self.inits['Tk'])

    @property
    def thermal_history(self):
        if not hasattr(self, '_thermal_history'):
            
            if not self.pf['approx_thermal_history']:
                self._thermal_history = self.inits
                return self._thermal_history
            
            z0 = max(self.inits['z'])
            solver = ode(self.cooling_rate)
            solver.set_integrator('vode', method='bdf')
            solver.set_initial_value([np.interp(z0, self.inits['z'],
                self.inits['Tk'])], z0)
                
            dz = self.pf['inits_Tk_dz']
            zf = final_redshift = 1.
            zall = []; Tall = []
            while solver.successful() and solver.t > zf:
                
                if solver.t-dz < 0:
                    break
                                
                zall.append(solver.t)
                Tall.append(solver.y[0])
                solver.integrate(solver.t-dz)
            
            self._thermal_history = {}
            self._thermal_history['z'] = np.array(zall)[-1::-1]
            self._thermal_history['Tk'] = np.array(Tall)[-1::-1]
            self._thermal_history['xe'] = 1e-3 * np.ones_like(zall)
        
        return self._thermal_history
            
    @property
    def cooling_pars(self):
        if not hasattr(self, '_cooling_pars'):
            self._cooling_pars = [self.pf['inits_Tk_p{}'.format(i)] for i in range(5)]
        return self._cooling_pars    
            
    def cooling_rate(self, z, T=None):
        if self.pf['approx_thermal_history'] in ['exp', 'tanh', 'exp+gauss']:

            # This shouldn't happen! Argh.
            if z < 0:
                return np.nan

            t = self.t_of_z(z)
            dtdz = self.dtdz(z)
            return (T / t) * self.log_cooling_rate(z) * -1. * dtdz
        else:
            return derivative(self.Tgas, z)

    def log_cooling_rate(self, z):
        if self.pf['approx_thermal_history'] == 'exp':
            pars = self.cooling_pars
            norm = -(2. + pars[2]) # Must be set so high-z limit -> -2/3
            return norm * (1. - np.exp(-(z / pars[0])**pars[1])) / 3. \
                   + pars[2] / 3.
        elif self.pf['approx_thermal_history'] == 'exp+gauss':
            pars = self.cooling_pars
            return 2. * (1. - np.exp(-(z / pars[0])**pars[1])) / 3. \
                - (4./3.) * (1. + pars[2] * np.exp(-((z - pars[3]) / pars[4])**2))
        elif self.pf['approx_thermal_history'] == 'tanh':
            pars = self.cooling_pars
            return (-2./3.) - (2./3.) * 0.5 * (np.tanh((pars[0] - z) / pars[1]) + 1.)
        else:
            return -1. * self.cooling_rate(z, self.Tgas(z)) \
                * (self.t_of_z(z) / self.Tgas(z)) / self.dtdz(z)

    @property
    def z_dec(self):
        if not hasattr(self, '_z_dec'):
            to_min = lambda zz: np.abs(self.log_cooling_rate(zz) + 1.)
            self._z_dec = fsolve(to_min, 150.)[0]
        return self._z_dec    

    @property
    def Tk_dec(self):
        return self.Tgas(self.z_dec)
    
    def EvolutionFunction(self, z):
        return self.omega_m_0 * (1.0 + z)**3  + self.omega_l_0
        
    def HubbleParameter(self, z):
        if self.approx_highz:
            return self.hubble_0 * np.sqrt(self.omega_m_0) * (1. + z)**1.5
        return self.hubble_0 * np.sqrt(self.EvolutionFunction(z))
    
    def HubbleLength(self, z):
        return c / self.HubbleParameter(z)
    
    def HubbleTime(self, z):
        return 1. / self.HubbleParameter(z)
        
    def OmegaMatter(self, z):
        if self.approx_highz:
            return 1.0
        return self.omega_m_0 * (1. + z)**3 / self.EvolutionFunction(z)
    
    def OmegaLambda(self, z):
        if self.approx_highz:
            return 0.0
        
        return self.omega_l_0 / self.EvolutionFunction(z)
    
    def MeanMatterDensity(self, z):
        return self.OmegaMatter(z) * self.CriticalDensity(z)
        
    def MeanBaryonDensity(self, z):
        return (self.omega_b_0 / self.omega_m_0) * self.MeanMatterDensity(z)
    
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
        Returns luminosity distance in cm.  Assumes we mean distance from us (z = 0).
        """
        
        integr = quad(lambda z: self.hubble_0 / self.HubbleParameter(z), 
            0.0, z)[0]
        
        return integr * c * (1. + z) / self.hubble_0
        
    def ComovingRadialDistance(self, z0, z):
        """
        Return comoving distance between redshift z0 and z, z0 < z.
        """
        
        if self.approx_highz:
            return 2. * c * ((1. + z0)**-0.5 - (1. + z)**-0.5) \
                / self.hubble_0 / np.sqrt(self.omega_m_0)
                
        # Otherwise, do the integral - normalize to H0 for numerical reasons
        integrand = lambda z: self.hubble_0 / self.HubbleParameter(z)
        return c * quad(integrand, z0, z)[0] / self.hubble_0  
            
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
    
            
