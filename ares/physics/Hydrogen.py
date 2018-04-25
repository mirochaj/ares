"""

Hydrogen.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Mar 12 18:02:07 2012

Description: Container for hydrogen physics stuff.

"""

import scipy
import numpy as np
from types import FunctionType
from scipy.optimize import fsolve
from ..util.ReadData import _load_inits
from ..util.ParameterFile import ParameterFile
from ..util.Math import central_difference, interp1d
from .Constants import A10, T_star, m_p, m_e, erg_per_ev, h, c, E_LyA, E_LL, \
    k_B
    
try:
    from scipy.special import gamma
    g23 = gamma(2. / 3.)
    g13 = gamma(1. / 3.)
    c1 = 4. * np.pi / 3. / np.sqrt(3.) / g23
    c2 = 8. * np.pi / 3. / np.sqrt(3.) / g13
    
    from scipy.special import airy, hyp2f1
    
except ImportError:
    pass
    
_scipy_ver = scipy.__version__.split('.')

# This keyword didn't exist until version 0.14 
if float(_scipy_ver[1]) >= 14:
    _interp1d_kwargs = {'assume_sorted': True}
else:
    _interp1d_kwargs = {}

# Rate coefficients for spin de-excitation - from Zygelman originally

# H-H collisions.
T_HH = \
    [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, \
     25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, \
     90.0, 100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, \
     2000.0, 3000.0, 5000.0, 7000.0, 10000.0]

kappa_HH = \
    [1.38e-13, 1.43e-13, 2.71e-13, 6.60e-13, 1.47e-12, 2.88e-12, \
     9.10e-12, 1.78e-11, 2.73e-11, 3.67e-11, 5.38e-11, 6.86e-11, \
     8.14e-11, 9.25e-11, 1.02e-10, 1.11e-10, 1.19e-10, 1.75e-10, \
     2.09e-10, 2.56e-10, 2.91e-10, 3.31e-10, 4.27e-10, 4.97e-10, \
     6.03e-10, 6.87e-10, 7.87e-10]
            
# H-e collisions.            
T_He = \
    [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 
     1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 
     10000.0, 15000.0, 20000.0]
        
kappa_He = \
    [2.39e-10, 3.37e-10, 5.30e-10, 7.46e-10, 1.05e-9, 1.63e-9, 
     2.26e-9, 3.11e-9, 4.59e-9, 5.92e-9, 7.15e-9, 7.71e-9, 
     8.17e-9, 8.32e-9, 8.37e-9, 8.29e-9, 8.11e-9]

T_HH = np.array(T_HH)
T_He = np.array(T_He)

l_LyA = h * c / E_LyA / erg_per_ev

class Hydrogen(object):
    def __init__(self, cosm=None, **kwargs):
        
        self.pf = ParameterFile(**kwargs)

        if cosm is None:
            from .Cosmology import Cosmology
            self.cosm = Cosmology(**self.pf)
        else:
            self.cosm = cosm
            
        self.interp_method = self.pf['interp_cc']
        self.approx_S = self.pf['approx_Salpha']
        
        self.nmax = self.pf['lya_nmax']

        self.tabulated_coeff = \
            {'kappa_H': kappa_HH, 'kappa_e': kappa_He, 
             'T_H': T_HH, 'T_e': T_He}

    @property
    def kappa_H_pre(self):
        if not hasattr(self, '_kappa_H_pre'):                            
            self._kappa_H_pre = interp1d(T_HH, kappa_HH, 
                kind=self.interp_method, bounds_error=False, fill_value=0.0,
                **_interp1d_kwargs)

        return self._kappa_H_pre

    @property
    def kappa_e_pre(self):
        if not hasattr(self, '_kappa_e_pre'):     
            self._kappa_e_pre = interp1d(T_He, kappa_He,
                kind=self.interp_method, bounds_error=False, fill_value=0.0,
                **_interp1d_kwargs)

        return self._kappa_e_pre

    @property
    def Tk_hi_H(self):
        if not hasattr(self, '_Tk_hi_H'):
            self._Tk_hi_H = \
                np.logspace(np.log10(self.tabulated_coeff['T_H'].min()),
                np.log10(self.tabulated_coeff['T_H'].max()), 1000)
            
        return self._Tk_hi_H
        
    @property    
    def Tk_hi_e(self):
        if not hasattr(self, '_Tk_hi_e'):       
            self._Tk_hi_e = \
                np.logspace(np.log10(self.tabulated_coeff['T_e'].min()),
                np.log10(self.tabulated_coeff['T_e'].max()), 1000)
            
        return self._Tk_hi_e

    @property
    def kH_hi(self):
        if not hasattr(self, '_kH_hi'):
            self._kH_hi = np.array(list(map(self.kappa_H, self.Tk_hi_H)))
    
        return self._kH_hi

    @property
    def ke_hi(self):
        if not hasattr(self, '_ke_hi'):
            self._ke_hi = np.array(list(map(self.kappa_e, self.Tk_hi_e)))

        return self._ke_hi

    @property
    def dlogkH_dlogT(self):  
        if not hasattr(self, '_dlogkH_dlogT'):
            Tk_p_H, dkHdT = central_difference(self.Tk_hi_H, self.kH_hi)
            dlogkH_dlogT = dkHdT * Tk_p_H / np.array(list(map(self.kappa_H, Tk_p_H)))
            
            _kH_spline = interp1d(Tk_p_H, dlogkH_dlogT)
            self._dlogkH_dlogT = lambda T: _kH_spline(T)
            
        return self._dlogkH_dlogT
        
    @property
    def dlogke_dlogT(self):
        if not hasattr(self, '_dlogke_dlogT'):
            Tk_p_e, dkedT = central_difference(self.Tk_hi_e, self.ke_hi)
            dlogke_dlogT = dkedT * Tk_p_e / np.array(list(map(self.kappa_e, Tk_p_e)))
            
            _ke_spline = interp1d(Tk_p_e, dlogke_dlogT)
            self._dlogke_dlogT = lambda T: _ke_spline(T)
    
        return self._dlogke_dlogT
    
    def _kappa(self, Tk, Tarr, spline):
        if Tk < Tarr[0]:
            return spline(Tarr[0])
        elif Tk > Tarr[-1]:
            return spline(Tarr[-1])
        else:
            return spline(Tk)
                               
    def kappa_H(self, Tk):
        """
        Rate coefficient for spin-exchange via H-H collsions.
        """
        if type(Tk) in [int, float, np.float64]:            
            return self._kappa(Tk, T_HH, self.kappa_H_pre)
        else:
            tmp = np.zeros_like(Tk)
            for i in range(len(Tk)):
                tmp[i] = self._kappa(Tk[i], T_HH, self.kappa_H_pre)
            return tmp
            
    def kappa_e(self, Tk):       
        """
        Rate coefficient for spin-exchange via H-electron collsions.
        """                            
        if type(Tk) in [int, float, np.float64]:
            return self._kappa(Tk, T_He, self.kappa_e_pre)
        else:
            tmp = np.zeros_like(Tk)
            for i in range(len(Tk)):
                tmp[i] = self._kappa(Tk[i], T_He, self.kappa_e_pre)
            return tmp

    def photon_energy(self, nu, nl=1):
        """
        Return energy of photon transitioning from nu to nl in eV.  
        Defaults to Lyman-series.
        """
        return Ryd * (1. / nl / nl - 1. / nu / nu) / erg_per_ev

    def photon_freq(self, nu, nl=1):
        return self.photon_energy(nu, nl) * erg_per_ev / h 

    def zmax(self, z, n):
        return (1. + z) * (1. - (n + 1)**-2) / (1. - n**-2) - 1.
    
    def OnePhotonPerHatom(self, z):
        """
        Flux of photons = 1 photon per hydrogen atom assuming Lyman alpha 
        frequency.
        """
        
        return self.cosm.nH0 * (1. + z)**3 * c / 4. / np.pi / self.nu_alpha
    
    def frec(self, n):
        """ From Pritchard & Furlanetto 2006. """
        if n == 2:    return 1.0
        elif n == 3:  return 0.0
        elif n == 4:  return 0.2609
        elif n == 5:  return 0.3078
        elif n == 6:  return 0.3259
        elif n == 7:  return 0.3353
        elif n == 8:  return 0.3410
        elif n == 9:  return 0.3448
        elif n == 10: return 0.3476
        elif n == 11: return 0.3496
        elif n == 12: return 0.3512
        elif n == 13: return 0.3524
        elif n == 14: return 0.3535
        elif n == 15: return 0.3543
        elif n == 16: return 0.3550
        elif n == 17: return 0.3556
        elif n == 18: return 0.3561
        elif n == 19: return 0.3565
        elif n == 20: return 0.3569
        elif n == 21: return 0.3572
        elif n == 22: return 0.3575
        elif n == 23: return 0.3578
        elif n == 24: return 0.3580
        elif n == 25: return 0.3582
        elif n == 26: return 0.3584
        elif n == 27: return 0.3586
        elif n == 28: return 0.3587
        elif n == 29: return 0.3589
        elif n == 30: return 0.3590
        else:
            raise ValueError('Only know frec for 2 <= n <= 30!')

    @property    
    def Tbg(self):
        if not hasattr(self, '_Tbg'):
            if self.pf['Tbg'] is not None:
                if self.pf['Tbg'] == 'pl':
                    p = self.Tbg_pars
                    self._Tbg = lambda z: p[0] * ((1. + z) / (1. + p[1]))**p[2]
                elif type(self.pf['Tbg']) is FunctionType:
                    self._Tbg = self.pf['Tbg']
                else:
                    raise NotImplemented('help')
            else:
                self._Tbg = None
    
        return self._Tbg
    
    @Tbg.setter
    def Tbg(self, value):
        """
        Must be a function of redshift.
        """
        self._Tbg = value
    
    @property
    def Tbg_pars(self):
        if not hasattr(self, '_Tbg_pars'):
            self._Tbg_pars = [self.pf['Tbg_p{}'.format(i)] for i in range(5)]
        return self._Tbg_pars            
        
    #def Tref(self, z):
    #    """
    #    Compute background temperature.
    #    """
    #
    #    return self.cosm.TCMB(z) + self.Tbg(z)

    def CollisionalCouplingCoefficient(self, z, Tk, xHII, ne, Tr=0.0):
        """
        Parameters
        ----------
        z : float
            Redshift
        Tk : float
            Kinetic temperature of the gas [K]
        xHII : float
            Hydrogen ionized fraction
        ne : float
            Proper electron density [cm**-3]
        
        References
        ----------
        Zygelman, B. 2005, ApJ, 622, 1356
        
        """
        sum_term = self.cosm.nH(z) * (1. - xHII) * self.kappa_H(Tk) \
            + ne * self.kappa_e(Tk)
              
        Tref = self.cosm.TCMB(z) + Tr
                
        return sum_term * T_star / A10 / Tref
    
    def RadiativeCouplingCoefficient(self, z, Ja, Tk=None, xHII=None, Tr=0.0):
        """
        Return radiative coupling coefficient (i.e., Wouthuysen-Field effect).
        
        .. note :: If approx_Sa > 3, will return x_a, S_a, and T_S, which are
            determined iteratively.
        
        """
        
        if self.approx_S < 4:
            return self.xalpha_tilde(z) * self.Sa(z=z, Tk=Tk, xHII=xHII) * Ja

        ##
        # Must solve iteratively.
        ##
        
        # Hirata (2006)
        if self.approx_S == 4:
            xi = (1e-7 * self.tauGP(z, xHII=xHII))**(1./3.) * Tk**(-2./3.)
            a = lambda Ts: 1. - 0.0631789 / Tk + 0.115995 / Tk**2 \
                - 0.401403 / Ts / Tk + 0.336463 / Ts / Tk**2
            b = 1. + 2.98394 * xi + 1.53583 * xi**2 + 3.85289 * xi**3
        
            Sa = lambda Ts: a(Ts) / b
            ne = 0.0 # fix
            
            #Tk = float(Tk)
            
            xc = self.CollisionalCouplingCoefficient(z, Tk, xHII, ne)
            xa = lambda Ts: self.xalpha_tilde(z) * Sa(Ts) * Ja
            Tcmb = self.cosm.TCMB(z)
            Trad = Tcmb + Tr
            
            Tr_inv = 1. / Trad
            Tk_inv = 1. / Tk
            Tc_inv = lambda Ts: Tk**-1. + 0.405535 / Tk / (Ts**-1. - Tk**-1.)
            
            Ts_inv = lambda Ts: (Tr_inv + xa(Ts) * Tc_inv(Ts) + xc * Tk_inv) \
                   / (1. + xa(Ts) + xc)
            
            to_solve = lambda Ts: 1. / Ts - Ts_inv(Ts)
            
            assert type(z) is not np.ndarray
            assert type(Tk) is not np.ndarray
        
            x = fsolve(to_solve, Tcmb, full_output=True)            
            Ts = float(x[0])
                                    
            return xa(Ts), Sa(Ts), Ts
        else:
            raise NotImplemented('approx_Salpha>4 not currently supported!')  
            
    def tauGP(self, z, xHII=0.):
        """ Gunn-Peterson optical depth. """
        return 1.5 * self.cosm.nH(z) * (1. - xHII) * l_LyA**3 * 50e6 \
            / self.cosm.HubbleParameter(z)

    def lya_width(self, Tk):
        """
        Returns Doppler line-width of the Ly-a line in eV.
        """
        return np.sqrt(2. * k_B * Tk / m_e / c**2) * E_LyA

    def xalpha_tilde(self, z):
        gamma = 5e7 # 50 MHz
        return 8. * np.pi * l_LyA**2 * gamma * T_star \
            / 9. / A10 / self.cosm.TCMB(z)

    def Sa(self, z=None, Tk=None, xHII=0.0, Ts=None, Ja=0.0):
        """
        Account for line profile effects.
        """

        if self.approx_S == 0:
            raise NotImplementedError('Must use analytical formulae.')
        elif self.approx_S == 1:
            S = 1.0
        elif self.approx_S == 2:
            S = np.exp(-0.37 * np.sqrt(1. + z) * Tk**(-2./3.)) \
                / (1. + 0.4 / Tk)
        elif int(self.approx_S) == 3:
            gamma = 1. / self.tauGP(z, xHII=xHII) / (1. + 0.4 / Tk)  # Eq. 4
            alpha = 0.717 * Tk**(-2./3.) * (1e-6 / gamma)**(1. / 3.) # Eq. 20
            
            # Gamma function approximation: Eq. 19
            if self.approx_S % 1 != 0:
                # c1 = 4. * np.pi / 3. / np.sqrt(3.) / Gamma(2/3)
                # c2 = 8. * np.pi / 3. / np.sqrt(3.) / Gamma(1/3)
                S = 1. - c1 * alpha + c2 * alpha**2 - 4. * alpha**3 / 3.
            # Actual solution: Eq. 18
            else:
                Ai, Aip, Bi, Bip = airy(-2. * alpha / 3.**(1./3.))
                F2 = hyp2f1(1., 4./3., 5./3., -8 * alpha**3 / 27.)
                S = 1. - 4. * alpha \
                    * (3.**(2./3.) * np.pi * Bi + 3 * alpha**2 * F2) / 9.
            
        elif self.approx_S == 4:
            xa, S, Ts = self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII)
        else:
            raise NotImplementedError('approx_Sa must be in [1,2,3,4].')
                
        return np.maximum(S, 0.0)

    def ELyn(self, n):
        """ Return energy of Lyman-n photon in eV. """
        return E_LL * (1. - 1. / n**2)

    @property
    def Ts_floor_pars(self):
        if not hasattr(self, '_Ts_floor_pars'):
            self._Ts_floor_pars = [self.pf['floor_Ts_p{}'.format(i)] for i in range(6)]
        return self._Ts_floor_pars    

    @property
    def Ts_floor(self):
        if not hasattr(self, '_Ts_floor'):
            if not self.pf['floor_Ts']:
                self._Ts_floor = lambda z: 0.0
            else:
                if self.pf['floor_Ts'] == 'pl':
                    pars = self.Ts_floor_pars
                    
                    self._Ts_floor = lambda z: pars[0] * ((1. + z) / pars[1])**pars[2]
                    
                else:
                    raise NotImplemented('sorry!')
                    
        return self._Ts_floor

    @Ts_floor.setter
    def Ts_floor(self, value):
        self._Ts_floor = value

    def Ts(self, z, Tk, Ja, xHII, ne, Tr=0.0):
        """
        Short-hand for calling `SpinTemperature`.
        """
        return self.SpinTemperature(z, Tk, Ja, xHII, ne, Tr)

    def SpinTemperature(self, z, Tk, Ja, xHII, ne, Tr=0.0):
        """
        Returns spin temperature of intergalactic hydrogen.

        Parameters
        ----------
        z : float, np.ndarray
            Redshift
        Tk : float, np.ndarray
            Gas kinetic temperature
        Ja : float, np.ndarray
            Lyman-alpha flux in units of [s**-1 cm**-2 Hz**-1 sr**-1]
        xHII : float, np.ndarray 
            Hydrogen ionized fraction
        ne : float, np.ndarray
            Proper electron density in [cm**-3]
            
        Returns
        -------
        Spin temperature in Kelvin.
         
        """
        
        x_c = self.CollisionalCouplingCoefficient(z, Tk, xHII, ne)
        
        if self.approx_S < 4:
            x_a = self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII, Tr)
            Tc = Tk

            Tref = self.cosm.TCMB(z) + Tr

            Ts = (1.0 + x_c + x_a) / \
                (Tref**-1. + x_c * Tk**-1. + x_a * Tc**-1.)
        else:
            x_a, S, Ts = self.RadiativeCouplingCoefficient(z, Ja, Tk, xHII, Tr)
                    
        return np.maximum(Ts, self.Ts_floor(z=z))

    def dTb(self, z, xHII, Ts, Tr=0.0):
        """
        Short-hand for calling `DifferentialBrightnessTemperature`.
        """
        return self.DifferentialBrightnessTemperature(z, xHII, Ts, Tr)

    def DifferentialBrightnessTemperature(self, z, xavg, Ts, Tr=0.0):
        """
        Global 21-cm signature relative to cosmic microwave background in mK.

        Parameters
        ----------
        z : float, np.ndarray
            Redshift
        xavg : float, np.ndarray 
            Volume-averaged ionized fraction, i.e., a weighted average 
            between the volume of fully ionized gas and the semi-neutral
            bulk IGM beyond.
        Ts : float, np.ndarray
            Spin temperature of intergalactic hydrogen.

        Returns
        -------
        Differential brightness temperature in milli-Kelvin.

        """
        
        Tref = self.cosm.TCMB(z) + Tr
        return 27. * (1. - xavg) * \
            (self.cosm.omega_b_0 * self.cosm.h70**2 / 0.023) * \
            np.sqrt(0.15 * (1.0 + z) / self.cosm.omega_m_0 / self.cosm.h70**2 / 10.) * \
            (1.0 - Tref / Ts)
    
    @property
    def inits(self):
        if not hasattr(self, '_inits'):
            self._inits = _load_inits()
        return self._inits

    def saturated_limit(self, z):
        return self.DifferentialBrightnessTemperature(z, 0.0, np.inf)

    def adiabatic_floor(self, z):
        Tk = np.interp(z, self.inits['z'], self.inits['Tk'])
        Ts = self.SpinTemperature(z, Tk, 1e50, 0.0, 0.0)        
        return self.DifferentialBrightnessTemperature(z, 0.0, Ts)

    def dTb_no_astrophysics(self, z):
        Ts = self.SpinTemperature(z, self.cosm.Tgas(z), 0., 0., 0.)
        return self.dTb(z, 0.0, Ts)
