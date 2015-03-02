"""

TanhModel.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep  9 16:25:41 MDT 2014

Description: 

"""

import time
import numpy as np
from ..util import ParameterFile
from ..analysis.Global21cm import Global21cm
from ..util.ReadData import load_inits
from ..physics import Hydrogen, Cosmology
from ..physics.Constants import k_B, J21_num, nu_0_mhz
from ..physics.RateCoefficients import RateCoefficients
from ..util.SetDefaultParameterValues import TanhParameters

# Default parameters
tanh_kwargs = TanhParameters()

tanh_pars = ['tanh_J0', 'tanh_Jz0', 'tanh_Jdz', 
    'tanh_T0', 'tanh_Tz0', 'tanh_Tdz', 'tanh_x0', 'tanh_xz0', 'tanh_xdz',
    'tanh_bias_temp', 'tanh_bias_freq']

def tanh_generic(z, zref, dz):
    return 0.5 * (np.tanh((zref - z) / dz) + 1.)

rc = RateCoefficients(recombination='A')
alpha_A = rc.RadiativeRecombinationRate(0, 1e4)

z_to_mhz = lambda z: nu_0_mhz / (1. + z)
freq_to_z = lambda nu: nu_0_mhz / nu - 1.

def shift_z(z, nu_bias):
    nu = np.array(map(z_to_mhz, z))
    nu += nu_bias

    return freq_to_z(nu)
    
tanh_gjah_to_ares = \
{
 'J_0/J_21': 'tanh_J0',
 'dz_J': 'tanh_Jdz',
 'z_{0,j}': 'tanh_Jz0',
 'T_0': 'tanh_T0',
 'dz_T': 'tanh_Tdz',
 'z_{0,T}': 'tanh_Tz0',
 'x_0': 'tanh_x0',
 'dz_x': 'tanh_xdz',
 'z_{0,x}': 'tanh_xz0',
}    

class TanhModel:
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

        # Cosmology class
        self.cosm = Cosmology(omega_m_0=self.pf["omega_m_0"], 
            omega_l_0=self.pf["omega_l_0"], 
            omega_b_0=self.pf["omega_b_0"], 
            hubble_0=self.pf["hubble_0"], 
            helium_by_number=self.pf['helium_by_number'], 
            cmb_temp_0=self.pf["cmb_temp_0"], 
            approx_highz=self.pf["approx_highz"])
        
        # Create instance of Hydrogen class
        self.hydr = Hydrogen(cosm=self.cosm,
            approx_Salpha=self.pf['approx_Salpha'])

        if self.pf['load_ics']:
            CR = load_inits()
            self.CR_TK = lambda z: np.interp(z, CR['z'], CR['Tk'])
            self.CR_ne = lambda z: np.interp(100, CR['z'], CR['xe']) \
                * self.cosm.nH(z)

    def Tgas_adiabatic(self, z):
        if self.pf['load_ics']:
            return self.CR_TK(z)
        else:    
            # Cosmology.Tgas can't handle arrays
            return self.cosm.TCMB(self.cosm.zdec) * (1. + z)**2 \
                / (1. + self.cosm.zdec)**2
                
    def electron_density(self, z):
        if self.pf['load_ics']:
            return self.CR_ne(z)
        else:
            return 0.0
        
    def temperature(self, z, Tref, zref, dz):
        return Tref * tanh_generic(z, zref=zref, dz=dz) \
            + self.Tgas_adiabatic(z)

    def ionized_fraction(self, z, xref, zref, dz):
        return xref * tanh_generic(z, zref=zref, dz=dz)

    def heating_rate(self, z, Tref, zref, dz, Lambda=None):

        Tk = self.temperature(z, Tref, zref, dz)

        dtdz = self.cosm.dtdz(z)
    
        dTkdz = 0.5 * Tref * (1. - np.tanh((zref - z) / dz)**2) / dz
        dTkdt = dTkdz / dtdz
    
        n = self.cosm.nH(z)
    
        if Lambda is None:
            cool = 0.0
        else:
            cool = Lambda(z)
    
        return 1.5 * n * k_B * (dTkdt + 2. * self.cosm.HubbleParameter(z) * Tk) \
            + cool / dtdz
        
    def ionization_rate(self, z, xref, zref, dz, C=1.):
        xi = self.ionized_fraction(z, xref, zref, dz)
        
        dtdz = self.cosm.dtdz(z)
        
        dxdz = 0.5 * xref * (1. - np.tanh((zref - z) / dz)**2) / dz
        dxdt = dxdz / dtdz
        
        n = self.cosm.nH(z)
        
        return dxdt + alpha_A * C * n * xi
        
    def __call__(self, z, **kwargs):
        """
        Check "tanh_pars" for list of acceptable parameters.
        
        Returns
        -------
        glorb.analysis.-21cm instance, which contains the entire signal
        and the turning points conveniently in the "turning_points" 
        attribute.
        
        """
        
        kw = tanh_kwargs.copy()
        
        for element in kwargs:
            kw.update({element: kwargs[element]})
            
        theta = [kw[par] for par in tanh_pars]
        
        return self.tanh_model_for_emcee(z, theta)
        
    def tanh_model_for_emcee(self, z, theta):
        """
        Compute 21-cm signal given tanh model parameters.
        
        Input parameters assumed to be in the following order:
        
        Jref, zref_J, dz_J, 
        Tref, zref_T, dz_T, 
        xref, xref_J, dz_x
        
        where Jref, Tref, and xref are the step heights, zref_? are the step
        locations, and dz_? are the step-widths.
        
        Note that Jref is in units of J21.
        
        Returns
        -------
        ares.analysis.Global21cm instance, which contains the entire signal
        and the turning points conveniently in the "turning_points" 
        attribute.
        
        """

        # Unpack parameters
        Jref, zref_J, dz_J, Tref, zref_T, dz_T, xref, zref_x, dz_x, \
            bias_T, bias_freq = theta

        Jref *= J21_num

        # Assumes z < zdec
        Tgas = self.Tgas_adiabatic(z)
        ne = self.electron_density(z)

        Ja = Jref * tanh_generic(z, zref=zref_J, dz=dz_J)
        Tk = Tref * tanh_generic(z, zref=zref_T, dz=dz_T) + Tgas
        xi = xref * tanh_generic(z, zref=zref_x, dz=dz_x)

        # Compute (proper) electron density assuming xHII = xHeII, xHeIII = 0.
        # Needed for collisional coupling.

        # Spin temperature
        Ts = self.hydr.SpinTemperature(z, Tk, Ja, xi, ne)

        # Brightness temperature
        dTb = self.hydr.DifferentialBrightnessTemperature(z, xi, Ts)

        if bias_T != 0:
            dTb += bias_T
        if bias_freq != 0:
            z = shift_z(z, bias_freq)

        # Save some stuff
        hist = \
        {
         'z': z,
         'dTb': dTb,
         'igm_Tk': Tk,
         'Ts': Ts,
         'Ja': Ja,
         'cgm_h_2': xi,
         'igm_h_1': np.ones_like(z),
         'igm_heat_h_1': self.heating_rate(z, Tref, zref_T, dz_T),
         'cgm_Gamma_h_1': self.ionization_rate(z, xref, zref_x, dz_x),
        }

        tmp = Global21cm(history=hist)    

        return tmp
