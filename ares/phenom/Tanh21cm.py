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
from scipy.misc import derivative
from ..physics import Hydrogen, Cosmology
from ..physics.Constants import k_B, J21_num, nu_0_mhz
from ..physics.RateCoefficients import RateCoefficients
from ..util.SetDefaultParameterValues import TanhParameters

# Default parameters
tanh_kwargs = TanhParameters()

tanh_pars = ['tanh_J0', 'tanh_Jz0', 'tanh_Jdz',
    'tanh_T0', 'tanh_Tz0', 'tanh_Tdz', 'tanh_x0', 'tanh_xz0', 'tanh_xdz',
    'tanh_bias_temp', 'tanh_bias_freq', 'tanh_scale_temp', 'tanh_scale_freq']

def tanh_generic(z, zref, dz):
    return 0.5 * (np.tanh((zref - z) / dz) + 1.)

rc = RateCoefficients(recombination='A')
alpha_A = rc.RadiativeRecombinationRate(0, 1e4)

z_to_mhz = lambda z: nu_0_mhz / (1. + z)
freq_to_z = lambda nu: nu_0_mhz / nu - 1.

def shift_z(z, nu_bias, nu_scale):
    nu = np.array(list(map(z_to_mhz, z)))
    nu = ((nu_scale * nu) + nu_bias)

    return freq_to_z(nu)

class Tanh21cm(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

        # Cosmology class
        self.cosm = Cosmology(**self.pf)

        # Create instance of Hydrogen class
        self.hydr = Hydrogen(cosm=self.cosm, **kwargs)

    def dTgas_dz(self, z):
        return derivative(self.cosm.Tgas, x0=z)

    def electron_density(self, z):
        return np.interp(z, self.cosm.thermal_history['z'],
            self.cosm.thermal_history['xe']) * self.cosm.nH(z)

    def temperature(self, z, Tref, zref, dz):
        return Tref * tanh_generic(z, zref=zref, dz=dz) \
            + self.cosm.Tgas(z)

    def ionized_fraction(self, z, xref, zref, dz):
        return xref * tanh_generic(z, zref=zref, dz=dz)

    def heating_rate(self, z, Tref, zref, dz):
        """
        Compute heating rate coefficient.
        """

        Tk = self.temperature(z, Tref, zref, dz)

        dtdz = self.cosm.dtdz(z)

        dTkdz = 0.5 * Tref * (1. - np.tanh((zref - z) / dz)**2) / dz
        dTkdt = dTkdz / dtdz

        #dTgas_dt = self.dTgas_dz(z) / dtdz

        return 1.5 * k_B * dTkdt

    def ionization_rate(self, z, xref, zref, dz, C=1.):
        """
        Compute ionization rate coefficient.
        """

        xi = self.ionized_fraction(z, xref, zref, dz)

        dtdz = self.cosm.dtdz(z)

        dxdz = 0.5 * xref * (1. - np.tanh((zref - z) / dz)**2) / dz
        dxdt = dxdz / dtdz

        n = self.cosm.nH(z)

        # Assumes ne = nH (bubbles assumed fully ionized)
        return (dxdt + alpha_A * C * n * xi) / (1. - xi)

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
            bias_T, bias_freq, scale_T, scale_freq = theta

        Jref *= J21_num

        # Assumes z < zdec
        Tgas = self.cosm.Tgas(z)
        ne = self.electron_density(z)

        Ja = Jref * tanh_generic(z, zref=zref_J, dz=dz_J)
        Tk = Tref * tanh_generic(z, zref=zref_T, dz=dz_T) + Tgas
        xi = xref * tanh_generic(z, zref=zref_x, dz=dz_x)

        # Spin temperature
        Ts = self.hydr.SpinTemperature(z, Tk, Ja, 0.0, ne)

        # Brightness temperature
        dTb = self.hydr.get_21cm_dTb(z, Ts, xavg=xi)

        if (bias_T != 0) or (scale_T != 1):
            dTb = ((scale_T * dTb) + bias_T)
        if (bias_freq != 0) or (scale_freq != 1):
            z = shift_z(z, bias_freq, scale_freq)

        # Save some stuff
        hist = \
        {
         'z': z,
         'dTb': dTb,
         'igm_dTb': dTb,
         'igm_Tk': Tk,
         'igm_Ts': Ts,
         'Ts': Ts,
         'Ja': Ja,
         'Jlw': np.zeros_like(z),
         'cgm_h_2': xi,
         'igm_h_1': np.ones_like(z),
         'igm_h_2': np.zeros_like(z),
         'cgm_k_ion_h_1': self.ionization_rate(z, xref, zref_x, dz_x),
         'cgm_k_ion_he_1': np.zeros_like(z),
         'cgm_k_ion_he_2': np.zeros_like(z),
         'igm_k_ion_h_1': np.zeros_like(z),
         'igm_k_ion_he_1': np.zeros_like(z),
         'igm_k_ion_he_2': np.zeros_like(z),
         'igm_k_heat_h_1': self.heating_rate(z, Tref, zref_T, dz_T),
         'igm_k_heat_he_1': np.zeros_like(z),
         'igm_k_heat_he_2': np.zeros_like(z),
        }

        return hist
