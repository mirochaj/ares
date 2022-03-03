"""

Parametric21cm.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Wed Dec 14 13:29:57 PST 2016

Description:

"""

import numpy as np
from types import FunctionType
from ..util import ParameterFile
from ..physics import Hydrogen, Cosmology
from ..physics.Constants import k_B, J21_num, nu_0_mhz
from ..util.ParameterFile import par_info, get_pq_pars
from .ParameterizedQuantity import ParameterizedQuantity

class Parametric21cm(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

        # Cosmology class
        self.cosm = Cosmology(**self.pf)

        # Create instance of Hydrogen class
        self.hydr = Hydrogen(cosm=self.cosm,
            approx_Salpha=self.pf['approx_Salpha'], **kwargs)

    def electron_density(self, z):
        return np.interp(z, self.cosm.thermal_history['z'],
            self.cosm.thermal_history['xe']) * self.cosm.nH(z)

    def __getattr__(self, name):
        # Indicates that this attribute is being accessed from within a
        # property. Don't want to override that behavior!
        if (name[0] == '_'):
            raise AttributeError('This will get caught. Don\'t worry!')

        full_name = 'pop_' + name

        # Now, possibly make an attribute
        if name not in self.__dict__.keys():

            if type(self.pf[full_name]) is FunctionType:
                self.__dict__[name] = self.pf[full_name]
                return self.__dict__[name]

            is_php = self.pf[full_name][0:2] == 'pq'

            if not is_php:
                self.__dict__[name] = lambda z: 0.0

            pars = get_pq_pars(self.pf[full_name], self.pf)

            self.__dict__[name] = ParameterizedQuantity(**pars)

        return self.__dict__[name]

    def __call__(self, z):
        """
        Check "tanh_pars" for list of acceptable parameters.

        Returns
        -------
        ares.analysis.-21cm instance, which contains the entire signal
        and the turning points conveniently in the "turning_points"
        attribute.

        """

        Ja = self.Ja(z=z)
        Tk = self.Tk(z=z) + self.cosm.Tgas(z)
        xi = self.xi(z=z)
        ne = self.electron_density(z)

        # Spin temperature
        Ts = self.hydr.SpinTemperature(z, Tk, Ja, 0.0, ne)

        # Brightness temperature
        dTb = self.hydr.get_21cm_dTb(z, Ts, xavg=xi)

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
         'igm_heat_h_1': np.zeros_like(z),#self.heating_rate(z, Tref, zref_T, dz_T),
         'cgm_Gamma_h_1': np.zeros_like(z),#self.ionization_rate(z, xref, zref_x, dz_x),
        }

        return hist
