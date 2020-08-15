"""

CosmologyCCL.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Sat 15 Aug 2020 14:59:44 EDT

Description:

"""

import numpy as np
from .Cosmology import CosmologyARES
from .Constants import cm_per_mpc

try:
    import pyccl
except ImportError:
    pass

class CosmologyCCL(CosmologyARES):

    @property
    def _ccl_instance(self):
        if not hasattr(self, '_ccl_instance_'):
            self._ccl_instance_ = pyccl.Cosmology(Omega_c=self.omega_cdm_0,
                Omega_b=self.omega_b_0, h=self.h70, n_s=self.primordial_index,
                sigma8=self.sigma_8,
                transfer_function='bbks')

        return self._ccl_instance_

    def MeanMatterDensity(self, z):
        return pyccl.rho_x(self._ccl_instance, 1./(1.+z), 'matter')

    def MeanBaryonDensity(self, z):
        return (self.omega_b_0 / self.omega_m_0) * self.MeanMatterDensity(z)

    def MeanHydrogenNumberDensity(self, z):
        return (1. - self.Y) * self.MeanBaryonDensity(z) / m_H

    def MeanHeliumNumberDensity(self, z):
        return self.Y * self.MeanBaryonDensity(z) / m_He

    def MeanBaryonNumberDensity(self, z):
        return self.MeanBaryonDensity(z) / (m_H * self.MeanHydrogenNumberDensity(z) +
            4. * m_H * self.y * self.MeanHeliumNumberDensity(z))

    def ComovingRadialDistance(self, z0, z):
        """
        Return comoving distance between redshift z0 and z, z0 < z.
        """

        d0 = pyccl.comoving_radial_distance(self._ccl_instance, 1./(1.+z)) \
            * cm_per_mpc

        if z0 == 0:
            return d0

        d1 = pyccl.comoving_radial_distance(self._ccl_instance, 1./(1.+z0)) \
            * cm_per_mpc

        return d0 - d1

    def ProperRadialDistance(self, z0, z):
        return self.ComovingRadialDistance(z0, z) / (1. + z0)

    def dldz(self, z):
        """ Proper differential line element. """
        return self.ProperLineElement(z)

    def LuminosityDistance(self, z):
        return pyccl.luminosity_distance(self._ccl_instance, 1./(1.+z))
