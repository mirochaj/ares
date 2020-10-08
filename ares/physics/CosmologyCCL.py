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
    """
    Create a class instance that looks like a CosmologyARES instance but is
    calling CCL under the hood.
    """

    @property
    def _ccl_instance(self):
        if not hasattr(self, '_ccl_instance_'):
            self._ccl_instance_ = pyccl.Cosmology(Omega_c=self.omega_cdm_0,
                Omega_b=self.omega_b_0, h=self.h70, n_s=self.primordial_index,
                sigma8=self.sigma_8,
                transfer_function='boltzmann_camb')

            #'hmf_dlna': 2e-6,           # hmf default value is 1e-2
            #'hmf_dlnk': 1e-2,
            #'hmf_lnk_min': -20.,
            #'hmf_lnk_max': 10.,
            #'hmf_transfer_k_per_logint': 11,
            #'hmf_transfer_kmax': 100.,

            self._ccl_instance_.cosmo.gsl_params.INTEGRATION_EPSREL = 1e-8
            self._ccl_instance_.cosmo.gsl_params.INTEGRATION_DISTANCE_EPSREL = 1e-5
            self._ccl_instance_.cosmo.gsl_params.INTEGRATION_SIGMAR_EPSREL = 1e-12
            self._ccl_instance_.cosmo.gsl_params.ODE_GROWTH_EPSREL = 1e-8
            self._ccl_instance_.cosmo.gsl_params.EPS_SCALEFAC_GROWTH = 1e-8

            # User responsible for making sure NM and DELTA are consistent.
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_MIN = 4
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_MAX = 18
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_NM = 1400
            self._ccl_instance.cosmo.spline_params.LOGM_SPLINE_DELTA = 0.01

            self._ccl_instance_.cosmo.spline_params.K_MIN = 1e-9
            self._ccl_instance_.cosmo.spline_params.K_MAX = 10000
            self._ccl_instance_.cosmo.spline_params.K_MAX_SPLINE = 10000
            self._ccl_instance_.cosmo.spline_params.N_K = 1000

            self._ccl_instance.cosmo.spline_params.A_SPLINE_NA = 500
            #self._ccl_instance.cosmo.spline_params.A_SPLINE_MIN_PK = 0.01



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
