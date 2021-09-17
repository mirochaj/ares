"""

Cosmology.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on 2010-03-01.

Description:

"""
import os
import numpy as np
from ..data import ARES
from scipy.misc import derivative
from scipy.optimize import fsolve
from scipy.integrate import quad, ode
from ..util.Math import interp1d
from ..util.ParameterFile import ParameterFile
from .InitialConditions import InitialConditions
from .Constants import c, G, km_per_mpc, m_H, m_He, sigma_SB, g_per_msun, \
    cm_per_mpc, cm_per_kpc, k_B, m_p

_ares_to_planck = \
{
 'omega_m_0': 'omegam*',
 'omega_b_0': 'omegabh2',
 'hubble_0': 'H_0',
 'omega_l_0': 'omegal*',
 'sigma_8': 'sigma8',
 'primordial_index': 'ns',
}

class Cosmology(InitialConditions):
    def __init__(self, pf=None, **kwargs):
        if pf is not None:
            self.pf = pf
        else:
            self.pf = ParameterFile(**kwargs)

        # Load "raw" cosmological parameters
        ########################################################################
        if self.pf['cosmology_name'] != 'user':
            self._load_cosmology()
        else:
            self.omega_m_0 = self.pf['omega_m_0']
            self.omega_b_0 = self.pf['omega_b_0']
            self.hubble_0 = self.pf['hubble_0'] * 100. / km_per_mpc
            self.omega_l_0 = self.pf['omega_l_0']
            self.sigma_8 = self.sigma8 = self.pf['sigma_8']
            self.omega_cdm_0 = self.omega_m_0 - self.omega_b_0
            self.h70 = self.pf['hubble_0']
            self.helium_by_mass = self.Y = self.pf['helium_by_mass']

        ####################################################################

        # Everything beyond this point is a derived quantity of some sort.
        self.cmb_temp_0 = self.pf['cmb_temp_0']
        self.approx_highz = self.pf['approx_highz']
        self.approx_lowz = False
        self.primordial_index = self.pf['primordial_index']

        self.CriticalDensityNow = self.rho_crit_0 = \
            (3. * self.hubble_0**2) / (8. * np.pi * G)

        self.mean_density0 = self.omega_m_0 * self.rho_crit_0 \
            * cm_per_mpc**3 / g_per_msun

        self.helium_by_number = self.y = 1. / (1. / self.Y - 1.) / 4.


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

        self.delta_c0 = 1.686
        self.TcmbNow = self.cmb_temp_0

        self.fbaryon = self.omega_b_0 / self.omega_m_0
        self.fcdm = self.omega_cdm_0 / self.omega_m_0

        self.fbar_over_fcdm = self.fbaryon / self.fcdm

        # Used in hmf
        #self.pars_for_hmf = {'omega_lambda':self.omega_l_0,
        # 'omega_b':self.omega_b_0,
        # 'omega_M':self.omega_m_0,
        # 'sigma_8':self.sigma8,
        # 'n': self.primordial_index}

    def nH(self, z):
        return self.nH0 * (1. + z)**3

    def nHe(self, z):
        return self.nHe0 * (1. + z)**3

    @property
    def path_ARES(self):
        if not hasattr(self, '_path_ARES'):
            self._path_ARES = ARES
        return self._path_ARES

    @property
    def path_Planck(self):
        if not hasattr(self, '_path_Planck'):
            name = self.pf['cosmology_name'].replace('planck_', '')
            self._path_Planck = self.path_ARES \
                 + '/input/planck/base/plikHM_{}'.format(name)
        return self._path_Planck

    def _load_cosmology(self):
        if self.pf['cosmology_name'].startswith('planck'):
            self._load_planck()
        else:
            raise NotImplemented('Only know how to read Planck cosmologies!')

    def _load_planck(self):

        name = self.pf['cosmology_name'].replace('planck_', '')
        path = self.path_Planck
        prefix = 'base_plikHM_{}'.format(name)

        if self.pf['cosmology_id'] == 'best':

            data = {}
            with open('{}/{}.minimum'.format(path, prefix), 'r') as f:
                for i, line in enumerate(f):
                    if i < 2:
                        continue

                    if not line.strip():
                        continue

                    if line.startswith(' -log(Like)'):
                        break

                    # Parse line
                    _line = line.split()

                    row = int(_line[0])
                    val = float(_line[1])
                    name = _line[2]
                    name_latex = str(_line[3:])

                    data[name] = val

            self._planck_raw = data

            self.h70 = data['H0'] / 100.
            self.omega_b_0 = data['omegabh2'] / self.h70**2
            self.omega_cdm_0 = data['omegach2'] / self.h70**2
            self.hubble_0 = data['H0'] / km_per_mpc
            self.omega_l_0 = data['omegal']
            self.omega_k_0 = data['omegak']
            self.omega_m_0 = 1. - self.omega_l_0 - self.omega_k_0

            self.sigma_8 = self.sigma8 = data['sigma8']
            self.helium_by_mass = self.Y = data['yhe']

        else:

            num = self.pf['cosmology_id']

            if type(num) not in [int, np.int32, np.int64]:
                if self.pf['verbose']:
                    print("# WARNING: Casting cosmology_id {} to int.".format(num))
                num = int(num)

            ##
            # Load chains as one long concatenated super-array
            data = []
            for filenum in range(1, 5):
                chain_fn = '{}/{}_{}.txt'.format(path, prefix, filenum)
                data.append(np.loadtxt(chain_fn, unpack=True))
            data = np.concatenate(data, axis=1)

            ##
            # Load parameter names
            pars = []
            for line in open('{}/{}.paramnames'.format(path, prefix)):
                if not line.strip():
                    continue

                chunks = line.split()

                pars.append(chunks[0].strip().replace('*', ''))

            pars_in = {}
            for line in open('{}/{}.inputparams'.format(path, prefix)):
                if not line.strip():
                    continue
                if not line.startswith('param['):
                    continue

                _pre, _post = line.split('=')
                pre = _pre.strip()
                post = _post.split()

                pars_in[pre.replace('param', '')[1:-1]] = \
                    np.array([float(elem) for elem in post])

            ##
            # Just need to map to right array element. Remember that first
            # two rows in Planck chains are (weights -loglike).
            self.h70 = data[pars.index('H0')+2,num] / 100.
            self.omega_b_0 = data[pars.index('omegabh2')+2,num] / self.h70**2
            self.omega_cdm_0 = data[pars.index('omegach2')+2,num] / self.h70**2
            self.hubble_0 = data[pars.index('H0')+2,num] / km_per_mpc
            self.omega_l_0 = data[pars.index('omegal')+2,num]

            # In another file
            if 'omegak' not in pars:
                self.omega_k_0 = pars_in['omegak'][0]
            else:
                self.omega_k_0 = data[pars.index('omegak')+2]

            self.omega_m_0 = 1. - self.omega_l_0 - self.omega_k_0

            self.sigma_8 = self.sigma8 = data[pars.index('sigma8')+2,num]

            if 'yhe' not in pars:
                self.helium_by_mass = self.Y = pars_in['yhe'][0]
            else:
                self.helium_by_mass = data[pars.index('yhe')+2,num]

            if self.pf['verbose']:
                s = "# Set cosmological parameters to values in {}th element of".format(num)
                s += " concatenated array made from the following files:"
                print(s)
                path_str = path.replace(self.path_ARES, '$ARES')
                print("# {}_{}_?.txt".format(path_str, prefix))

        return

    def get_prefix(self):
        name = self.pf['cosmology_name']

        if self.pf['cosmology_id'] is None:
            pass
        elif type(self.pf['cosmology_id']) == str:
            name += '_' + self.pf['cosmology_id']
        else:
            num = self.pf['cosmology_id']
            if type(num) not in [int, np.int32, np.int64]:
                if self.pf['verbose']:
                    print("# WARNING: Casting `cosmology_id` {} to int.".format(num))
                num = int(num)

            name += '_{}'.format(str(num).zfill(5))

        return name

    @property
    def inits(self):
        if not hasattr(self, '_inits'):
            self._inits = self.get_inits_rec()
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

    def _Tgas_CosmoRec(self, z):
        if not hasattr(self, '_Tgas_CosmoRec_'):
            self._Tgas_CosmoRec_ = interp1d(self.inits['z'], self.inits['Tk'],
                kind='cubic', bounds_error=False)

        return self._Tgas_CosmoRec_(z)

    def Tgas(self, z):
        """
        Gas kinetic temperature at z in the absence of heat sources.
        """

        if self.pf['approx_thermal_history'] == 'piecewise':

            if type(z) == np.ndarray:
                T = np.zeros_like(z)

                hiz = z >= self.zdec
                loz = z < self.zdec

                T[hiz] = self.TCMB(z[hiz])
                T[loz] = self.TCMB(self.zdec) * (1. + z[loz])**2 \
                    / (1. + self.zdec)**2
                return T

            if z >= self.zdec:
                return self.TCMB(z)
            else:
                return self.TCMB(self.zdec) * (1. + z)**2 / (1. + self.zdec)**2
        elif self.pf['approx_thermal_history']:
            return np.interp(z, self.thermal_history['z'],
                        self.thermal_history['Tk']) * 1.
            #if not hasattr(self, '_Tgas'):
            #    self._Tgas = interp1d(self.thermal_history['z'],
            #        self.thermal_history['Tk'], kind='cubic',
            #        bounds_error=False)
            #
            #return self._Tgas(z)

        elif not self.pf['approx_thermal_history']:
            if not hasattr(self, '_Tgas'):
                self._Tgas = interp1d(self.inits['z'], self.inits['Tk'],
                    kind='cubic', bounds_error=False)

            # Make sure this is a float
            return self._Tgas(z) * 1.

    @property
    def thermal_history(self):
        if not hasattr(self, '_thermal_history'):

            if not self.pf['approx_thermal_history']:
                self._thermal_history = self.inits
                return self._thermal_history

            z0 = self.inits['z'][-2]
            solver = ode(self.cooling_rate)
            solver.set_integrator('vode', method='bdf')
            solver.set_initial_value([np.interp(z0, self.inits['z'],
                self.inits['Tk'])], z0)

            dz = self.pf['inits_Tk_dz']
            zf = final_redshift = 2.
            zall = []; Tall = []
            while solver.successful() and solver.t > zf:

                #print(solver.t, solver.y[0])

                if solver.t-dz < 0:
                    break

                zall.append(solver.t)
                Tall.append(solver.y[0])
                solver.integrate(solver.t-dz)

            self._thermal_history = {}
            self._thermal_history['z'] = np.array(zall)[-1::-1]
            self._thermal_history['Tk'] = np.array(Tall)[-1::-1]
            self._thermal_history['xe'] = 2e-4 * np.ones_like(zall)

        return self._thermal_history

    @property
    def cooling_pars(self):
        if not hasattr(self, '_cooling_pars'):
            self._cooling_pars = [self.pf['inits_Tk_p{}'.format(i)] for i in range(6)]
        return self._cooling_pars

    def cooling_rate(self, z, T=None):
        """
        This is dTk/dz.
        """
        if self.pf['approx_thermal_history'] in ['exp', 'tanh', 'exp+gauss', 'exp+pl']:

            # This shouldn't happen! Argh.
            if type(z) is np.ndarray:
                assert np.all(np.isfinite(z))
            else:
                if z < 0:
                    return np.nan

            if T is None:
                T = self.Tgas(z)

            t = self.t_of_z(z)
            dtdz = self.dtdz(z)
            return (T / t) * self.log_cooling_rate(z) * -1. * dtdz
        elif self.pf['approx_thermal_history'] in ['propto_xe']:
            #raise NotImplemented('help')

            if type(z) is np.ndarray:
                assert np.all(np.isfinite(z))
            else:
                if z < 0:
                    return np.nan

            # Start from CosmoRec

            ##
            # Need to do this self-consistently?
            ##s
            #func = lambda zz: np.interp(zz, self.inits['z'], self.inits['Tk'])

            dTdz = derivative(self._Tgas_CosmoRec, z, dx=1e-2)

            xe = np.interp(z, self.inits['z'], self.inits['xe'])

            #raise ValueError('help')
            pars = self.cooling_pars
            xe_cool = np.maximum(1. - xe, 0.0)
            mult = pars[0] * ((1. + z) / pars[1])**pars[2]

            #print(z, T, self.dtdz(z), self.HubbleParameter(z))
            dtdz = self.dtdz(z)

            if T is None:
                T = self.Tgas(z)

            hubble = 2. * self.HubbleParameter(z) * T * dtdz

            print(z, dTdz, xe_cool * mult / dTdz)
            return dTdz + xe_cool * mult

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
        elif self.pf['approx_thermal_history'] == 'exp+pl':
            pars = self.cooling_pars
            norm = -(2. + pars[2]) # Must be set so high-z limit -> -2/3
            exp = norm * (1. - np.exp(-(z / pars[0])**pars[1])) / 3. \
                + pars[2] / 3.
            pl = pars[4] * ((1. + z) / pars[0])**pars[5]

            if type(total) is np.ndarray:
                total[z >= 1100] = -2./3.
            elif z >= 1100:
                total = -2. / 3.

            # FIX ME
            #raise ValueError('help')
            xe_cool = np.maximum(1. - xe, 0.0) * pars[0] * ((1. + z) / pars[1])**pars[2]

            total = exp + pl

            return total
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
        Returns luminosity distance in cm.  Assumes we mean distance from
        us (z = 0).
        """

        integr = quad(lambda z: self.hubble_0 / self.HubbleParameter(z),
            0.0, z)[0]

        return integr * c * (1. + z) / self.hubble_0

    def DifferentialRedshiftElement(self, z, dl):
        """
        Given a redshift and a LOS distance, return the corresponding dz.

        Parameters
        ----------
        z0 : int, float
            Redshift.
        dl : int, float
            Distance in Mpc.
        """

        if not self.approx_highz:
            raise NotImplemented('sorry!')

        dz = ((1. + z)**-0.5 \
           - dl * cm_per_mpc * self.hubble_0 * np.sqrt(self.omega_m_0) / 2. / c)**-2 \
           - (1. + z)


        return dz

    def DeltaZed(self, z0, dR):
        f = lambda z2: self.ComovingRadialDistance(z0, z2) / cm_per_mpc - dR
        return fsolve(f, x0=z0+0.1)[0] - z0

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

    def ProjectedVolume(self, z, angle, dz=1.):
        """
        Compute the co-moving volume of a spherical shell defined by `angle`
        and thickness `dz`.

        Parameters
        ----------
        z : int, float
            Redshift of shell center.
        area : int, float
            Angular scale in degrees.
        dz : int, float
            Shell thickness in differential redshift element.

        Returns
        -------
        Volume in comoving Mpc^3.

        """

        dA = self.AngleToComovingLength(z, angle * 60.) * cm_per_mpc
        dldz = quad(self.ComovingLineElement, z-0.5*dz, z+0.5*dz)[0]

        return dA**2 * dldz / cm_per_mpc**3

    def JeansMass(self, z, Tgas=None, mu=0.6):

        if Tgas is None:
            Tgas = self.Tgas(z)

        k_J = (2. * k_B * Tgas / 3. / mu / m_p)**-0.5 \
            * np.sqrt(self.OmegaMatter(z)) * self.hubble_0

        l_J = 2. * np.pi / k_J

        return 4. * np.pi * (l_J / 2)**3 * self.rho_b_z0 / 3. / g_per_msun

    def ComovingLengthToAngle(self, z, R):
        """
        Convert a length scale (co-moving) to an observed angle [arcmin].
        """
        f = lambda ang: self.AngleToComovingLength(z, ang) - R
        return fsolve(f, x0=0.1)[0]

    def AngleToComovingLength(self, z, angle):
        return self.AngleToProperLength(z, angle) * (1. + z)

    def AngleToProperLength(self, z, angle):
        """
        Convert an angle to a co-moving length-scale at the observed redshift.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        angle : int, float
            Angle in arcminutes.

        Returns
        -------
        Length scale in Mpc.

        """

        d = self.LuminosityDistance(z) / (1. + z)**2 # cm

        in_rad = (angle / 60.) * np.pi / 180.

        x = np.tan(in_rad) * d / cm_per_mpc

        return x
