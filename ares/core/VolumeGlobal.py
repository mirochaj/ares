"""

IntergalacticMedium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri May 24 11:31:06 2013

Description:

"""

import numpy as np
from ..util.Warnings import *
from ..util import ProgressBar
from ..physics.Constants import *
import types, os, re, sys
from ..physics import SecondaryElectrons
from scipy.integrate import dblquad, romb, simps, quad, trapz

try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

log10 = np.log(10.)
E_th = np.array([E_LL, 24.6, 54.4])

defkwargs = \
{
 'zf':None,
 'xray_flux':None,
 'epsilon_X': None,
 'Gamma': None,
 'gamma': None,
 'return_rc': False,
 'energy_units':False,
 'Emax': None,
 #'zxavg':0.0,
 #'igm':True,
 'xavg': 0.0,
 'igm_h_1': 1.0,
 'igm_h_2': 0.0,
 'igm_he_1': 1.0,
 'igm_he_2': 0.0,
 'igm_he_3': 0.0,
 'cgm_h_1': 1.0,
 'cgm_h_2': 0.0,
 'cgm_he_2': 0.0,
 'cgm_he_3': 0.0,
 'igm_e': 0.0,
}

species_i_to_str = {0:'h_1', 1:'he_1', 2:'he_2'}

class GlobalVolume(object):
    def __init__(self, background):
        """
        Initialize a GlobalVolume.

        Parameters
        ----------
        background : ares.solvers.UniformBackground instance.

        """

        self.background = background
        self.pf = background.pf
        self.grid = background.grid
        self.cosm = background.cosm
        self.hydr = background.hydr
        self.pops = background.pops
        self.Npops = len(self.pops)

        # Include helium opacities approximately?
        self.approx_He = self.pf['include_He'] and self.pf['approx_He']

        # Include helium opacities self-consistently?
        self.self_consistent_He = self.pf['include_He'] \
            and (not self.pf['approx_He'])

        self.esec = \
            SecondaryElectrons(method=self.pf["secondary_ionization"])

        # Choose function for computing bound-free absorption cross-sections
        if self.pf['approx_sigma']:
            from ..physics.CrossSections import \
                ApproximatePhotoIonizationCrossSection as sigma
        else:
            from ..physics.CrossSections import \
                PhotoIonizationCrossSection as sigma

        self.sigma = sigma
        self.sigma0 = sigma(E_th[0])    # Hydrogen ionization threshold

        self._set_integrator()

    @property
    def rates_no_RT(self):
        if not hasattr(self, '_rates_no_RT'):
            self._rates_no_RT = \
                {'k_ion': np.zeros((self.grid.dims,
                    self.grid.N_absorbers)),
                 'k_heat': np.zeros((self.grid.dims,
                    self.grid.N_absorbers)),
                 'k_ion2': np.zeros((self.grid.dims,
                    self.grid.N_absorbers, self.grid.N_absorbers)),
                }

        return self._rates_no_RT

    @property
    def E(self):
        if not hasattr(self, '_E'):
            self._tabulate_atomic_data()

        return self._E

    @property
    def sigma_E(self):
        if not hasattr(self, '_sigma_E'):
            self._tabulate_atomic_data()

        return self._sigma_E

    def _tabulate_atomic_data(self):
        """
        Pre-compute cross sections and such for each source population.

        Returns
        -------
        Nothing. Sets the following attributes:

        sigma_E
        log_sigma_E
        fheat, flya, fion

        """

        # Remember: these will all be [Npops, Nbands/pop, Nenergies/band]
        self._E = self.background.energies
        self.logE = [[] for k in range(self.Npops)]
        self.dlogE = [[] for k in range(self.Npops)]
        self.fheat = [[] for k in range(self.Npops)]
        self.flya = [[] for k in range(self.Npops)]
        self.fexc = [[] for k in range(self.Npops)]

        # These are species dependent
        self._sigma_E = {}
        self.fion = {}
        for species in ['h_1', 'he_1', 'he_2']:
            self._sigma_E[species] = [[] for k in range(self.Npops)]
            self.fion[species] = [[] for k in range(self.Npops)]

        ##
        # Note: If secondary_ionization > 1, there will be an ionized fraction
        # dimension in fion and fheat.
        ##

        # Loop over populations
        for i, pop in enumerate(self.pops):

            # This means the population is completely approximate
            if not np.any(self.background.solve_rte[i]):
                self.logE[i] = [None]
                self.dlogE[i] = [None]
                self.fheat[i] = [None]
                self.flya[i] = [None]
                self.fexc[i] = [None]

                for species in ['h_1', 'he_1', 'he_2']:
                    self.fion[species][i] = [None]
                    self._sigma_E[species][i] = [None]

                continue

            ##
            # If we make it here, the population has at least one band that
            # requires a detailed solution to the RTE
            ##

            Nbands = len(self.background.energies[i])

            self.logE[i] = [None for k in range(Nbands)]
            self.dlogE[i] = [None for k in range(Nbands)]
            self.fheat[i] = [None for k in range(Nbands)]
            self.flya[i] = [None for k in range(Nbands)]
            self.fexc[i] = [None for k in range(Nbands)]
            for species in ['h_1', 'he_1', 'he_2']:
                self.fion[species][i] = [None for k in range(Nbands)]
                self._sigma_E[species][i] = [None for k in range(Nbands)]

            # Loop over each band for this population
            for j, band in enumerate(self.background.bands_by_pop[i]):

                if band is None:
                    continue

                need_tab = self.pops[i].is_src_xray \
                    and np.any(np.array(band) > E_LL) \
                    and self.background.solve_rte[i][j]

                if (not self.background.solve_rte[i][j]) or \
                   (not need_tab):
                    continue
                else:
                    self.fheat[i][j] = \
                        [np.ones([self.background.energies[i][j].size,
                         len(self.esec.x)]) \
                         for j in range(Nbands)]
                    self.flya[i] = \
                        [np.ones([self.background.energies[i][j].size,
                         len(self.esec.x)]) \
                         for j in range(Nbands)]
                    self.fexc[i] = \
                        [np.ones([self.background.energies[i][j].size,
                         len(self.esec.x)]) \
                         for j in range(Nbands)]

                    for species in ['h_1', 'he_1', 'he_2']:
                        if self.esec.method > 1:
                            self._sigma_E[species][i] = \
                                [np.ones([self.background.energies[i][j].size,
                                 len(self.esec.x)]) \
                                 for j in range(Nbands)]
                            self.fion[species][i] = \
                                [np.ones([self.background.energies[i][j].size,
                                 len(self.esec.x)]) \
                                 for j in range(Nbands)]

                        else:
                            self._sigma_E[species][i] = [None for k in range(Nbands)]
                            self.fion[species][i] = [None for k in range(Nbands)]
                            self.fheat[i] = [None for k in range(Nbands)]
                            self.flya[i] = [None for k in range(Nbands)]
                            self.fexc[i] = [None for k in range(Nbands)]

                # More convenient variables
                E = self._E[i][j]
                N = E.size

                # Compute some things we need, like bound-free cross-section
                self.logE[i][j] = np.log10(E)
                self.dlogE[i][j] = np.diff(self.logE[i][j])

                #
                for k, species in enumerate(['h_1', 'he_1', 'he_2']):
                    self._sigma_E[species][i][j] = \
                        np.array([self.sigma(Eval, k) for Eval in E])

                # Pre-compute secondary ionization and heating factors
                if self.esec.method > 1:

                    # Don't worry: we'll fill these in in a sec!
                    self.fheat[i][j] = np.ones([N, len(self.esec.x)])
                    self.flya[i][j] = np.ones([N, len(self.esec.x)])
                    self.fexc[i][j] = np.ones([N, len(self.esec.x)])

                    # Must evaluate at ELECTRON energy, not photon energy
                    for k, nrg in enumerate(E - E_th[0]):
                        self.fheat[i][j][k] = \
                            self.esec.DepositionFraction(self.esec.x, E=nrg,
                            channel='heat')
                        self.fion['h_1'][i][j][k] = \
                            self.esec.DepositionFraction(self.esec.x, E=nrg,
                            channel='h_1')

                        if self.pf['secondary_lya']:
                            self.flya[i][j][k] = \
                                self.esec.DepositionFraction(self.esec.x, E=nrg,
                                channel='lya')
                            self.fexc[i][j][k] = \
                                self.esec.DepositionFraction(self.esec.x, E=nrg,
                                channel='exc')

                    # Helium
                    if self.pf['include_He'] and not self.pf['approx_He']:

                        # Don't worry: we'll fill these in in a sec!
                        self.fion['he_1'][i][j] = np.ones([N, len(self.esec.x)])
                        self.fion['he_2'][i][j] = np.ones([N, len(self.esec.x)])

                        for k, nrg in enumerate(E - E_th[1]):
                            self.fion['he_1'][i][j][k] = \
                                self.esec.DepositionFraction(self.esec.x,
                                E=nrg, channel='he_1')

                        for k, nrg in enumerate(E - E_th[2]):
                            self.fion['he_2'][i][j][k] = \
                                self.esec.DepositionFraction(self.esec.x,
                                E=nrg, channel='he_2')

                    else:
                        self.fion['he_1'][i][j] = np.zeros([N, len(self.esec.x)])
                        self.fion['he_2'][i][j] = np.zeros([N, len(self.esec.x)])

        return

    def _set_integrator(self):
        self.integrator = self.pf["unsampled_integrator"]
        self.sampled_integrator = self.pf["sampled_integrator"]
        self.rtol = self.pf["integrator_rtol"]
        self.atol = self.pf["integrator_atol"]
        self.divmax = int(self.pf["integrator_divmax"])

    def RestFrameEnergy(self, z, E, zp):
        """
        Return energy of a photon observed at (z, E) and emitted at zp.
        """

        return E * (1. + zp) / (1. + z)

    def ObserverFrameEnergy(self, z, Ep, zp):
        """
        What is the energy of a photon observed at redshift z and emitted
        at redshift zp and energy Ep?
        """

        return Ep * (1. + z) / (1. + zp)

    def Jc(self, z, E):
        """
        Flux corresponding to one photon per hydrogen atom at redshift z.
        """

        return c * self.cosm.nH0 * (1. + z)**3 / 4. / np.pi \
            / (E * erg_per_ev / h)

    def rate_to_coefficient(self, z, species=0, zone='igm', **kw):
        """
        Convert an ionization/heating rate to a rate coefficient.

        Provides units of per atom.
        """

        if self.pf['photon_counting']:
            prefix = zone
        else:
            prefix = 'igm'

        if species == 0:
            weight = 1. / self.cosm.nH(z)  / kw['{!s}_h_1'.format(prefix)]
        elif species == 1:
            weight = 1. / self.cosm.nHe(z) / kw['{!s}_he_1'.format(prefix)]
        elif species == 2:
            weight = 1. / self.cosm.nHe(z) / kw['{!s}_he_2'.format(prefix)]

        return weight

    def coefficient_to_rate(self, z, species=0, **kw):
        return 1. / self.rate_to_coefficient(z, species, **kw)

    def _fix_kwargs(self, functionify=False, popid=0, band=0, **kwargs):

        kw = defkwargs.copy()
        kw.update(kwargs)

        pop = self.pops[popid]

        if functionify and type(kw['xavg']) is not types.FunctionType:
            tmp = kw['xavg']
            kw['xavg'] = lambda z: tmp

        if kw['zf'] is None and pop is not None:
            kw['zf'] = pop.zform

        if not self.background.solve_rte[popid][band]:
            pass
        elif (kw['Emax'] is None) and self.background.solve_rte[popid][band] and \
            np.any(np.array(self.background.bands_by_pop[popid]) > pop.pf['pop_Emin_xray']):

            kw['Emax'] = self.background.energies[popid][band][-1]

        return kw

    def HeatingRate(self, z, species=0, popid=0, band=0, **kwargs):
        """
        Compute heating rate density due to emission from this population.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        species : int
            Atom whose liberated electrons cause heating.
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)

        ===============
        relevant kwargs
        ===============
        fluxes : np.ndarray
            Array of fluxes corresponding to photon energies in self.igm.E.
        return_rc : bool
            Return actual heating rate, or rate coefficient for heating?
            Former has units of erg s**-1 cm**-3, latter has units of
            erg s**-1 cm**-3 atom**-1.

        Returns
        -------
        Proper heating rate density in units of in erg s**-1 cm**-3 at redshift z,
        due to electrons previously bound to input species.

        """

        pop = self.pops[popid]

        if (not pop.is_src_heat_igm) or (z >= pop.zform):
            return 0.0

        if pop.pf['pop_heat_rate'] is not None:
            return pop.HeatingRate(z)

        # Grab defaults, do some patches if need be
        kw = self._fix_kwargs(popid=popid, **kwargs)

        species_str = species_i_to_str[species]

        if pop.pf['pop_heat_rate'] is not None:
            return pop.pf['pop_heat_rate'](z)

        if band is not None:
            solve_rte = self.background.solve_rte[popid][band]
        else:
            solve_rte = False

        # Compute fraction of photo-electron energy deposited as heat
        if pop.pf['pop_fXh'] is None:

            # Interpolate in energy and ionized fraction
            if (self.esec.method > 1) and solve_rte:
                if kw['igm_e'] <= self.esec.x[0]:
                    fheat = self.fheat[popid][band][:,0]
                else:
                    i_x = np.argmin(np.abs(kw['igm_e'] - self.esec.x))
                    if self.esec.x[i_x] > kw['igm_e']:
                        i_x -= 1

                    j = i_x + 1

                    fheat = self.fheat[popid][band][:,i_x] \
                        + (self.fheat[popid][band][:,j] - self.fheat[popid][band][:,i_x]) \
                        * (kw['igm_e'] - self.esec.x[i_x]) \
                        / (self.esec.x[j] - self.esec.x[i_x])
            elif self.esec.method > 1:
                print("popid={}".format(popid))
                raise ValueError('Only know how to do advanced secondary ionization with solve_rte=True')
            else:
                fheat = self.esec.DepositionFraction(kw['igm_e'])[0]

        else:
            fheat = pop.pf['pop_fXh']

        # Assume heating rate density at redshift z is only due to emission
        # from sources at redshift z
        if not solve_rte:
            weight = self.rate_to_coefficient(z, species, **kw)

            Lx = pop.LuminosityDensity(z, Emin=pop.pf['pop_Emin_xray'],
                Emax=pop.pf['pop_Emax'])

            return weight * fheat * Lx * (1. + z)**3

        ##
        # Otherwise, do the full calculation
        ##

        # Re-normalize to help integrator
        norm = J21_num * self.sigma0

        # Computes excess photo-electron energy due to ionizations by
        # photons with energy E (normalized by sigma0 * Jhat)
        if kw['fluxes'][popid] is None:

            # If we're approximating helium, must add contributions now
            # since we'll never explicitly call this method w/ species=1.
            if self.approx_He:
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg']) \
                    * (self.sigma(E) * (E - E_th[0]) \
                    + self.cosm.y * self.sigma(E, species=1) * (E - E_th[1])) \
                    * fheat / norm / ev_per_hz

            # Otherwise, just heating via hydrogen photo-electrons
            else:
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'],
                    zxavg=kw['zxavg']) * self.sigma(E, species=1) \
                    * (E - E_th[species]) * fheat / norm / ev_per_hz

        # This means the fluxes have been computed already - integrate
        # over discrete set of points
        else:

            integrand = self.sigma_E[species_str][popid][band] \
                * (self._E[popid][band] - E_th[species])

            if self.approx_He:
                integrand += self.cosm.y * self.sigma_E['he_1'][popid][band] \
                    * (self._E[popid][band] - E_th[1])

            integrand *= kw['fluxes'][popid][band] * fheat / norm / ev_per_hz

        # Compute integral over energy
        if type(integrand) == types.FunctionType:
            heat, err = dblquad(integrand, z, kw['zf'], lambda a: self.E0,
                lambda b: kw['Emax'], epsrel=self.rtol, epsabs=self.atol)
        else:
            if kw['Emax'] is not None:
                imax = np.argmin(np.abs(self._E[popid][band] - kw['Emax']))
                if imax == 0:
                    return 0.0
                elif imax == (len(self._E[popid][band]) - 1):
                    imax = None

                if self.sampled_integrator == 'romb':
                    raise ValueError("Romberg's method cannot be used for integrating subintervals.")
                    heat = romb(integrand[0:imax] * self.E[0:imax],
                        dx=self.dlogE[0:imax])[0] * log10
                else:
                    heat = simps(integrand[0:imax] * self._E[popid][band][0:imax],
                        x=self.logE[popid][band][0:imax]) * log10

            else:
                imin = np.argmin(np.abs(self._E[popid][band] - pop.pf['pop_Emin']))

                if self.sampled_integrator == 'romb':
                    heat = romb(integrand[imin:] * self._E[popid][band][imin:],
                        dx=self.dlogE[popid][band][imin:])[0] * log10
                elif self.sampled_integrator == 'trapz':
                    heat = np.trapz(integrand[imin:] * self._E[popid][band][imin:],
                        x=self.logE[popid][band][imin:]) * log10
                else:
                    heat = simps(integrand[imin:] * self._E[popid][band][imin:],
                        x=self.logE[popid][band][imin:]) * log10

        # Re-normalize, get rid of per steradian units
        heat *= 4. * np.pi * norm * erg_per_ev

        # Currently a rate coefficient, returned value depends on return_rc
        if kw['return_rc']:
            pass
        else:
            heat *= self.coefficient_to_rate(z, species, **kw)

        return heat

    def IonizationRateCGM(self, z, species=0, popid=0, band=0, **kwargs):
        """
        Compute growth rate of HII regions.

        Parameters
        ----------
        z : float
            current redshift
        species : int
            Ionization rate for what atom?
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)

        ===============
        relevant kwargs
        ===============
        fluxes : np.ndarray
            Array of fluxes corresponding to photon energies in self.igm.E.
        return_rc : bool
            Return actual heating rate, or rate coefficient for heating?
            Former has units of erg s**-1 cm**-3, latter has units of
            erg s**-1 cm**-3 atom**-1.

        Returns
        -------
        Ionization rate. Units determined by value of return_rc keyword
        argument, which is False by default.

        """

        pop = self.pops[popid]

        if band is not None:
            b = self.background.bands_by_pop[popid][band]
            if not np.any(np.array(b) > E_LL):
                return 0.0
            if not np.allclose(b[0], E_LL, atol=0.1, rtol=0):
                return 0.0
        else:
            b = [E_LL, 24.6]

        if (not pop.is_src_ion_cgm) or (z > pop.zform):
            return 0.0

        # Need some guidance from 1-D calculations to do this
        if species > 0:
            return 0.0

        if pop.pf['pop_ion_rate_cgm'] is not None:
            return pop.IonizationRateCGM(z)

        kw = defkwargs.copy()
        kw.update(kwargs)

        #if pop.pf['pop_ion_rate_cgm'] is not None:
        #    return self.pf['pop_k_ion_cgm'](z)

        if kw['return_rc']:
            weight = self.rate_to_coefficient(z, species, **kw)
        else:
            weight = 1.0

        Qdot = pop.PhotonLuminosityDensity(z, Emin=E_LL, Emax=24.6)

        return weight * Qdot * (1. + z)**3

    def IonizationRateIGM(self, z, species=0, popid=0, band=0, **kwargs):
        """
        Compute volume averaged hydrogen ionization rate.

        Parameters
        ----------
        z : float
            redshift
        species : int
            HI, HeI, or HeII (species=0, 1, 2, respectively)

        Returns
        -------
        Volume averaged ionization rate in units of ionizations per
        second. If return_rc=True, will be in units of ionizations per
        second per atom.

        """

        pop = self.pops[popid]

        # z between zform, zdead? must be careful for BHs
        if (not pop.is_src_ion_igm) or (z > pop.zform):
            return 0.0

        # Grab defaults, do some patches if need be
        kw = self._fix_kwargs(**kwargs)

        species_str = species_i_to_str[species]

        if pop.pf['pop_ion_rate_igm'] is not None:
            return pop.IonizationRateIGM(z)

        if band is not None:
            solve_rte = self.background.solve_rte[popid][band]
        else:
            solve_rte = False

        if (not solve_rte) or \
            (not np.any(np.array(self.background.bands_by_pop[popid]) > pop.pf['pop_Emin_xray'])):

            Lx = pop.LuminosityDensity(z, Emin=pop.pf['pop_Emin_xray'],
                Emax=pop.pf['pop_Emax'])

            weight = self.rate_to_coefficient(z, species, **kw)
            primary = weight * Lx \
                * (1. + z)**3 / pop.pf['pop_Ex'] / erg_per_ev
            fion = self.esec.DepositionFraction(kw['igm_e'], channel='h_1')[0]

            return primary * (1. + fion) * (pop.pf['pop_Ex'] - E_th[0]) \
                / E_th[0]

        # Full calculation - much like computing integrated flux
        norm = J21_num * self.sigma0

        # Integrate over function
        if kw['fluxes'][popid] is None:
            integrand = lambda E, zz: \
                self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'],
                zxavg=kw['zxavg']) * self.sigma(E, species=species) \
                / norm / ev_per_hz

            ion, err = dblquad(integrand, z, kw['zf'], lambda a: self.E0,
                lambda b: kw['Emax'], epsrel=self.rtol, epsabs=self.atol)

        # Integrate over set of discrete points
        else:
            integrand = self.sigma_E[species_str][popid][band] \
                * kw['fluxes'][popid][band] / norm / ev_per_hz

            if self.sampled_integrator == 'romb':
                ion = romb(integrand * self.E[popid][band],
                    dx=self.dlogE[popid][band])[0] * log10
            else:
                ion = simps(integrand * self.E[popid][band],
                    x=self.logE[popid][band]) * log10

        # Re-normalize
        ion *= 4. * np.pi * norm

        # Currently a rate coefficient, returned value depends on return_rc
        if kw['return_rc']:
            pass
        else:
            ion *= self.coefficient_to_rate(z, species, **kw)

        return ion

    def SecondaryIonizationRateIGM(self, z, species=0, donor=0, popid=0,
        band=0, **kwargs):
        """
        Compute volume averaged secondary ionization rate.

        Parameters
        ----------
        z : float
            redshift
        species : int
            Ionization rate of what atom?
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)
        donor : int
            Which atom gave the electron?
            Can be 0, 1, or 2 (HI, HeI, and HeII, respectively)

        ===============
        relevant kwargs
        ===============
        fluxes : np.ndarray
            Array of fluxes corresponding to photon energies in self.igm.E.
        return_rc : bool
            Return actual heating rate, or rate coefficient for heating?
            Former has units of erg s**-1 cm**-3, latter has units of
            erg s**-1 cm**-3 atom**-1.

        Returns
        -------
        Volume averaged ionization rate due to secondary electrons,
        in units of ionizations per second.

        """

        pop = self.pops[popid]

        if self.pf['secondary_ionization'] == 0:
            return 0.0

        if not pop.pf['pop_ion_src_igm']:
            return 0.0

        if band is not None:
            solve_rte = self.background.solve_rte[popid][band]
        else:
            solve_rte = False

        # Computed in IonizationRateIGM in this case
        if not solve_rte:
            return 0.0

        if not np.any(np.array(self.background.bands_by_pop[popid]) > pop.pf['pop_Emin_xray']):
            return 0.0

        if ((donor or species) in [1,2]) and (not self.pf['include_He']):
            return 0.0

        # Grab defaults, do some patches if need be
        kw = self._fix_kwargs(**kwargs)

        #if self.pf['gamma_igm'] is not None:
        #    return self.pf['gamma_igm'](z)

        species_str = species_i_to_str[species]
        donor_str = species_i_to_str[donor]

        if self.esec.method > 1 and solve_rte:

            fion_const = 1.
            if kw['igm_e'] == 0:
                fion = self.fion[species_str][popid][band][:,0]
            else:
                i_x = np.argmin(np.abs(kw['igm_e'] - self.esec.x))
                if self.esec.x[i_x] > kw['igm_e']:
                    i_x -= 1

                j = i_x + 1

                fion = self.fion[species_str][popid][band][:,i_x] \
                    + (self.fion[species_str][popid][band][:,j] - self.fion[species_str][popid][:,i_x]) \
                    * (kw['igm_e'] - self.esec.x[i_x]) \
                    / (self.esec.x[j] - self.esec.x[i_x])
        elif self.esec.method > 1:
            raise ValueError('Only know how to do advanced secondary ionization with solve_rte=True')
        else:
            fion = 1.0
            fion_const = self.esec.DepositionFraction(kw['igm_e'],
                channel=species_str)[0]

        norm = J21_num * self.sigma0

        if kw['fluxes'][popid] is None:
            if self.pf['approx_He']: # assumes lower integration limit > 4 Ryd
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'],
                    zxavg=kw['zxavg']) * (self.sigma(E) * (E - E_th[0]) \
                    + self.cosm.y * self.sigma(E, 1) * (E - E_th[1])) \
                    / E_th[0] / norm / ev_per_hz
            else:
                integrand = lambda E, zz: \
                    self.rb.AngleAveragedFluxSlice(z, E, zz, xavg=kw['xavg'],
                    zxavg=kw['zxavg']) * self.sigma(E) * (E - E_th[0]) \
                    / E_th[0] / norm / ev_per_hz
        else:
            integrand = fion * self.sigma_E[donor_str][popid][band] \
                * (self.E[popid][band] - E_th[donor])

            if self.pf['approx_He']:
                integrand += self.cosm.y * self.sigma_E['he_1'][popid][band] \
                    * (self.E[popid][band] - E_th[1])

            integrand *= kw['fluxes'][popid][band] / E_th[species] / norm \
                / ev_per_hz

        if type(integrand) == types.FunctionType:
            ion, err = dblquad(integrand, z, kw['zf'], lambda a: self.E0,
                lambda b: kw['Emax'], epsrel=self.rtol, epsabs=self.atol)
        else:
            if self.sampled_integrator == 'romb':
                ion = romb(integrand * self.E[popid][band],
                    dx=self.dlogE[popid][band])[0] * log10
            else:
                ion = simps(integrand * self.E[popid][band],
                    x=self.logE[popid][band]) * log10

        # Re-normalize
        ion *= 4. * np.pi * norm * fion_const

        # Currently a rate coefficient, returned value depends on return_rc
        if kw['return_rc']:
            pass
        else:
            ion *= self.coefficient_to_rate(z, species, **kw)

        return ion

    def SecondaryLymanAlphaFlux(self, z, species=0, popid=0, band=0,
        **kwargs):
        """
        Flux of Lyman-alpha photons induced by photo-electron collisions.

        Can only be sourced by X-ray populations.

        """

        pop = self.pops[popid]

        if not self.pf['secondary_lya']:
            return 0.0

        if not pop.is_src_ion_igm:
            return 0.0

        species_str = species_i_to_str[species]
        donor_str = species_i_to_str[donor]
        band = 0

        # Grab defaults, do some patches if need be
        kw = self._fix_kwargs(**kwargs)

        E = self.E

        # Compute fraction of photo-electron energy deposited as Lya excitation
        if self.esec.method > 1 and (kw['fluxes'][popid] is not None):

            ##
            # Recall that flya is measured relative to fexc
            ##


            if kw['igm_e'] == 0:
                flya = self.flya[popid][band][:,0] \
                     * self.fexc[popid][band][:,0]

            else:
                flya = 1.
                for tab in [self.fexc, self.flya]:

                    i_x = np.argmin(np.abs(kw['igm_e'] - self.esec.x))
                    if self.esec.x[i_x] > kw['igm_e']:
                        i_x -= 1

                    j = i_x + 1

                    f = tab[popid][band][:,i_x] \
                        + (tab[popid][band][:,j] - tab[popid][band][:,i_x]) \
                        * (kw['igm_e'] - self.esec.x[i_x]) \
                        / (self.esec.x[j] - self.esec.x[i_x])

                    flya *= f

        else:
            return 0.0

        norm = J21_num * self.sigma0

        integrand = self.sigma_E[species_str][popid][band] \
            * (self._E[popid][band] - E_th[species])

        if self.approx_He:
            integrand += self.cosm.y * self.sigma_E['he_1'][popid][band] \
                * (self._E[popid][band] - E_th[1])

        # Must get back to intensity units
        integrand *= kw['fluxes'][popid][band] * flya / norm / E_LyA / ev_per_hz

        if kw['Emax'] is not None:
            imax = np.argmin(np.abs(self._E[popid][band] - kw['Emax']))
            if imax == 0:
                return 0.0
            elif imax == (len(self._E[popid][band]) - 1):
                imax = None

            if self.sampled_integrator == 'romb':
                raise ValueError("Romberg's method cannot be used for integrating subintervals.")
                e_ax = romb(integrand[0:imax] * self.E[0:imax],
                    dx=self.dlogE[0:imax])[0] * log10
            else:
                e_ax = simps(integrand[0:imax] * self._E[popid][band][0:imax],
                    x=self.logE[popid][band][0:imax]) * log10
        else:
            imin = np.argmin(np.abs(self._E[popid][band] - pop.pf['pop_Emin']))

            if self.sampled_integrator == 'romb':
                e_ax = romb(integrand[imin:] * self._E[popid][band][imin:],
                    dx=self.dlogE[popid][band][imin:])[0] * log10
            elif self.sampled_integrator == 'trapz':
                e_ax = np.trapz(integrand[imin:] * self._E[popid][band][imin:],
                    x=self.logE[popid][band][imin:]) * log10
            else:
                e_ax = simps(integrand[imin:] * self._E[popid][band][imin:],
                    x=self.logE[popid][band][imin:]) * log10

        # Re-normalize. This is essentially a photon emissivity modulo 4 pi ster
        # This is a *proper* emissivity, BTW.
        e_ax *= norm

        # Just normalizing by electron donor species abundance
        e_ax *= self.coefficient_to_rate(z, species, **kw)

        # At this point, we've got a diffuse Ly-a emissivity.
        # Need to convert it to a flux. Assume infinitesimally narrow line
        # profile, i.e., emissivity translates instantaneously to flux only
        # at this redshift.

        # Convert to a co-moving emissivity [photons / s / cm^3]
        e_ax /= (1. + z)**3

        # We get a factor of nu_alpha from integrating over the line profile
        # (assuming it's a delta function).
        e_ax /= nu_alpha

        # Convert to a flux
        Ja = e_ax * (1 + z)**2 * c / self.cosm.HubbleParameter(z)

        return Ja
