"""

UniformBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:15:36 MDT 2014

Description:

"""

import numpy as np
from math import ceil
from ..data import ARES
import os, re, types, gc
from ..util import ParameterFile
from ..static import GlobalVolume
from ..util.Misc import num_freq_bins
from ..util.Math import interp1d
from .OpticalDepth import OpticalDepth
from ..util.Warnings import no_tau_table
from ..physics import Hydrogen, Cosmology
from ..populations.Composite import CompositePopulation
from ..populations.GalaxyAggregate import GalaxyAggregate
from scipy.integrate import quad, romberg, romb, trapz, simps
from ..physics.Constants import ev_per_hz, erg_per_ev, c, E_LyA, E_LL, dnu, h_p
#from ..util.ReadData import flatten_energies, flatten_flux, split_flux, \
#    flatten_emissivities
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

try:
    import h5py
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

log10 = np.log(10.)    # for when we integrate in log-space
four_pi = 4. * np.pi
c_over_four_pi = c / four_pi

# Put this stuff in utils
defkwargs = \
{
 'zf':None,
 'xray_flux':None,
 'xray_emissivity': None,
 'lw_flux':None,
 'lw_emissivity': None,
 'tau':None,
 'return_rc': False,
 'energy_units':False,
 'xavg': 0.0,
 'zxavg':0.0,
}

class UniformBackground(object):
    def __init__(self, pf=None, grid=None, **kwargs):
        """
        Initialize a UniformBackground object.

        Creates an object capable of evolving the radiation background created
        by some population of objects, which are characterized by a comoving
        volume emissivity and a spectrum. The evolution of the IGM opacity can
        be computed self-consistently or imposed artificially.

        Parameters
        ----------
        grid : instance
            ares.static.Grid instance

        """

        self._kwargs = kwargs.copy()

        if pf is None:
            self.pf = ParameterFile(**kwargs)
        else:
            self.pf = pf

        # Some useful physics modules
        if grid is not None:
            self.grid = grid
            self.cosm = grid.cosm
        else:
            self.grid = None
            self.cosm = Cosmology(pf=self.pf, **self.pf)

        self._set_integrator()

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)

        return self._hydr

    @property
    def volume(self):
        if not hasattr(self, '_volume'):
            self._volume = GlobalVolume(self)

        return self._volume

    @property
    def solve_rte(self):
        """
        By population and band, are we solving the RTE in detail?
        """

        if not hasattr(self, '_solve_rte'):

            self._solve_rte = []
            for i, pop in enumerate(self.pops):

                if self.bands_by_pop[i] is None:
                    self._solve_rte.append([False])
                    continue

                tmp = []
                for j, band in enumerate(self.bands_by_pop[i]):

                    if band is None:
                        # When does this happen?
                        tmp.append(False)
                        #if j == 0:
                        #    break
                    elif type(pop.pf['pop_solve_rte']) is bool:
                        tmp.append(pop.pf['pop_solve_rte'])
                       #if j == 0:
                       #    break
                    elif type(pop.pf['pop_solve_rte']) is tuple:

                        # Is this too restrictive? Prone to user error.
                        lo_close = band[0] >= pop.pf['pop_solve_rte'][0]
                        hi_close = band[1] <= pop.pf['pop_solve_rte'][1]

                        # As long as within 0.1 eV, call it a match
                        if not lo_close:
                            lo_close = np.allclose(band[0],
                                pop.pf['pop_solve_rte'][0], atol=0.1)
                        if not hi_close:
                            hi_close = np.allclose(band[1],
                                pop.pf['pop_solve_rte'][1], atol=0.1)

                        # Want to make sure we don't have to be *exactly*
                        # right.
                        if band == pop.pf['pop_solve_rte']:
                            tmp.append(True)
                        # Round (close enough?)
                        elif np.allclose(band, pop.pf['pop_solve_rte'],
                            atol=1e-1, rtol=0.):
                            tmp.append(True)
                        elif lo_close and hi_close:
                            tmp.append(True)
                        else:
                            tmp.append(False)
                    #elif type(pop.pf['pop_solve_rte']) is list:
                    #    if band in pop.pf['pop_solve_rte']
                    else:
                        tmp.append(False)

                self._solve_rte.append(tmp)

        assert len(self._solve_rte) == len(self.pops)

        return self._solve_rte

    def _needs_tau(self, popid):
        if self.solve_rte[popid]:
            return True
        else:
            return False

    def _get_bands(self, pop):
        """
        Break radiation field into chunks we know how to deal with.

        For example, HI, HeI, and HeII sawtooth modulation.

        Returns
        -------
        List of band segments that will be handled by the generator.

        """

        Emin, Emax = pop.src.Emin, pop.src.Emax

        # Pure X-ray
        if (Emin > E_LL) and (Emin > 4 * E_LL):
            return [(Emin, Emax)]

        bands = []

        # Check for optical/IR
        if (Emin < E_LyA) and (Emax <= E_LyA):
            bands.append((Emin, Emax))
            return bands

        # Emission straddling Ly-a -- break off low energy chunk.
        if (Emin < E_LyA) and (Emax > E_LyA):
            bands.append((Emin, E_LyA))

            # Keep track as we go
            _Emin_ = np.max(bands)
        else:
            _Emin_ = Emin

        # Check for sawtooth
        if _Emin_ >= E_LyA and _Emin_ < E_LL:
            bands.append((_Emin_, min(E_LL, Emax)))

        #if (abs(Emin - E_LyA) < 0.1) and (Emax >= E_LL):
        #    bands.append((E_LyA, E_LL))
        #elif abs(Emin - E_LL) < 0.1 and (Emax < E_LL):
        #    bands.append((max(E_LyA, E_LL), Emax))

        if Emax <= E_LL:
            return bands

        # Check for HeII
        if Emax > (4 * E_LL):
            bands.append((E_LL, 4 * E_LyA))
            bands.append((4 * E_LyA, 4 * E_LL))
            bands.append((4 * E_LL, Emax))
        else:
            bands.append((E_LL, Emax))

        return bands

    @property
    def approx_all_pops(self):
        if not hasattr(self, '_approx_all_pops'):

            self._approx_all_pops = True
            for i, pop in enumerate(self.pops):

                # Can't use self.approx_rte in this case... :(

                if pop.pf['pop_solve_rte'] == False:
                    continue
                else:
                    self._approx_all_pops = False
                    break

        return self._approx_all_pops

    @property
    def pops(self):
        if not hasattr(self, '_pops'):
            self._pops = CompositePopulation(pf=self.pf, cosm=self.cosm,
                **self._kwargs).pops

        return self._pops

    @property
    def Npops(self):
        return len(self.pops)

    @property
    def energies(self):
        if not hasattr(self, '_energies'):
            bands = self.bands_by_pop
        return self._energies

    @property
    def redshifts(self):
        if not hasattr(self, '_redshifts'):
            bands = self.bands_by_pop
        return self._redshifts

    @property
    def effects_by_pop(self):
        if not hasattr(self, '_effects_by_pop'):
            self._effects_by_pop = [[] for i in range(self.Npops)]

            for i, pop in enumerate(self.pops):
                bands = self.bands_by_pop[i]

                for j, band in enumerate(bands):
                    if band is None:
                        self.effects_by_pop[i].append(None)
                        continue

                    if pop.zone == 'cgm' and band[1] <= E_LL:
                        self._effects_by_pop[i].append(None)
                        continue

                    self._effects_by_pop[i].append(pop.zone)

        return self._effects_by_pop

    @property
    def effects_by_pop(self):
        if not hasattr(self, '_effects_by_pop'):
            self._effects_by_pop = [[] for i in range(self.Npops)]

            for i, pop in enumerate(self.pops):
                bands = self.bands_by_pop[i]

                for j, band in enumerate(bands):
                    if band is None:
                        self.effects_by_pop[i].append(None)
                        continue

                    if pop.zone == 'cgm' and band[1] <= E_LL:
                        self._effects_by_pop[i].append(None)
                        continue

                    self._effects_by_pop[i].append(pop.zone)

        return self._effects_by_pop

    @property
    def bands_by_pop(self):
        if not hasattr(self, '_bands_by_pop'):
            # Figure out which band each population emits in
            if self.approx_all_pops:
                self._energies = [[None] for i in range(self.Npops)]
                self._redshifts = [None for i in range(self.Npops)]
                self._bands_by_pop = [[None] for i in range(self.Npops)]
            else:
                # Really just need to know if it emits ionizing photons,
                # or has any sawtooths we need to care about
                self._bands_by_pop = []
                self._energies = []
                self._redshifts = []
                for i, pop in enumerate(self.pops):
                    bands = self._get_bands(pop)

                    self._bands_by_pop.append(bands)

                    if (bands is None) or (not pop.pf['pop_solve_rte']):
                        z = nrg = ehat = tau = None
                    else:
                        z, nrg, tau, ehat = self._set_grid(pop, bands)

                    self._energies.append(nrg)
                    self._redshifts.append(z)

        return self._bands_by_pop

    @property
    def tau(self):
        if not hasattr(self, '_tau'):
            self._tau = []
            for i, pop in enumerate(self.pops):
                if np.any(self.solve_rte[i]):
                    bands = self.bands_by_pop[i]
                    z, nrg, tau, ehat = self._set_grid(pop, bands,
                        compute_tau=True)
                else:
                    z = nrg = ehat = tau = None

                self._tau.append(tau)

        return self._tau

    @property
    def emissivities(self):
        if not hasattr(self, '_emissivities'):
            self._emissivities = []
            for i, pop in enumerate(self.pops):
                if np.any(self.solve_rte[i]):
                    bands = self.bands_by_pop[i]
                    z, nrg, tau, ehat = self._set_grid(pop, bands,
                        compute_emissivities=True)
                else:
                    z = nrg = ehat = tau = None

                self._emissivities.append(ehat)

        return self._emissivities

    def _set_grid(self, pop, bands, zi=None, zf=None, nz=None,
        compute_tau=False, compute_emissivities=False):
        """
        Create energy and redshift arrays.

        Parameters
        ----------
        pop : int
            Population ID number.
        bands : list
            Each element of this list is a (Emin, Emax) pair defining
            a band in which the RTE is solved.

        Returns
        -------
        Tuple of redshifts and energies for this particular population.

        References
        ----------
        Haardt, F. & Madau, P. 1996, ApJ, 461, 20

        """

        if zi is None:
            zi = pop.pf['first_light_redshift']
        if zf is None:
            zf = pop.pf['final_redshift']
        if nz is None:
            nz = pop.pf['tau_redshift_bins']

        x = np.logspace(np.log10(1 + zf), np.log10(1 + zi), nz)
        z = x - 1.
        R = x[1] / x[0]

        # Loop over bands, build energy arrays
        tau_by_band = []
        energies_by_band = []
        emissivity_by_band = []
        for j, band in enumerate(bands):

            E0, E1 = band

            # Identify bands that should be split into sawtooth components.
            # Be careful to not punish users unnecessarily if Emin and Emax
            # aren't set exactly to Ly-a energy or Lyman limit.
            has_sawtooth  = (abs(E0 - E_LyA) < 0.1) or (abs(E0 - 4 * E_LyA) < 0.1)
            has_sawtooth &= E1 > E_LyA

            # Special treatment if LWB or UVB
            if has_sawtooth:

                HeII = band[0] > E_LL

                E = []
                narr = np.arange(2, self.pf['lya_nmax'])
                for n in narr:
                    Emi = self.hydr.ELyn(n)
                    Ema = self.hydr.ELyn(n + 1)

                    if HeII:
                        Emi *= 4
                        Ema *= 4

                    N = num_freq_bins(nz, zi=zi, zf=zf, Emin=Emi, Emax=Ema)

                    # Create energy array
                    EofN = Emi * R**np.arange(N)

                    # A list of lists!
                    E.append(EofN)

                #if compute_tau and self.pf['tau_clumpy'] is not None:
                #    tau = [self._set_tau(z, Earr, pop)[2] for Earr in E]
                #else:
                tau = [np.zeros([len(z), len(Earr)]) for Earr in E]

                if compute_emissivities:
                    ehat = [self.TabulateEmissivity(z, Earr, pop) for Earr in E]
                else:
                    ehat = None

                # Store stuff for this band
                tau_by_band.append(tau)
                energies_by_band.append(E)
                emissivity_by_band.append(ehat)

            else:
                N = num_freq_bins(x.size, zi=zi, zf=zf, Emin=E0, Emax=E1)

                if pop.src.is_delta or pop.src.has_nebular_lines:
                    E = np.flip(E1 * R**-np.arange(N), 0)
                else:
                    E = E0 * R**np.arange(N)

                # Tabulate optical depth
                if compute_tau and self.solve_rte[pop.id_num][j]:
                    _z, _E, tau = self._set_tau(z, E, pop)
                else:
                    tau = None

                # Tabulate emissivity
                if compute_emissivities:
                    ehat = self.TabulateEmissivity(z, E, pop)
                else:
                    ehat = None

                # Store stuff for this band
                tau_by_band.append(tau)
                energies_by_band.append(E)
                emissivity_by_band.append(ehat)

        return z, energies_by_band, tau_by_band, emissivity_by_band

    @property
    def tau_solver(self):
        if not hasattr(self, '_tau_solver'):
            # Create an ares.simulations.OpticalDepth instance
            for i, pop in enumerate(self.pops):
                if not self._needs_tau(i):
                    continue

                self._tau_solver = OpticalDepth(**pop.pf)

        return self._tau_solver

    @tau_solver.setter
    def tau_solver(self, value):
        self._tau_solver = value

    def _set_tau(self, z, E, pop):
        """
        Tabulate the optical depth.

        The results of this function depend on a few global parameters.

        If tau_approx is True, then the optical depth will be assumed to be
        zero for all redshifts and energies.

        If tau_approx == 'neutral', then the optical depth will be computed
        assuming a neutral medium for all times (including HeI).

        If tau_approx is a function, it is assumed to describe the ionized
        hydrogen fraction as a function of redshift.

        If tau_approx is 'post_EoR', we assume the only species left that
        isn't fully ionized is helium, with 100% of helium having been
        ionized once. That is, xHI = xHeI = 0, xHeII = 1.

        Parameters
        ----------
        z : np.ndarray
            Array of redshifts.
        energies : np.ndarray
            Array of rest-frame photon energies.

        Returns
        -------
        A 2-D array of optical depths, of shape (len(z), len(E)).

        """

        # Default to optically thin if nothing is supplied
        if self.pf['tau_approx'] == True:
            return z, E, np.zeros([len(z), len(E)])

        # Not necessarily true in the future if we include H2 opacity
        if (E.max() <= E_LL):
            return z, E, np.zeros([len(z), len(E)])

        #if (E.min() >= E_LL) and (E.max() < 4 * E_LL):
        #    return z, E, np.inf * np.ones([len(z), len(E)])

        # See if we've got an instance of OpticalDepth already handy
        if self.pf['tau_instance'] is not None:
            self.tau_solver = self.pf['tau_instance']
            return self.tau_solver.z_fetched, self.tau_solver.E_fetched, \
                self.tau_solver.tau_fetched
        elif self.pf['tau_arrays'] is not None:
            # Assumed to be (z, E, tau)
            z, E, tau = self.pf['tau_arrays']
            return z, E, tau

        # Create an ares.simulations.OpticalDepth instance
        # (if we don't have one already)
        tau_solver = self.tau_solver

        # Try to load file from disk.
        if pop.is_src_xray:
            _z, _E, tau = tau_solver._fetch_tau(pop, z, E)
        else:
            _z = z
            _E = E
            tau = np.zeros((_z.size, _E.size))

        # Generate it now if no file was found.
        if tau is None:
            no_tau_table(self)

            if self.pf['tau_approx'] is 'neutral':
                tau_solver.ionization_history = lambda z: 0.0
            elif self.pf['tau_approx'] is 'post_EoR':
                tau_solver.ionization_history = lambda z: 1.0
            elif type(self.pf['tau_approx']) is types.FunctionType:
                tau_solver.ionization_history = self.pf['tau_approx']
            else:
                raise NotImplemented('Unrecognized approx_tau option.')

            tau = tau_solver.TabulateOpticalDepth()

        ##
        # Optional: optical depth from discrete absorbers
        #if self.pf['tau_clumpy'] is not None:
        #    assert self.pf['tau_instance'] is None
        #    assert self.pf['tau_arrays'] is None
        #    tau_d = np.zeros((_z.size, _E.size))
        #    lam = c * 1e8 / (_E * erg_per_ev / h_p)
        #    for i, _z_ in enumerate(_z):
        #        tau_d[i] = tau_solver.ClumpyOpticalDepth(_z_, lam)
        #else:
        #    tau_d = 0.0

        # Return what we got, not what we asked for
        return _z, _E, tau

    @property
    def generators(self):
        """
        Create generators for each population.

        Returns
        -------
        Nothing. Sets attribute `generators`.

        """
        if not hasattr(self, '_generators'):

            self._generators = []
            for i, pop in enumerate(self.pops):
                if not np.any(self.solve_rte[i]):
                    gen = None
                else:
                    gen = self.FluxGenerator(popid=i)

                self._generators.append(gen)

        return self._generators

    def _set_integrator(self):
        """
        Initialize attributes pertaining to numerical integration.
        """

        # For integration over redshift / frequency
        self._integrator = self.pf["unsampled_integrator"]
        self._sampled_integrator = self.pf["sampled_integrator"]
        self._rtol = self.pf["integrator_rtol"]
        self._atol = self.pf["integrator_atol"]
        self._divmax = int(self.pf["integrator_divmax"])

    def update_rate_coefficients(self, z, popid=None, **kwargs):
        """
        Compute ionization and heating rate coefficients.

        Returns
        -------
        Dictionary containing ionization and heating rate coefficients.

        """

        # Setup arrays for results - sorted by sources and absorbers
        # The middle dimension of length 1 is the number of cells
        # which is always 1 for these kinds of calculations
        self.k_ion  = np.zeros([self.Npops, 1, self.grid.N_absorbers])
        self.k_ion2 = np.zeros([self.Npops, 1, self.grid.N_absorbers,
            self.grid.N_absorbers])
        self.k_heat = np.zeros([self.Npops, 1, self.grid.N_absorbers])

        # Loop over sources
        for i, source in enumerate(self.pops):

            ##
            ## What to do for approximate RTE populations?
            ##

            if popid is not None:
                if i != popid:
                    continue

            # Loop over absorbing species
            for j, species in enumerate(self.grid.absorbers):

                if not np.any(self.solve_rte[i]):
                    self._update_by_band_and_species(z, i, j, None, **kwargs)
                    continue

                # Sum over bands
                for k, band in enumerate(self.energies[i]):

                    # Skip ahead if this *band* doesn't have the influence
                    # of its parent population, i.e., UV populations
                    # that emit Ly-a and LyC photons -- must make sure we
                    # don't double count when computing ionization rate.
                    if self.effects_by_pop[i][k] is None:
                        continue

                    # Still may not necessarily solve the RTE
                    if self.solve_rte[i][k]:
                        self._update_by_band_and_species(z, i, j, k,
                            **kwargs)
                    else:
                        self._update_by_band_and_species(z, i, j, None,
                            **kwargs)

        # Sum over sources
        self.k_ion_tot = np.sum(self.k_ion, axis=0)
        self.k_ion2_tot = np.sum(self.k_ion2, axis=0)
        self.k_heat_tot = np.sum(self.k_heat, axis=0)

        to_return = \
        {
         'k_ion': self.k_ion_tot,
         'k_ion2': self.k_ion2_tot,
         'k_heat': self.k_heat_tot,
        }

        return to_return

    def _update_by_band_and_species(self, z, i, j, k, **kwargs):
        """
        i, j, k = source, species, band.
        """
        if kwargs['zone'] in ['igm', 'both']:


            self.k_ion[i,0,j] += \
                self.volume.IonizationRateIGM(z, species=j, popid=i,
                band=k, **kwargs)
            self.k_heat[i,0,j] += \
                self.volume.HeatingRate(z, species=j, popid=i,
                band=k, **kwargs)

            for h, donor in enumerate(self.grid.absorbers):
                self.k_ion2[i,0,j,h] += \
                    self.volume.SecondaryIonizationRateIGM(z,
                    species=j, donor=h, popid=i, band=k, **kwargs)

        elif kwargs['zone'] in ['cgm', 'both']:

            Gamma = self.volume.IonizationRateCGM(z, species=j, popid=i,
                band=k, **kwargs)

            self.k_ion[i,0,j] += Gamma

    def LymanWernerFlux(self, z, E=None, popid=0, **kwargs):
        """
        Compute flux at observed redshift z and energy E (eV).

        Parameters
        ----------
        z : float
            observer redshift
        E : float
            observed photon energy (eV)

        ===============
        relevant kwargs
        ===============
        tau : func, e.g. tau = lambda E, z1, z2: 0.0 # const. tau
            If supplied, represents the optical depth between redshifts z1
            and z2 as a function of observed energy E.
        xavg : func, array
            Average ionized fraction. Can be function of redshift, or array
            of values
        zxavg : array
            If xavg is an array, this is the array of corresponding redshifts.
        zf : float
            Upper limit of redshift integration (i.e. exclude emission from
            sources at z' > zf).

        Returns
        -------
        Flux in units of erg s**-1 cm**-2 Hz**-1 sr**-1

        """

        pop = self.pops[popid]

        kw = defkwargs.copy()
        kw.update(kwargs)

        # Flat spectrum, no injected photons, instantaneous emission only
        if not np.any(self.solve_rte[popid]):
            norm = c * self.cosm.dtdz(z) / four_pi

            rhoLW = pop.PhotonLuminosityDensity(z, Emin=E_LyA, Emax=E_LL) \
                * (E_LL - 11.18) / (E_LL - E_LyA)

            # Crude mean photon energy
            dnu_LW = (E_LL - 11.18) / ev_per_hz
            return 0.5 * (11.2 + 13.6) * erg_per_ev * norm * (1. + z)**3 \
                * rhoLW / dnu_LW

        else:
            raise NotImplemented('this shouldnt happen')

        # Closest Lyman line (from above)
        n = ceil(np.sqrt(E_LL / (E_LL - E)))

        if n > self.pf['lya_nmax']:
            return 0.0

        En =  E_LL * (1. - 1. / n**2)

        # Corresponding zmax ("dark screen" as Z. Haiman likes to say)
        if kw['tau'] == 0.0:
            if kw['zf'] is None:
                zmax = pop.zform
            else:
                zmax = kw['zf']
        else:
            zmax = En * (1. + z) / E - 1.

        zmax = min(zmax, pop.zform)

        # Normalize to help integrator
        Jc = 1e-10

        integrand = lambda zu: self.AngleAveragedFluxSlice(z, E, zu,
            tau=0.0) / Jc

        flux = quad(integrand, z, zmax,
            epsrel=self._rtol, epsabs=self._atol, limit=self._divmax)[0]

        # Flux in units of photons s^-1 cm^-2 Hz^-1 sr^-1
        flux *= Jc

        # Possibly convert to energy flux units
        if kw['energy_units']:
            flux *= E * erg_per_ev

        return flux

    @property
    def frec(self):
        if not hasattr(self, '_frec'):
            n = np.arange(2, self.pf['lya_nmax'])
            self._frec = np.array(list(map(self.hydr.frec, n)))

        return self._frec

    @property
    def narr(self):
        if not hasattr(self, '_narr'):
            self._narr = np.arange(2, self.pf['lya_nmax'])

        return self._narr

    def LymanAlphaFlux(self, z=None, fluxes=None, popid=0, **kwargs):
        """
        Compute background flux at Lyman-alpha resonance.

        ..note:: Optionally includes products of Ly-n cascades if approx_lwb=0.

        Parameters
        ----------
        z : int, float
            Redshift of interest
        fluxes : np.ndarray
            Fluxes grouped by LW band at a single redshift.

        Returns
        -------
        Lyman alpha flux at given redshift.

        """

        pop = self.pops[popid]

        if (not pop.is_src_lya) or (z > pop.zform):
            return 0.0

        if pop.pf['pop_Ja'] is not None:
            return pop.LymanAlphaFlux(z)

        # Flat spectrum, no injected photons, instantaneous emission only
        if not np.any(self.solve_rte[popid]):
            norm = c * self.cosm.dtdz(z) / four_pi

            rhoLW = pop.PhotonLuminosityDensity(z, Emin=E_LyA, Emax=E_LL)

            return norm * (1. + z)**3 * (1. + pop.pf['pop_frec_bar']) * \
                rhoLW / dnu

        raise ValueError('this should only be called if approx RTE')

        # Full calculation
        J = 0.0

        for i, n in enumerate(self.narr):

            # Continuum photons included by default
            if n == 2:
                continue

            # Injected line photons optional
            if n > 2 and not pop.pf['include_injected_lya']:
                continue

            Jn = self.hydr.frec(n) * fluxes[i][0]

            J += Jn

        return J

    def load_sed(self, prefix=None):
        fn = pop.src.sed_name()

        if prefix is None:
            if not ARES:
                print("No $ARES environment variable.")
                return None

            input_dirs = ['{!s}/input/seds'.format(ARES)]

        else:
            if isinstance(prefix, basestring):
                input_dirs = [prefix]
            else:
                input_dirs = prefix

        guess = '{0!s}/{1!s}.txt'.format(input_dirs[0], fn)
        self.tabname = guess
        if os.path.exists(guess):
            return guess

        pre, tmp2 = fn.split('_logE_')
        post = '_logE_' + tmp2.replace('.txt', '')

        good_tab = None
        for input_dir in input_dirs:
            for fn1 in os.listdir(input_dir):

                # If source properties are right
                if re.search(pre, fn1):
                    good_tab = '{0!s}/{1!s}'.format(input_dir, fn1)

                # If number of redshift bins and energy range right...
                if re.search(pre, fn1) and re.search(post, fn1):
                    good_tab = '{0!s}/{1!s}'.format(input_dir, fn1)
                    break

        self.tabname = good_tab
        return good_tab

    def TabulateEmissivity(self, z, E, pop):
        """
        Tabulate emissivity over photon energy and redshift.

        For a scalable emissivity, the tabulation is done for the emissivity
        in the (EminNorm, EmaxNorm) band because conversion to other bands
        can simply be applied after-the-fact. However, if the emissivity is
        NOT scalable, then it is tabulated separately in the (10.2, 13.6),
        (13.6, 24.6), and X-ray band.

        Parameters
        ----------
        z : np.ndarray
            Array of redshifts
        E : np.ndarray
            Array of photon energies [eV]
        pop : object
            Better be some kind of Galaxy population object.

        Returns
        -------
        A 2-D array, first axis corresponding to redshift, second axis for
        photon energy.

        Units of emissivity are: erg / s / Hz / cMpc

        """

        Nz, Nf = len(z), len(E)

        Inu = np.zeros(Nf)

        # Special case: delta function SED! Can't normalize a-priori without
        # knowing binning, so we do it here.
        if pop.src.is_delta:
            # This is a little weird. Trapezoidal integration doesn't make
            # sense for a delta function, but it's what happens later, so
            # insert a factor of a half now so we recover all the flux we
            # should.
            Inu[-1] = 1.
        else:
            for i in range(Nf):
                Inu[i] = pop.src.Spectrum(E[i])

        # Convert to photon *number* (well, something proportional to it)
        Inu_hat = Inu / E

        # Now, redshift dependent parts
        epsilon = np.zeros([Nz, Nf])

        #if Nf == 1:
        #    return epsilon

        scalable = pop.is_emissivity_scalable
        separable = pop.is_emissivity_separable

        H = np.array(list(map(self.cosm.HubbleParameter, z)))

        if scalable:
            Lbol = pop.Emissivity(z)
            for ll in range(Nz):
                epsilon[ll,:] = Inu_hat * Lbol[ll] * ev_per_hz / H[ll] \
                    / erg_per_ev

        else:

            # There is only a distinction here for computational
            # convenience, really. The LWB gets solved in much more detail
            # than the LyC or X-ray backgrounds, so it makes sense
            # to keep the different emissivity chunks separate.
            ct = 0
            for band in [(10.2, 13.6), (13.6, 24.6), None]:

                if band is not None:

                    if pop.src.Emin > band[1]:
                        continue

                    if pop.src.Emax < band[0]:
                        continue

                # Remind me of this distinction?
                if band is None:
                    b = pop.full_band
                    fix = 1.

                    # Means we already generated the emissivity.
                    if ct > 0:
                        continue

                else:
                    b = band

                    # If aging population, is handled within the pop object.
                    if not pop.is_aging:
                        fix = 1. / pop._convert_band(*band)
                    else:
                        fix = 1.

                in_band = np.logical_and(E >= b[0], E <= b[1])

                # Shouldn't be any filled elements yet
                if np.any(epsilon[:,in_band==1] > 0):
                    raise ValueError("Non-zero elements already!")

                if not np.any(in_band):
                    continue

                ###
                # No need for spectral correction in this case, at least
                # in Lyman continuum. Treat LWB more carefully.
                if pop.is_aging and band == (13.6, 24.6):
                    fix = 1. / Inu_hat[in_band==1]

                elif pop.is_aging and band == (10.2, 13.6):

                    if pop.pf['pop_synth_lwb_method'] == 0:
                        # No approximation: loop over energy below
                        raise NotImplemented('sorry dude')
                    elif pop.pf['pop_synth_lwb_method'] == 1:
                        # Assume SED of continuousy-star-forming source.
                        Inu_hat_p = pop._src_csfr.Spectrum(E[in_band==1]) \
                            / E[in_band==1]
                        fix = Inu_hat_p / Inu_hat[in_band==1][0]
                    else:
                        raise NotImplemented('sorry dude')
                ###

                # By definition, rho_L integrates to unity in (b[0], b[1]) band
                # BUT, Inu_hat is normalized in (EminNorm, EmaxNorm) band,
                # hence the 'fix'.

                for ll, redshift in enumerate(z):

                    if (redshift < self.pf['final_redshift']):
                        continue
                    if (redshift < pop.zdead):
                        continue
                    if (redshift > pop.zform):
                        continue
                    if redshift < self.pf['kill_redshift']:
                        continue
                    if redshift > self.pf['first_light_redshift']:
                        continue

                    # Use Emissivity here rather than rho_L because only
                    # GalaxyCohort objects will have a rho_L attribute.
                    epsilon[ll,in_band==1] = fix \
                        * pop.Emissivity(redshift, Emin=b[0], Emax=b[1]) \
                        * ev_per_hz * Inu_hat[in_band==1] / H[ll] / erg_per_ev

                    #ehat = pop.Emissivity(redshift, Emin=b[0], Emax=b[1])

                    #if ll == 1:
                    #    print("Set emissivity for pop {} band #{}".format(pop.id_num, band))
                    ##    print(f'fix={fix}, raw={ehat} z={redshift}')

                ct += 1

        return epsilon

    def _flux_generator_generic(self, energies, redshifts, ehat, tau=None,
        flux0=None, my_id=None, accept_photons=False):
        """
        Generic flux generator.

        Parameters
        ----------
        energies : np.ndarray
            1-D array of photon energies
        redshifts : np.ndarray
            1-D array of redshifts
        ehat : np.ndarray
            2-D array of tabulate emissivities (divided by H(z)).
        tau : np.ndarray
            2-D array of optical depths, or reference to an array that will
            be modified with time.
        flux0 : np.ndarray
            1-D array of initial flux values.

        """

        # Some stuff we need
        x = 1. + redshifts
        xsq = x**2
        R = x[1] / x[0]
        Rsq = R**2

        # Shorthand
        zarr = redshifts

        # Remember what band I'm in and what Population I belong to.
        popid, bandid = my_id

        # Should receive Ly-A? This is a bit hacky.
        receive_lya = self.pops[popid].pf['pop_lya_permeable'] \
            and energies[-1] < E_LyA and abs(E_LyA - energies[-1]) < 0.2

        receive_lyn = (energies[0] == E_LyA) and self.pf['include_injected_lya']

        if receive_lyn:
            narr = self.narr
            En = self.grid.hydr.ELyn(narr)

        # Initialize flux-passing to zero
        self._fluxes_from[(popid, bandid)] = energies[0], 0.0

        #if tau is None:
        #    if type(energies) is list:
        #        tau = [np.zeros([redshifts.size, Earr.size]) \
        #            for Earr in energies]
        #    else:
        #        tau = np.zeros([redshifts.size, energies.size])

        if flux0 is None:
            flux = np.zeros_like(energies)
        else:
            flux = flux0

        L = redshifts.size
        ll = self._ll = L - 1

        otf = False

        # Pre-roll some stuff
        tau_r = np.roll(tau, -1, axis=1)
        ehat_r = np.roll(np.roll(ehat, -1, axis=0), -1, axis=1)
        # Won't matter that we carried the first element to the end because
        # the incoming flux in that bin is always zero.


        # Loop over redshift - this is the generator
        z = redshifts[-1]
        while z >= redshifts[0]:

            # First iteration: no time for there to be flux yet
            # (will use argument flux0 if the EoR just started)
            if ll == (L - 1):
                pass

            # General case
            else:

                if otf:
                    exp_term = np.exp(-np.roll(tau, -1))
                else:
                    exp_term = np.exp(-tau_r[ll])

                # Special case: delta function SED
                if self.pops[popid].src.is_delta:
                    trapz_base = 1.
                else:
                    trapz_base = 0.5 * (zarr[ll+1] - zarr[ll])

                # Equivalent to Eq. 25 in Mirocha (2014)
                # Less readable, but faster!
                flux = c_over_four_pi \
                    * ((xsq[ll] * trapz_base) * ehat[ll]) \
                    + exp_term * (c_over_four_pi * xsq[ll+1] \
                    * trapz_base * ehat_r[ll] \
                    + np.hstack((flux[1:], [0])) / Rsq)
                    #+ np.roll(flux, -1) / Rsq)

                ##
                # Add Ly-a flux from cascades
                ##
                if receive_lyn:

                    # Loop over higher Ly-n lines, grab fluxes, * frec
                    for i, k in enumerate(range(bandid, bandid+len(narr))):

                        # k is a band ID number
                        # i is just an index
                        _E, J = self._fluxes_from[(popid, k)]

                        n = narr[i]

                        # This is Ly-a flux, which we've already got!
                        if n == 2:
                            continue

                        assert _E == En[i]

                        Jn = self.grid.hydr.frec(n) * J
                        flux[0] += Jn

                    # Still need to set flux[-1] = 0!
                    # Will happen below in 'else' bracket

                # This band receives Ly-a
                if receive_lya and (not receive_lyn):
                    # Note that this is redundant, the two will never
                    # both be True. But, safety first, kids.

                    if (bandid+1) in self._fluxes_from:

                        # Inbound flux
                        in_flux = self._fluxes_from[(popid, bandid+1)][1]

                        ##
                        # Very approximate at the moment. Could correct
                        # for slight redshifting and dilution.
                        ##
                        flux[-1] = in_flux
                    else:
                        flux[-1] = 0.0

                # Otherwise, can be no flux at highest energy, because SED
                # is truncated and there's no where it could have come from.
                #elif not accept_photons:
                elif energies[-1] != E_LyA:
                    flux[-1] = 0.0

                self._fluxes_from[my_id] = energies[0], flux[0]

            ##
            # Yield results, move on to next iteration
            ##
            yield redshifts[ll], flux

            # Increment redshift
            ll -= 1
            z = redshifts[ll]

            if ll == -1:
                break

    def _flux_generator_sawtooth(self, E, z, ehat, tau, my_id=None):
        """
        Create generators for the flux between all Lyman-n bands.
        """

        imax = len(E) - 1

        gens = []
        for i, nrg in enumerate(E):
            gens.append(self._flux_generator_generic(nrg, z, ehat[i],
                tau[i], my_id=(my_id[0], my_id[1]+i), accept_photons=i!=imax))

        # Generator over redshift
        for i in range(z.size):
            flux = []
            for gen in gens:
                z, new_flux = next(gen)
                flux.append(new_flux)

            # Increment fluxes
            #line_flux = self._compute_line_flux(flux)

            yield z, flux #+ flatten_flux(line_flux)

    def FluxGenerator(self, popid):
        """
        Evolve some radiation background in time.

        Parameters
        ----------
        popid : str
            Create flux generator for a single population.

        Returns
        -------
        Generator for the flux, each in units of s**-1 cm**-2 Hz**-1 sr**-1.
        Each step returns the current redshift, and the flux as a function of
        energy at the current redshift.

        """

        # List of all intervals in rest-frame photon energy
        bands = self.bands_by_pop[popid]

        ct = 0
        generators_by_band = []
        for i, band in enumerate(bands):
            if not self.solve_rte[popid][i]:
                gen = None
                ct += 1
            elif type(self.energies[popid][i]) is list:
                gen = self._flux_generator_sawtooth(E=self.energies[popid][i],
                    z=self.redshifts[popid], ehat=self.emissivities[popid][i],
                    tau=self.tau[popid][i], my_id=(popid,ct))
                ct += len(self.energies[popid][i])
            else:

                gen = self._flux_generator_generic(self.energies[popid][i],
                    self.redshifts[popid], self.emissivities[popid][i],
                    tau=self.tau[popid][i], my_id=(popid,ct))
                ct += 1

            generators_by_band.append(gen)

        return generators_by_band

    @property
    def _fluxes_from(self):
        """
        For internal use only.

        A storage container for fluxes to be passed from one generator to the
        next, to create continuity between sub-bands.

        """
        if not hasattr(self, '_fluxes_from_'):
            self._fluxes_from_ = {}

        return self._fluxes_from_
