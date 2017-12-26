"""

UniformBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:15:36 MDT 2014

Description: 

"""

import numpy as np
from math import ceil
import os, re, types, gc
from ..util.Misc import logbx
from ..util import ParameterFile
from ..static import GlobalVolume
from ..util.Misc import num_freq_bins
from scipy.interpolate import interp1d
from .OpticalDepth import OpticalDepth
from ..util.Warnings import no_tau_table
from ..physics import Hydrogen, Cosmology
from ..populations.Composite import CompositePopulation
from ..populations.GalaxyAggregate import GalaxyAggregate
from scipy.integrate import quad, romberg, romb, trapz, simps
from ..physics.Constants import ev_per_hz, erg_per_ev, c, E_LyA, E_LL, dnu
from ..util.ReadData import flatten_energies, flatten_flux, split_flux, \
    flatten_emissivities
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

ARES = os.getenv('ARES')

log10 = np.log(10.)    # for when we integrate in log-space
four_pi = 4. * np.pi

E_th = np.array([13.6, 24.4, 54.4])

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
    def __init__(self, grid=None, **kwargs):
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
        self.pf = ParameterFile(**kwargs)
        
        # Some useful physics modules
        if grid is not None:
            self.grid = grid
            self.cosm = grid.cosm
        else:
            self.grid = None
            self.cosm = Cosmology(**self.pf)

        self._set_integrator()
        
    @property
    def OpticalDepth(self):
        if not hasattr(self, '_OpticalDepth'):
            self._OpticalDepth = OpticalDepth(**self.pf)
        return self._OpticalDepth
        
    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(self.cosm, **self.pf)

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
                        if band == pop.pf['pop_solve_rte']:
                            tmp.append(True)
                        # Eh, close  enough (to 0.1 eV)
                        elif np.allclose(band, pop.pf['pop_solve_rte'], 
                            atol=1e-1, rtol=0.):
                            tmp.append(True)
                        else:
                            tmp.append(False)
                            
                self._solve_rte.append(tmp)            

        assert len(self._solve_rte) == len(self.pops)                
                                
        return self._solve_rte
        
    def _needs_tau(self, popid):
        if self.solve_rte[popid] and self.pops[popid].pf['pop_Emin'] >= E_LL:
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
        
        Emin, Emax = pop.pf['pop_Emin'], pop.pf['pop_Emax']
        
        # Pure X-ray
        if (Emin > E_LL) and (Emin > 4 * E_LL):
            return [(Emin, Emax)]
        
        bands = []
        
        # Check for optical/IR
        if Emin < E_LyA:
            bands.append((Emin, E_LyA))
        
        # Check for sawtooth
        if Emin < E_LL:
            bands.append((E_LyA, E_LL))

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
            self._pops = CompositePopulation(**self._kwargs).pops
            
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
            has_sawtooth = (E0 == E_LyA) and (E1 == E_LL)
            has_sawtooth |= (E0 == 4*E_LyA) or (E1 == 4*E_LL)
                        
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
                
                if compute_tau:
                    # Don't allow optical depth in this region 
                    #if band == (E_LyA, E_LL) or (4 * E_LyA, 4 * E_LL):
                    tau = [np.zeros([len(z), len(Earr)]) for Earr in E]
                    #else:
                    #    tau = [self._set_tau(z, Earr, pop) for Earr in E]
                else:
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
        if E.max() <= E_LL:
            return z, E, np.zeros([len(z), len(E)])
                    
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
        _z, _E, tau = tau_solver._fetch_tau(pop, z, E)
                        
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
            self.k_ion[i,0,j] += \
                self.volume.IonizationRateCGM(z, species=j, popid=i,
                band=k, **kwargs)
        
    def AngleAveragedFlux(self, z, E, popid=0, **kwargs):
        """
        Compute flux at observed redshift z and energy E (eV).

        Local flux (i.e. flux at redshift z) depends (potentially) on emission 
        from sources at all redshifts z' > z. This method performs an integral
        over redshift, properly accounting for geometrical dilution, redshift,
        source SEDs, and the fact that emissivities were (possibly) different
        at higher redshift. That is, we actually solve the cosmological 
        radiative transfer equation.

        Parameters
        ----------
        z : float
            observer redshift
        E : float
            observed photon energy (eV)

        ===============
        relevant kwargs
        ===============
        tau : func, e.g. tau = lambda E, z1, z2: 0.0
            If supplied, represents the optical depth between redshifts z1
            and z2 as a function of observed energy E.
        xavg : func, array
            Average ionized fraction. Can be function of redshift, or array
            of values.
        zxavg : array
            If xavg is an array, this is the array of corresponding redshifts.  
        zf : float
            Upper limit of redshift integration (i.e. exclude emission from
            sources at z' > zf).
    
        Notes
        -----
        If none of the "relevant kwargs" are passed, will assume a neutral 
        medium.
    
        Returns
        -------
        Flux in units of s**-1 cm**-2 Hz**-1 sr**-1.
    
        See Also
        --------
        AngleAveragedFluxSlice : the function we're integrating over.
    
        """
        
        pop = self.pops[popid]
    
        if E < E_LyA:
            thin = False
            if 'tau' in kwargs:
                if kwargs['tau'] == 0.0:
                    thin = True
    
            flux = self.LymanWernerFlux(z, E, **kwargs)  
    
            if thin:
                return flux
    
            ze = (E_LyA / E) * (1. + z) - 1.
            return flux + self.LymanAlphaFlux(ze, **kwargs) \
                * ((1. + z) / (1. + ze))**2
    
        if E <= E_LL:
            return self.LymanWernerFlux(z, E, **kwargs)
    
        kw = defkwargs.copy()
        kw.update(kwargs)
    
        # Set limits of integration in redshift space
        zi = max(z, pop.zdead)
        if kw['zf'] is None:
            zf = min(pop.zform, self.pf['first_light_redshift'])
        else:
            zf = kw['zf']
    
        # Normalize to help integrator
        Jc = 1e-21
    
        # Define integrand              
        #if kw['tau'] is not None:  # like zarr
        #    if type(kw['tau']) is types.FunctionType:
        #        integrand = lambda zz: self.AngleAveragedFluxSlice(z, E, zz,
        #            **kwargs) / Jc
        #    else:
        #        # Compute flux at this energy due to emission at z' > z
        #        integrand = np.zeros(len(kw['zxavg']))
        #        for i in np.arange(len(kw['zxavg'])):
        #            integrand[i] = self.AngleAveragedFluxSlice(z, E, 
        #                kw['zxavg'][i], tau=kw['tau'][i],
        #                xray_emissivity=None) / Jc
    
        #if kw[''] is not None:
        #if type(kw['xavg']) is types.FunctionType:
        integrand = lambda zu: self.AngleAveragedFluxSlice(z, E, zu,
            xavg=kw['xavg']) / Jc
        #else:
        #    integrand = np.array(list(map(lambda zu: \
        #        self.AngleAveragedFluxSlice(z, E, zu,
        #        xavg=kw['xavg'], zxavg=kw['zxavg']), kw['zxavg']))) / Jc
        #else:
        #    # Assume neutral medium
        #    integrand = lambda zu: self.AngleAveragedFluxSlice(z, E, zu,
        #        h_2=lambda zz: 0.0) / Jc
    
        # Compute integral
        if type(integrand) == types.FunctionType:
            if pop.burst:
                raise ValueError('Burst needs correctness-check.')
                #flux = integrand(self.pop.zform)
            elif self._integrator == 'quad':
                flux = quad(integrand, zi, zf,
                    epsrel=self._rtol, epsabs=self._atol, limit=self._divmax)[0]
            elif self._integrator == 'romb':
                flux = romberg(integrand, zi, zf,
                    tol=self._atol, divmax=self._divmax)
            else:
                raise ValueError('Uncrecognized integrator \'{!s}\''.format(\
                    self._integrator))
        else:
            if self._sampled_integrator == 'simps':
                flux = simps(integrand, x=kw['zxavg'], even='first')
            elif self._sampled_integrator == 'trapz':
                flux = trapz(integrand, x=kw['zxavg'])
            elif self._sampled_integrator == 'romb':
    
                assert logbx(2, len(kw['zxavg']) - 1) % 1 == 0, \
                    "If sampled_integrator == 'romb', redshift_bins must be a power of 2 plus one."
    
                flux = romb(integrand, dx=np.diff(kw['zxavg'])[0])   
            else:
                raise ValueError('Uncrecognized integrator \'{!s}\''.format(\
                    self._sampled_integrator))
    
        # Flux in units of photons s^-1 cm^-2 Hz^-1 sr^-1                                        
        flux *= Jc
    
        # Possibly convert to energy flux units
        if kw['energy_units']:
            flux *= E * erg_per_ev
    
        return flux
    
    def AngleAveragedFluxSlice(self, z, E, zp, popid=0, **kwargs):
        """
        Compute flux at observed redshift z due to sources at higher redshift.
    
        This is the integrand of 'AngleAveragedFlux,' the integral over 
        redshift we must compute to determine the specific flux at any given 
        redshift. It is the contribution to the specific flux at observed
        redshift z from sources at a single redshift, zp > z.
    
        Parameters
        ----------
        z : float
            observer redshift
        E : float
            observed photon energy (eV)
        zp : float
            redshift where photons were emitted
    
        Notes
        -----
        Will assume optically thin medium if none of the following kwargs
        are passed: tau, xavg, emissivity.    
    
        ===============
        relevant kwargs
        ===============
        tau : func, e.g. tau = lambda z1, z2, E: 0.0 # const. tau
            If supplied, represents the optical depth between redshifts z1
            and z2 as a function of observed energy E. 
        xavg : func, np.ndarray
            Average ionized fraction. Can be function of redshift, or array
            of values
        zxavg : np.ndarray
            If xavg is an array, this is the array of corresponding redshifts.
        xray_emissivity : np.ndarray
    
        Returns
        -------
        Flux in units of s**-1 cm**-2 Hz**-1 sr**-1.
    
        See Also
        --------
        AngleAveragedFlux : integrates over this function.
    
        """
    
        pop = self.pops[popid]
    
        kw = defkwargs.copy()
        kw.update(kwargs)
    
        if kw['xray_emissivity'] is None: # should include LyA too
            H = self.cosm.HubbleParameter(zp)
            E0 = self.volume.RestFrameEnergy(z, E, zp)
            epsilonhat = pop.NumberEmissivity(zp, E0)
            epsilonhat_over_H = epsilonhat / H
    
            if (E0 > pop.src.Emax) or (E0 < pop.src.Emin):
                return 0.0
    
        else:
            epsilonhat_over_H = kw['xray_emissivity']
    
        # Compute optical depth (perhaps)
        if kw['tau'] is not None:
            if type(kw['tau']) is types.FunctionType:
                tau = kw['tau'](z, zp, E)
            else:
                tau = kw['tau']
        elif kw['xavg'] is not None:
            if E > E_LL:
                tau = self.volume.OpticalDepth(z, zp, E, xavg=kw['xavg'],
                    zxavg=kw['zxavg'])
            else:
                tau = 0.0
        else:
            raise NotImplemented('this needs fixing')
            tau = self.volume.OpticalDepth(z, zp, E, xavg=kw['xavg'])
    
        return c * (1. + z)**2 * epsilonhat_over_H * np.exp(-tau) / four_pi
        
    def LymanWernerFlux(self, z, E=None, popid=0, **kwargs):
        """
        Compute flux at observed redshift z and energy E (eV).
    
        Same as AngleAveragedFlux, but for emission in the Lyman-Werner band.
    
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
    
        See Also
        --------
        AngleAveragedFluxSlice : the function we're integrating over.
    
        """
        
        pop = self.pops[popid]
    
        kw = defkwargs.copy()
        kw.update(kwargs)

        # Flat spectrum, no injected photons, instantaneous emission only
        if not np.any(self.solve_rte[popid]):
            norm = c * self.cosm.dtdz(z) / four_pi

            rhoLW = pop.PhotonLuminosityDensity(z, Emin=10.2, Emax=13.6) \
                * (E_LL - 11.18) / (E_LL - E_LyA)

            # Crude mean photon energy
            dnu_LW = (E_LL - 11.18) / ev_per_hz
            return 0.5 * (11.2 + 13.6) * erg_per_ev * norm * (1. + z)**3 \
                * rhoLW / dnu_LW

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

            rhoLW = pop.PhotonLuminosityDensity(z, Emin=10.2, Emax=13.6)

            return norm * (1. + z)**3 * (1. + pop.pf['pop_frec_bar']) * \
                rhoLW / dnu

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
        for i in range(Nf): 
            Inu[i] = pop.src.Spectrum(E[i])

        # Convert to photon energy (well, something proportional to it)
        Inu_hat = Inu / E

        # Now, redshift dependent parts    
        epsilon = np.zeros([Nz, Nf])
        
        #if Nf == 1:
        #    return epsilon

        scalable = pop.is_emissivity_scalable
        separable = pop.is_emissivity_separable

        H = np.array(list(map(self.cosm.HubbleParameter, z)))

        if scalable:
            for ll in range(Nz):
                Lbol = pop.Emissivity(z[ll])
                epsilon[ll,:] = Inu_hat * Lbol * ev_per_hz / H[ll] \
                    / erg_per_ev
        else:
                
            # There is only a distinction here for computational
            # convenience, really. The LWB gets solved in much more detail
            # than the LyC or X-ray backgrounds, so it makes sense 
            # to keep the different emissivity chunks separate.                                  
            for band in [(10.2, 13.6), (13.6, 24.6), None]:
                                
                if band is not None:
                    if pop.pf['pop_Emin'] > band[1]:
                        continue
                    
                    if pop.pf['pop_Emax'] < band[0]:
                        continue
                
                # Remind me of this distinction?
                if band is None:
                    b = pop.full_band
                    fix = 1.
                else:
                    b = band
                    fix = 1. / pop._convert_band(*band)
                
                # Setup interpolant
                # If there's an attribute error here, it probably means
                # is_emissivity_scalable isn't being set correctly.
                rho_L = pop.rho_L(Emin=b[0], Emax=b[1])
                
                if rho_L is None:
                    continue

                in_band = np.logical_and(E >= b[0], E <= b[1])

                # By definition, rho_L integrates to unity in (b[0], b[1]) band
                # BUT, Inu_hat is normalized in (EminNorm, EmaxNorm) band

                for ll, redshift in enumerate(z):
                    
                    if redshift < self.pf['final_redshift']:
                        continue    
                    if redshift < self.pf['kill_redshift']:
                        continue    
                    if redshift > self.pf['first_light_redshift']:
                        continue
                                            
                    epsilon[ll,in_band] = fix \
                        * pop.Emissivity(redshift, Emin=b[0], Emax=b[1]) \
                        * ev_per_hz * Inu_hat[in_band] / H[ll] / erg_per_ev

        return epsilon

    def _flux_generator_generic(self, energies, redshifts, ehat, tau=None,
        flux0=None):
        """
        Generic flux generator.

        Parameters
        ----------
        energies : np.ndarray
            1-D array of photon energies
        redshifts : np.ndarray
            1-D array of redshifts
        ehat : np.ndarray
            2-D array of tabulate emissivities.
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
                    exp_term = np.exp(-np.roll(tau[ll], -1))
    
                trapz_base = 0.5 * (zarr[ll+1] - zarr[ll])

                # Equivalent to Eq. 25 in Mirocha (2014)
                # Less readable, but faster!
                flux = (c / four_pi) \
                    * ((xsq[ll+1] * trapz_base) * ehat[ll]) \
                    + exp_term * ((c / four_pi) * xsq[ll+1] \
                    * trapz_base * np.roll(ehat[ll+1], -1, axis=-1) \
                    + np.roll(flux, -1) / Rsq)
                                    
            # No higher energies for photons to redshift from.
            # An alternative would be to extrapolate, and thus mimic a
            # background spectrum that is not truncated at Emax
            flux[-1] = 0.0
                
            yield redshifts[ll], flux
    
            # Increment redshift
            ll -= 1
            z = redshifts[ll]
    
            if ll == -1:
                break
                
    def _compute_line_flux(self, fluxes):  
        """
        Compute emission in lines.
        
        ..note:: Includes Ly-a emission only at this point.
        
        Parameters
        ----------
        List of fluxes, for each sub-band in a sawtooth generator.
        
        """     
        
        line_flux = [np.zeros_like(fluxes[i]) for i in range(len(fluxes))]

        # Compute Lyman-alpha flux
        if self.pf['include_injected_lya']:
            line_flux[0][0] += self.LymanAlphaFlux(z=None, fluxes=fluxes)
        
        return line_flux

    def _flux_generator_sawtooth(self, E, z, ehat, tau):
        """
        Create generators for the flux between all Lyman-n bands.
        """

        gens = []
        for i, nrg in enumerate(E):
            gens.append(self._flux_generator_generic(nrg, z, ehat[i], tau[i]))

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
        
        generators_by_band = []
        for i, band in enumerate(bands):
            if not self.solve_rte[popid][i]:
                gen = None
            elif type(self.energies[popid][i]) is list:   
                gen = self._flux_generator_sawtooth(E=self.energies[popid][i],
                    z=self.redshifts[popid], ehat=self.emissivities[popid][i],
                    tau=self.tau[popid][i])                    
            else:        
                gen = self._flux_generator_generic(self.energies[popid][i],
                    self.redshifts[popid], self.emissivities[popid][i],
                    tau=self.tau[popid][i])

            generators_by_band.append(gen)

        return generators_by_band
        
    
