"""

UniformBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 15:15:36 MDT 2014

Description: This will be like glorb.evolve.RadiationBackground.

"""

import numpy as np
from math import ceil
import os, re, types, gc
from ..physics.Constants import *
from .IntergalacticMedium import IGM
from ..util.PrintInfo import print_rb
from scipy.interpolate import interp1d
from ..physics import Hydrogen, Cosmology
from ..util.Misc import parse_kwargs, logbx
from ..populations import StellarPopulation
from scipy.integrate import quad, romberg, romb, trapz, simps
from ..populations import BlackHolePopulation, StellarPopulation

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

class UniformBackground:
    def __init__(self, pop=None, use_tab=True, **kwargs):
        """
        Initialize a UniformBackground object.
        
        Creates an object capable of evolving the radiation background created
        by some population of objects, which are characterized by a comoving 
        volume emissivity and a spectrum. The evolution of the IGM opacity can 
        be computed self-consistently or imposed artificially.
        
        Parameters
        ----------
        pop : glorb.populations.StellarPopulation instance, or
              glorb.populations.BlackHolePopulation instance
        
        """
                
        if pop is not None:
            self.pop = pop
            self.pf = self.pop.pf
        else:
            self.pf = parse_kwargs(**kwargs)
            
            if self.pf['source_type'] == 'star':
                self.pop = StellarPopulation(**kwargs)
            else:
            #elif self.pf['source_type'] == 'bh':
                self.pop = BlackHolePopulation(**kwargs)
            #else:
            #    raise ValueError('source_type %s not recognized.' \
            #        % self.pf['source_type'])
                 
        # Some useful physics modules
        self.cosm = self.pop.cosm
        self.hydr = Hydrogen(self.pop.cosm, 
            approx_Salpha=self.pf['approx_Salpha'])
        
        # IGM instance
        self.igm = IGM(rb=self, use_tab=use_tab, **kwargs)
        
        self._set_integrator()
        
        if self.pf['verbose'] and \
            (not self.pf['approx_lya'] or not self.pf['approx_xray']):
             print_rb(self)
             
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
    
    def AngleAveragedFlux(self, z, E, **kwargs):
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
        zi = max(z, self.pop.zdead)
        if kw['zf'] is None:
            zf = self.pop.zform
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
        #    integrand = np.array(map(lambda zu: \
        #        self.AngleAveragedFluxSlice(z, E, zu,
        #        xavg=kw['xavg'], zxavg=kw['zxavg']), kw['zxavg'])) / Jc
        #else:
        #    # Assume neutral medium
        #    integrand = lambda zu: self.AngleAveragedFluxSlice(z, E, zu,
        #        h_2=lambda zz: 0.0) / Jc
    
        # Compute integral
        if type(integrand) == types.FunctionType:
            if self.pop.burst:
                raise ValueError('Burst needs correctness-check.')
                #flux = integrand(self.pop.zform)
            elif self._integrator == 'quad':
                flux = quad(integrand, zi, zf,
                    epsrel=self._rtol, epsabs=self._atol, limit=self._divmax)[0]
            elif self._integrator == 'romb':
                flux = romberg(integrand, zi, zf,
                    tol=self._atol, divmax=self._divmax)
            else:
                raise ValueError('Uncrecognized integrator \'%s\'' \
                    % self._integrator)
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
                raise ValueError('Uncrecognized integrator \'%s\'' \
                    % self._sampled_integrator)
    
        # Flux in units of photons s^-1 cm^-2 Hz^-1 sr^-1                                        
        flux *= Jc
    
        # Possibly convert to energy flux units
        if kw['energy_units']:
            flux *= E * erg_per_ev
    
        return flux
    
    def AngleAveragedFluxSlice(self, z, E, zp, **kwargs):
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
    
        kw = defkwargs.copy()
        kw.update(kwargs)
    
        if kw['xray_emissivity'] is None: # should include LyA too
            H = self.cosm.HubbleParameter(zp)
            E0 = self.igm.RestFrameEnergy(z, E, zp)
            epsilonhat = self.pop.NumberEmissivity(zp, E0)
            epsilonhat_over_H = epsilonhat / H
    
            if (E0 > self.pop.rs.Emax) or (E0 < self.pop.rs.Emin):
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
                tau = self.igm.OpticalDepth(z, zp, E, xavg=kw['xavg'],
                    zxavg=kw['zxavg'])
            else:
                tau = 0.0
        else:
            tau = self.igm.OpticalDepth(z, zp, E, xavg=kw['xavg'])
    
        return c * (1. + z)**2 * epsilonhat_over_H * np.exp(-tau) / four_pi
    
    def LWBackground(self, zmin, zmax):
        """
        Compute LW Background over all redshifts.
    
        Returns
        -------
        Redshifts, photon energies, and CXB fluxes, (z, E, flux).
        The flux array has shape (# redshift points, # frequency points).
    
        """
    
        raise NotImplemented('still thinking about this!')
    
        # Now, compute background flux
        cxrb = self.XrayFluxGenerator(self.igm.tau)
    
        fluxes = np.zeros([self.igm.L, self.igm.N])
    
        for i, flux in enumerate(cxrb):
            j = self.igm.L - i - 1  # since we're going from high-z to low-z
            fluxes[j,:] = flux.copy()
    
        return self.igm.z, self.igm.E, fluxes
    
    def LymanWernerFlux(self, z, E, **kwargs):
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
        Flux in units of s**-1 cm**-2 Hz**-1 sr**-1
    
        See Also
        --------
        AngleAveragedFluxSlice : the function we're integrating over.
    
        """
    
        kw = defkwargs.copy()
        kw.update(kwargs)
    
        # Closest Lyman line (from above)
        n = ceil(np.sqrt(E_LL / (E_LL - E)))
    
        if n > self.pf['nmax']:
            return 0.0
    
        En =  E_LL * (1. - 1. / n**2)
    
        # Corresponding zmax ("dark screen" as Z. Haiman likes to say)
        if kw['tau'] == 0.0:
            if kw['zf'] is None:
                zmax = self.pop.zform
            else:
                zmax = kw['zf']
        else:
            zmax = En * (1. + z) / E - 1.
    
        zmax = min(zmax, self.pop.zform)
    
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
    
    def LymanAlphaFlux(self, z, **kwargs):
        """
        Compute background flux at Lyman-alpha resonance. Includes products
        of Ly-n cascades if approx_lya=0.
        """
    
        if not self.pf['is_lya_src'] or (z > self.pop.zform):
            return 0.0
            
        if self.pf['Ja'] is not None:
            return self.pf['Ja'](z)    
    
        # Full calculation
        if self.pf['approx_lya'] == 0:
    
            J = 0.0
            for n in np.arange(2, self.pf['nmax']): 
    
                if n == 2 and not self.pf['lya_continuum']:
                    continue
                if n > 2 and not self.pf['lya_injected']:
                    continue
    
                En = self.hydr.ELyn(n)
                Enp1 = self.hydr.ELyn(n+1)
                Eeval = En + 0.01 * (Enp1 - En)
                Jn = self.hydr.frec(n) * self.LymanWernerFlux(z, Eeval, 
                    **kwargs)
    
                J += Jn
    
            return J
    
        # Flat spectrum, no injected photons, instantaneous emission only
        elif self.pf['approx_lya'] == 1:
            norm = self.pf['xi_alpha'] * c * self.cosm.dtdz(z) / four_pi
            return norm * (1. + z)**3 * \
                self.pop.LymanWernerPhotonLuminosityDensity(z) / dnu
        else:
            raise NotImplementedError('Haven\'t implemented approx_lya > 1!')
    
    def load_sed(self, prefix=None):
        fn = self.pop.rs.sed_name()
    
        if prefix is None:
            if not ARES:
                print "No $ARES environment variable."
                return None
    
            input_dirs = ['%s/input/seds' % ARES]
    
        else:
            if type(prefix) is str:
                input_dirs = [prefix]
            else:
                input_dirs = prefix
    
        guess = '%s/%s.txt' % (input_dirs[0], fn)
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
                    good_tab = '%s/%s' % (input_dir, fn1)    
    
                # If number of redshift bins and energy range right...
                if re.search(pre, fn1) and re.search(post, fn1):
                    good_tab = '%s/%s' % (input_dir, fn1)
                    break
    
        self.tabname = good_tab
        return good_tab
    
    def TabulateXrayEmissivity(self, zarr=None, **kwargs):
        """ 
        Compute emissivity of sources over redshift and frequency.
    
        Parameters
        ----------
        z : float
            observer redshift
    
        ===============
        relevant kwargs
        ===============
        zarr : np.ndarray
            array of redshifts corresponding to entries in xarr
    
        Returns
        -------
        Emissivity (in units of s**-1 cm**-3 Hz**-1 sr**-1) as function of observed photon 
        energy and redshift, z' > z. Actually, emissivity / HubbleParameter.
    
        shape = (z, E)
    
        """
    
        if zarr is None:
            Nz = self.igm.L
            zarr = self.igm.z
        else:
            Nz = len(zarr)
    
        found = False
    
        # SED table supplied by-hand
        if self.pf['spectrum_table'] is not None:
            if os.path.exists(self.pf['spectrum_table']):
                found = True
                if re.search('.hdf5', self.pf['spectrum_table']) or \
                   re.search('.h5', self.pf['spectrum_table']):
                    f = h5py.File(self.pf['spectrum_table'], 'r')
                    E = f['E'].value
                    LE = f['LE'].value
                    f.close()
                    if rank == 0 and self.pf['verbose']:
                        print "Loaded %s." % self.pf['spectrum_table']
                else:
                    E, LE = np.loadtxt(self.pf['spectrum_table'], unpack=True)
    
                if self.pf['verbose'] and rank == 0:
                    print "Loaded %s." % self.pf['spectrum_table']
    
            else:
                if rank == 0 and self.pf['verbose'] \
                    and self.pf['spectrum_table'] is not None:
                    print "%s does not exist.\nCreating now..." % self.pf['spectrum_table']
    
        # Auto-find SED table
        elif self.pf['load_sed']:
            tab = self.load_sed(self.pf['sed_prefix'])
    
            if tab is not None:
                E, LE = np.loadtxt(tab, unpack=True)
                found = True
    
                if self.pf['verbose'] and rank == 0:
                    print "Loaded %s." % tab
                    self.pf['spectrum_table'] = tab
    
            else:
                if rank == 0 and self.pf['verbose']:
                    print "No SED table found. Creating SED now..."
    
        if found:    
            # Fix energy sampling if need be    
            if E.size != self.igm.E.size or (not np.allclose(E, self.igm.E)):    
                if rank == 0:
                    print 'Photon energies in SED table do not match specifications in parameter file.'
                    print 'Interpolating to required values...'
    
                func = interp1d(np.log10(E), np.log10(LE), kind='linear', 
                    fill_value=-np.inf, bounds_error=False)
                self.Inu = 10**np.array(map(func, self.igm.logE))
    
            else:
                self.Inu = LE        
    
            # Apply a power-law cutoff
            if np.isfinite(self.pf['spectrum_logN']):
                print "Applying power-law cutoff to input SED."
                cutoff = lambda nrg: np.exp(-10.**self.pf['spectrum_logN'] \
                    * (self.igm.sigma(nrg, 0) + self.pf['approx_helium'] \
                    * self.cosm.y * self.igm.sigma(nrg, 1)))
    
                self.Inu *= np.array(map(cutoff, self.igm.E))
    
        # Generate table from scratch
        if not found:
            # Assumes spectral shape doesn't change with time
    
            Inu = np.zeros(self.igm.N)
            for i in range(self.igm.N): 
                Inu[i] = self.pop.rs.Spectrum(self.igm.E[i])
    
            self.Inu = Inu
    
            if self.pf['spectrum_table'] is not None and rank == 0:
                f = h5py.File(self.pf['spectrum_table'], 'w')
                f.create_dataset('E', data=self.igm.E)
                f.create_dataset('LE', data=self.Inu)
                f.close()
                if rank == 0 and self.pf['verbose']:
                    print "Wrote %s." % self.pf['spectrum_table']
    
        self.Inu_hat = self.Inu / self.igm.E
    
        # Now, redshift dependent parts    
        epsilon = np.zeros([Nz, self.igm.N])                     
        for ll in xrange(Nz):
            H = self.cosm.HubbleParameter(zarr[ll])                 
            Lbol = self.pop.XrayLuminosityDensity(zarr[ll])                   
            epsilon[ll,:] = self.Inu_hat * Lbol * ev_per_hz / H / erg_per_ev                
    
        self._xray_emissivity = epsilon
        return epsilon
    
    def XrayBackground(self):
        """
        Compute Cosmic X-ray Background over all redshifts.
    
        Returns
        -------
        Redshifts, photon energies, and CXB fluxes, (z, E, flux).
        The flux array has shape (# redshift points, # frequency points).
    
        """
    
        # Now, compute background flux
        cxrb = self.XrayFluxGenerator(self.igm.tau)
    
        fluxes = np.zeros([self.igm.L, self.igm.N])
        for i, flux in enumerate(cxrb):
            j = self.igm.L - i - 1  # since we're going from high-z to low-z
            fluxes[j,:] = flux.copy()
    
        return self.igm.z, self.igm.E, fluxes
    
    def XrayFluxGenerator(self, tau=None, emissivity=None, flux0=None):
        """ 
        Compute X-ray background flux in a memory efficient way.
    
        Parameters
        ----------
        tau : np.ndarray
            2-D optical depth, dimensions (self.L, self.N)
        emissivity : np.ndarray
            2-D, dimensions (self.L, self.N)
        flux0 : np.ndarray
            1-D array of fluxes, size is self.N
    
        Notes
        -----
        1. We only tabulate the emissivity in log-x formalism. 
        2. This cannot be parallelized.
          -I suppose we could parallelize over frequency but not redshift,
           but then fluxes would have to be communicated on every redshift
           step. Probably not worth it.
    
        Returns
        -------
        Generator for the flux, each in units of s**-1 cm**-2 Hz**-1 sr**-1
    
        """
    
        if self.pf['redshift_bins'] is None and self.pf['tau_table'] is None:
            raise ValueError('This method only works if redshift_bins != None.')
        
        if emissivity is None:
            emissivity_over_H = self.TabulateXrayEmissivity()
        else:
            emissivity_over_H = emissivity
    
        if tau is None:
            tau = self.igm.tau
            
        optically_thin = False
        if np.all(tau == 0):
            optically_thin = True
    
        otf = False
        if tau.shape == self.igm.E.shape:
            otf = True
    
        if flux0 is None:    
            flux = np.zeros_like(self.igm.E)
        else:
            flux = flux0.copy()
    
        ll = self.igm.L - 1
        self.tau = tau
    
        # Loop over redshift - this is the generator                    
        z = self.igm.z[-1]
        while z >= self.igm.z[0]:
            
            # First iteration: no time for there to be flux yet
            # (will use argument flux0 if the EoR just started)
            if ll == (self.igm.L - 1):
                pass
    
            # General case
            else:
                    
                if otf:
                    exp_term = np.exp(-np.roll(tau, -1))
                else:   
                    exp_term = np.exp(-np.roll(tau[ll], -1))
    
                trapz_base = 0.5 * (self.igm.z[ll+1] - self.igm.z[ll])

                # First term in Eq. 25 of Mirocha (2014)
                #fnm1 = np.roll(emissivity_over_H[ll+1], -1, axis=-1)                
                #fnm1 *= exp_term     
                #fnm1 += emissivity_over_H[ll]
                #fnm1 *= trapz_base
                #
                ## Second term in Eq. 25 of Mirocha (2014)       
                #flux = np.roll(flux, -1) * exp_term / self.igm.Rsq
                #
                ## Add two terms together to get final flux
                #flux += fnm1 * c * self.igm.x[ll]**2 / four_pi
    
                # Less readable version, but faster!
                # Equivalent to Eq. 25 in Mirocha (2014)
                flux = (c / four_pi) \
                    * ((self.igm.xsq[ll+1] * trapz_base) \
                    * emissivity_over_H[ll]) \
                    + exp_term * ((c / four_pi) * self.igm.xsq[ll+1] \
                    * trapz_base * np.roll(emissivity_over_H[ll+1], -1, axis=-1) \
                    + np.roll(flux, -1) / self.igm.Rsq)
                
            # No higher energies for photons to redshift from.
            # An alternative would be to extrapolate, and thus mimic a
            # background spectrum that is not truncated at Emax
            flux[-1] = 0.0
    
            yield flux
    
            # Increment redshift
            ll -= 1
            z = self.igm.z[ll]
    
            if ll == -1:
                break
    
    def TabulateLWFlux(self, z, zarr):
        """
        Compute Lyman-series background flux as a function of energy.
    
        Parameters
        ----------
        z : float
            observer redshift
    
        ===============
        relevant kwargs
        ===============
    
        Returns
        -------
        Flux (in units of photon number) as function of observed photon energy,
        integrated over emission from all redshifts z' > z.
    
        """
    
        emiss = self.tabulate_lw_emissivity(z, zarr)    
    
        flux = map(lambda i: self.AngleAveragedFlux(z, self.E[i], zxavg=zarr, 
            xavg=xarr, tau=tau[i], emissivity=emiss[i], **kwargs), self.Ei)
    
        self._xray_flux = np.array(flux)            
        return self._xray_flux.copy()
             