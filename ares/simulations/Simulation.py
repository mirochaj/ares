import os
import copy
import pickle
import numpy as np
from types import FunctionType
from ..util import ProgressBar
from ..util import ParameterFile
from ..util.Stats import bin_c2e
from .Global21cm import Global21cm
from ..util.Misc import get_wave_or_equivalent
from .PowerSpectrum21cm import PowerSpectrum21cm
from ..physics.Constants import cm_per_mpc, c, s_per_yr, erg_per_ev, \
    erg_per_s_per_nW, h_p, cm_per_m, sqdeg_per_std

class Simulation(object):
    def __init__(self, pf=None, pf_updates=None, **kwargs):
        """ Wrapper class designed to facilitate easy runs of any simulation. """

        if pf is None:
            assert kwargs is not None, \
                "Must provide parameters to initialize a Simulation!"

        self.kwargs = kwargs

        if pf is None:
            self.pf = ParameterFile(is_sim_level=True, **kwargs)
        else:
            self.pf = pf

            ##
            # This is a sneaky way to not have to re-initialize an
            # entire ParameterFile if we're running lots of models.
            # Just need to remember that `self.pf` does not contain
            # population-specific parameters, so much parse parameter
            # names here and inject into correct element of self.pf.pfs,
            # i.e., the individual ParameterFile for each population.
            if pf_updates is not None:
                for par in pf_updates:
                    ib = par.find('{')
                    if ib == -1:
                        if self.pf['verbose']:
                            print(f"# Ignoring par={par} in updates.")
                        continue

                    popid = int(par[ib+1:ib+2])
                    newname = par.strip(f"{{{popid}}}")
                    self.pf.pfs[popid][newname] = pf_updates[par]

    @property
    def sim_gs(self):
        if not hasattr(self, '_sim_gs'):
            self._sim_gs = Global21cm(pf=self.pf, **self.kwargs)
        return self._sim_gs

    @sim_gs.setter
    def sim_gs(self, value):
        """ Set global 21cm instance by hand. """
        self._sim_gs = value

    @property
    def sim_ps(self):
        if not hasattr(self, '_sim_ps'):
            self._sim_ps = PowerSpectrum21cm(pf=self.pf, **self.kwargs)
            self._sim_ps.gs = self.sim_gs
        return self._sim_ps

    #@ps.setter
    #def ps(self, value):
    #    """ Set power spectrum 21cm instance by hand. """
    #    self._ps = value

    @property
    def history(self):
        if not hasattr(self, '_history'):
            self._history = {}
        return self._history

    @property
    def mean_intensity(self):
        if not hasattr(self, '_mean_intensity'):
            self._mean_intensity = self.sim_gs.medium.field
        return self._mean_intensity

    @property
    def background_intensity(self):
        return self.mean_intensity

    def _cache_ebl(self, wave_units='mic', flux_units='SI', zlow=None, 
        compute_via_counts=False):
        if not hasattr(self, '_cache_ebl_'):
            self._cache_ebl_ = {}

        # Could be clever and convert units here.
        if (wave_units, flux_units, zlow, compute_via_counts) in self._cache_ebl_:
            _data = self._cache_ebl_[(wave_units, flux_units, zlow, compute_via_counts)]
            return _data

        return None

    def get_ebl_intensity(self, wave_units='mic', flux_units='SI', pops=None,
        zlow=None, bands=None, magbins=None, compute_via_counts=False, **kwargs):
        """
        Return the extragalactic background light (EBL) over all wavelengths.

        Parameters
        ----------
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'
        pops : list
            If supplied, should be a list of populations to be included, i.e.,
            their (integer) ID numbers (see `self.pops` attribute for list
            of objects).
        zlow : int, float
            If provided, will truncate integral over redshift so that the EBL
            includes only emission from sources at z >= zlow.
        bands : np.ndarray
            If provided, a 2-D array defining a series of band edges (in 
            microns). In this case, rather than integrating RTE to obtain 
            mean EBL intensity, we will first generate galaxy number counts 
            in these `bands`, and subsequently integrate to obtain the mean
            EBL intensity. This is a useful cross-check and should yield 
            consistent results with the `bands=None` solution.
        magbins : np.ndarray 
            If `bands` is not None, also need to decide on magnitude bins.
            These are bin centers in *apparent* AB mags.


        .. note :: 'SI' units means nW / m^2 / sr, 'cgs' means erg/s/Hz/sr.

        Returns
        -------
        Dictionary containing EBL for each source population, with the ID
        number used as a dictionary key. Each element is a tuple containing
        the (observed energies (or wavelengths) in `wave_units`,
        observed fluxes in `flux_units`).

        """

        cached_result = self._cache_ebl(wave_units, flux_units, zlow, compute_via_counts)
        if cached_result is not None:
            data = cached_result
        else:
            data = {}

        if (not self.background_intensity._run_complete) and (not compute_via_counts):
            self.background_intensity.run()

        for i in range(len(self.pops)):
            if i in data:
                continue

            if pops is not None:
                if i not in pops:
                    continue

            if zlow is not None:
                zf = zlow
            else:
                zf = self.pops[i].zdead
            
            if self.pops[i].pf['pop_mask'] is not None:
                print(f"! WARNING: pop_mask != None, non-standard for mean EBL runs!")
            #assert self.pops[i].pf['pop_mask'] is None, \
            #    "Turn off mask (via `pop_mask`) before computing mean EBL!"
            
            if compute_via_counts:
                assert bands is not None, "Must provide `bands`."
                assert bands.ndim == 2, "Must provide `bands` as 2-D array of band edges."
                assert magbins is not None, "Must provide `magbins`."

                magbins_e = bin_c2e(magbins)

                fbins = 10**((magbins + 48.60) / -2.5)
                
                ##
                # Loop over bands, integrate galaxy counts
                x = np.mean(bands, axis=1)
                flux = np.zeros(bands.shape[0])

                pb = ProgressBar(x.size, name=f'ebl(pop={i})', use=self.pf['progress_bar'])
                pb.start()
                for j, band in enumerate(bands):
                    pb.update(j)

                    nu = c / (np.mean(band) * 1e-4)
                    
                    num = self.get_galaxy_number_counts(band, magbins, popids=i,
                        **kwargs)
                    
                    # Cumulative flux [convert to nW m^-2 sr^-1 Hz^-1]
                    tot_Jy = np.trapezoid(num[i] * fbins, x=magbins) / 1e-23
                    
                    flux[j] = tot_Jy * 1e-23 * nu * (1e2)**2 \
                        * sqdeg_per_std / erg_per_s_per_nW
                
                pb.finish()

                # In this case, x and flux are always in ascending wavelength

            else:
                _x, _flux = self.mean_intensity.get_spectrum(zf=zf, popids=i,
                    units=flux_units, xunits=wave_units)
                
                # Need to flip
                _x = _x[-1::-1]
                _flux = _flux[-1::-1]

                if bands is None:
                    x = _x
                    flux = _flux
                else:
                    x = bands.mean(axis=1)
                    flux = np.interp(x, _x, _flux)

            data[i] = x, flux

        # Cache
        # Can't cache: compute_via_counts may have provided zmin, zmax
        #self._cache_ebl_[(wave_units, flux_units, zlow, compute_via_counts)] = data

        return data

    def get_ebl_ps(self, scales, waves, 
        masking_criteria=None, waves2=None, 
        wave_units='mic', wave_units2=None,
        flux_units='SI', flux_units2=None, pops=None,
        include_inter_pop=True, cache_ipop_mtx=None, **kwargs):
        """
        Compute power spectrum of EBL at some observed wavelength(s).

        Parameters
        ----------
        scales : int, float, np.ndarray
            Ell modes of interest.
        waves : int, float, np.ndarray
            Wavelengths at which to compute power spectra in `wave_units`.
            Note that if 2-D, must have shape (number of bins, 2), in which
            case the power spectra will be computed in series of bandpasses.
        waves2 : int, float, np.ndarray

        pops : list, tuple
            If provided, sets the ID numbers of populations that will be
            included in the model. In other words, any population *not* included
            in this list will be skipped. By default, this is None and all
            source populations defined by the parameters (self.pf) are
            included.
        include_inter_pop : bool
            This flag determines whether "inter-population cross terms" are
            included in the calculation.
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'

        Optional keyword arguments
        --------------------------
        The `get_ps_obs` methods within ares.populations objects take a
        number of optional arguments that control the output. These include:

        include_1h : bool
            If False, exclude 1-halo term from calculation [Default: True]
        include_2h : bool
            If False, exclude 2-halo term from calculation [Default: True]
        include_shot : bool
            If False, exclude shot noise term from calculation [Default: True]


        Returns
        -------
        Tuple containing (scales, 2 pi / scales or l*l(+1),
            waves, power spectra).

        Note that the power spectra are return as 2-D arrays with shape
        (len(scales), len(waves))

        """

        # Make sure we do mean background first in case LW feedback is on.
        #if not self.mean_intensity._run_complete:
        #    self.mean_intensity.run()

        # Make sure things are arrays
        if type(scales) != np.ndarray:
            scales = np.array([scales])
        if type(waves) != np.ndarray:
            waves = np.array([waves])

        if waves.ndim == 2:
            assert waves.shape[1] == 2, \
                "If `waves` is 2-D, must have shape (num waves, 2)."

        # Eventually might modify for cross-correlations
        # Could keep flux_units2 to correspond to waves2 or something.
        if flux_units.lower() == 'si':
            to_ps_units = cm_per_m**2 / erg_per_s_per_nW
        elif flux_units.lower() == 'mjy':
            to_ps_units = 1e17
        elif flux_units.lower() == 'cgs':
            to_ps_units = 1
        else:
            raise NotImplemented('help')
        
        if flux_units2 is None:
            flux_units2 = flux_units
        
        if flux_units2.lower() == 'si':
            to_ps_units2 = cm_per_m**2 / erg_per_s_per_nW
        elif flux_units.lower() == 'mjy':
            to_ps_units2 = 1e17
        elif flux_units2.lower() == 'cgs':
            to_ps_units2 = 1
        else:
            raise NotImplemented('help')
        

        #if wave_units.lower().startswith('mic'):
        #    pass
#
        #else:
        #    raise NotImplemented('help')

        # Do some error-handling if waves is 2-D: means the user provided
        # bandpasses instead of a set of wavelengths.

        # If waves2 is None, it means we're doing autos.
        # In general, for each channel in `waves`, with index `k`,
        # we'll cross-correlate with the k'th element of waves2.
        # In principle 
        if waves2 is None:
            waves2 = waves
            is_autos = True
        else:
            is_autos = False

        if wave_units2 is None:
            wave_units2 = wave_units

        # Convert input units to microns, native unit for all fluctuation calculations
        xmic = get_wave_or_equivalent(waves, wave_units, 'mic')
        xmic2 = get_wave_or_equivalent(waves2, wave_units2, 'mic')

        ps = np.zeros((len(self.pops), len(scales), len(waves)))
        px = np.zeros((len(self.pops), len(self.pops), len(scales), len(waves)))
        # Save contributing pieces

        # [optonal] Save redshift chunks
        ps_z = np.zeros((len(self.pops), len(self.pops),
            len(scales), len(waves), self.pops[0].halos.tab_z.size))
        
        ##
        # Need to determine fraction of halos that are selected
        if masking_criteria is None:
            print(f"! WARNING: did you mean not to provide a mask?")
            fmask = [np.zeros(len(self.pops))] * len(waves)
        else:
            fmask = []
            for mask in masking_criteria:
                fmask.append(self.get_galaxy_subsample(mask))

        ##
        # Loop over source populations and compute power spectrum.
        for i, pop in enumerate(self.pops):

            # Honor user-supplied list of populations to include
            if pops is not None:
                if i not in pops:
                    continue

            for j, popx in enumerate(self.pops):
                # Avoid double counting
                # Convention here only populates lower half of 
                # inter-population cross-correlation matrix
                # Note that we only do this for autos because for
                # crosses, we need to make sure each channel hits
                # each population.
                if is_autos and (j > i):
                    break

                # Honor user-supplied list of populations to include
                if pops is not None:
                    if j not in pops:
                        continue
                
                # Try to load from cache [optional]
                if (cache_ipop_mtx is not None) and include_inter_pop:
                    _px, _pz = cache_ipop_mtx
                    _npops = _px.shape[0]
                    # If we're covered by the cache, use it
                    if i < _npops:
                        # Assumes cache_ipop_mtx is in 
                        # same units as requested here!
                        # Could add check later.
                        px[i,j,:,:] = _px[i,j,:,:] / to_ps_units
                        ps_z[i,j,:,:,:] = _pz[i,j,:,:,:] / to_ps_units
                        continue

                for k, wave in enumerate(waves):
                    # Will default to 1h + 2h + shot
                    if j == i:
                        px[i,j,:,k] = pop.get_ps_obs(scales,
                            wave_obs1=xmic[k], wave_obs2=xmic2[k],
                            fsel1=1-fmask[k][i,:,:],
                            **kwargs)
                        ps[i,:,k] = px[i,j,:,k]
                        ps_z[i,i,:,k,:] = pop._ps_obs_integrand.copy()
                        continue

                    if not include_inter_pop:
                        continue

                    ##
                    # Cross terms only from here on
                    px[i,j,:,k] = pop.get_ps_obs(scales,
                        wave_obs1=xmic[k], wave_obs2=xmic2[k],
                        fsel1=1-fmask[k][i], fsel2=1-fmask[k][j],
                        pop2=popx if i != j else None, **kwargs)
                    # Setting pop2 to None if i == j avoids recomputing
                    # the luminosity etc. inside other get_ps_* functions
                    ps_z[i,j,:,k,:] = pop._ps_obs_integrand.copy()

                ##
                # Clear out some memory -- u(k|M) tabs can be big.
                #if hasattr(pop.halos, '_tab_u_nfw'):
                #    del pop.halos._tab_u_nfw

        ##
        # Increment `ps` with cross terms.
        # Convention is that fluctuations for population `i` includes
        # all crosses with

        ##
        # Modify PS units before return
        px *= to_ps_units * to_ps_units2 / cm_per_mpc**4
        ps_z *= to_ps_units * to_ps_units2 / cm_per_mpc**4
        
        # Sum over source populations
        ptot = px.sum(axis=0).sum(axis=0)

        if pops is None:
            hist = self.history # poke
            self._history['ps_nirb'] = ptot

        self.ps_auto = ps
        self.ps_by_pop = px
        self.ps_by_z = ps_z

        return ptot
    
    def get_galaxy_subsample(self, selection_criteria, pops=None,
        return_fraction=True):
        """
        Subject model galaxies to cuts in redshift, magnitude, and/or color.

        .. note :: This is used both for masking and for sample selection.

        Parameters
        ----------
        selection_criteria : dict

        Returns
        -------
        A 3-D array with dimensions corresponding to (population, z, Mh).
        Each element is the fraction of halos that satisfy the selection criteria.

        """

        f_sel = np.zeros((len(self.pops), self.halos.tab_z.size, self.halos.tab_M.size))

        for i, pop in enumerate(self.pops):
            if pops is not None:
                if i not in pops:
                    continue

            f_sel[i,:,:] = pop.get_galaxy_subsample(selection_criteria, 
                return_fraction=return_fraction, logic='and')
        
        return f_sel
    
    def get_ebl_x_galaxies(self, scales, waves, zbins, 
        selection_criteria, masking_criteria,
        wave_units='mic', flux_units='SI', pops=None,
        include_inter_pop=True, **kwargs):
        """
        Compute cross spectrum between EBL and galaxy population.

        Parameters
        ----------
        scales : int, float, np.ndarray
            Ell modes of interest.
        waves : int, float, np.ndarray
            Wavelengths at which to compute power spectra in `wave_units`.
            Note that if 2-D, must have shape (number of bins, 2), in which
            case the power spectra will be computed in series of bandpasses.
        galaxy_prop : dict
            A dictionary defining the magnitude and/or color and/or redshift
            cuts used to select galaxies. At the moment, this is just a
            magnitude cut provided as galaxy_prop={'mag': (cam, filter, cut)}
        pops : list, tuple
            If provided, sets the ID numbers of populations that will be
            included in the model. In other words, any population *not* included
            in this list will be skipped. By default, this is None and all
            source populations defined by the parameters (self.pf) are
            included.
        include_inter_pop : bool
            This flag determines whether "inter-population cross terms" are
            included in the calculation.
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'

        Returns
        -------
        Tuple containing (scales, 2 pi / scales or l*l(+1),
            waves, power spectra).

        Note that the power spectra are returned as 3-D arrays with shape
        (number of populations, number of ell modes, number of wavelengths).

        """

        # Make sure things are arrays
        if type(scales) != np.ndarray:
            scales = np.array([scales])
        if type(waves) != np.ndarray:
            waves = np.array([waves])
        if type(zbins) != np.ndarray:
            zbins = np.array(zbins)
        
        assert zbins.ndim == 2

        # Do some error-handling if waves is 2-D: means the user provided
        # bandpasses instead of a set of wavelengths.
        if waves.ndim == 2:
            assert waves.shape[1] == 2, \
                "If `waves` is 2-D, must have shape (num waves, 2)."

        if wave_units.lower().startswith('mic'):
            pass
        else:
            raise NotImplemented('help')

        ps = np.zeros((len(scales), len(waves), len(zbins)))
        
        # Save contributing pieces

        # [optonal] Save redshift chunks
        zarr = self.pops[0].halos.tab_z
        Hofz = np.array([self.cosm.HubbleParameter(z) for z in zarr])
        ps_z = np.zeros((len(self.pops), len(self.pops),
            len(scales), len(waves), len(zbins), zarr.size))

        
        # Loop over source populations and compute cross spectrum.
        
        ##
        # Need to determine fraction of halos that are masked
        if masking_criteria is None:
            print(f"! WARNING: did you mean not to provide a mask?")
            fmask = [np.zeros(len(self.pops))] * len(waves)
        else:
            fmask = []
            for mask in masking_criteria:
                fmask.append(self.get_galaxy_subsample(mask, pops=pops))

        # Get full z-dependent number density
        num_pz = np.zeros((len(self.pops), len(zarr)))
        fsel_allz = self.get_galaxy_subsample(selection_criteria, pops=pops)
        
        for i, pop in enumerate(self.pops):
            if pops is not None:
                if i not in pops:
                    continue
            num_pz[i,:] = self.pops[i].get_num_from_fsel(fsel_allz[i])

        ##
        # Now get Limber integrand
        num_p = np.zeros((len(self.pops), len(zbins)))
        for h, zbin in enumerate(zbins):

            galaxy_prop = {'z': zbin}
            galaxy_prop.update(selection_criteria)
            
            ##
            # Need to determine fraction of halos that are selected 
            fsel = self.get_galaxy_subsample(galaxy_prop, pops=pops)

            if np.all(fsel == 0):
                print(f"! No galaxies found satisfying selection!")
                print(f"! z={zbin}, selection:", selection_criteria)
                continue
            
            for i, pop in enumerate(self.pops):

                num_p[i,h] = self.pops[i].get_num_from_fsel(fsel[i], 
                    zbin=zbin)

                # Honor user-supplied list of populations to include
                if pops is not None:
                    if i not in pops:
                        continue

                for j, popx in enumerate(self.pops):

                    # Honor user-supplied list of populations to include
                    if pops is not None:
                        if j not in pops:
                            continue

                    for k, wave in enumerate(waves):
                        fsel1 = fsel[i]
                        fsel2 = 1 - fmask[k][j]
                    
                        # (pops, pops, scales, waves, zbin, zall)
                        ps_z[i,j,:,k,h,:] = pop.get_xs_obs(scales,
                            wave_obs=wave, zg=zbin, 
                            isnum1=1, isnum2=0,
                            idnum1=i, idnum2=j,
                            fsel1=fsel1, fsel2=fsel2,
                            pop2=popx, do_limber=False, **kwargs)
                        
            ##
            # We do the Limber integral here for crosses
            W_g = num_pz.sum(axis=0) / num_p[:,h].sum(axis=0)**2
            for k, wave in enumerate(waves):
                limber_integ = W_g[None,:] \
                    * ps_z.sum(axis=0).sum(axis=0)[:,k,h,:] \
                    / ((c / cm_per_mpc) / Hofz)
                ps[:,k,h] = np.trapezoid(limber_integ, x=zarr, axis=-1)

            print(f"! Done: shot power in 0th channel is {ps[-1,0,:]}")
        ##
        # Modify PS units before return
        if flux_units.lower() == 'si':
            ps *= cm_per_m**2 / erg_per_s_per_nW / cm_per_mpc**2
        else:
            raise NotImplemented()

        self.num_by_pop = num_p
        self.num_by_pop_z = num_pz

        #if pops is None:
        #    hist = self.history # poke
        #    self._history['ps_nirb_x_gal'] = scales, scales_inv, waves, ps

        return ps
    
    def get_galaxy_ps(self, scales, zbins, selection_criteria, 
        wave_units='mic', flux_units='SI', pops=None,
        include_inter_pop=True, **kwargs):
        """
        Compute auto spectrum of galaxies.

        Parameters
        ----------
        scales : int, float, np.ndarray
            Ell modes of interest.
        galaxy_prop : dict
            A dictionary defining the magnitude and/or color and/or redshift
            cuts used to select galaxies. At the moment, this is just a
            magnitude cut provided as galaxy_prop={'mag': (cam, filter, cut)}
        pops : list, tuple
            If provided, sets the ID numbers of populations that will be
            included in the model. In other words, any population *not* included
            in this list will be skipped. By default, this is None and all
            source populations defined by the parameters (self.pf) are
            included.
        include_inter_pop : bool
            This flag determines whether "inter-population cross terms" are
            included in the calculation.
        wave_units : str
            Current options: 'eV', 'microns', 'Ang'
        flux_units : str
            Current options: 'cgs', 'SI'

        Returns
        -------
        Tuple containing (scales, 2 pi / scales or l*l(+1),
            waves, power spectra).

        Note that the power spectra are returned as 3-D arrays with shape
        (number of populations, number of ell modes, number of wavelengths).

        """

        # Make sure things are arrays
        if type(scales) != np.ndarray:
            scales = np.array([scales])

        if type(zbins) != np.ndarray:
            zbins = np.array(zbins)
        
        assert zbins.ndim == 2


        ps = np.zeros((len(self.pops), len(self.pops), len(scales), len(zbins)))
        
        # Save contributing pieces

        # [optonal] Save redshift chunks
        #ps_z = np.zeros((len(self.pops), len(self.pops),
        #    len(scales), len(waves), self.pops[0].halos.tab_z.size))

        
        # Loop over source populations and compute cross spectrum.
        num_p = np.zeros((len(self.pops), len(zbins)))
        for h, zbin in enumerate(zbins):

            galaxy_prop = {'z': zbin}
            galaxy_prop.update(selection_criteria)
            
            ##
            # Need to determine fraction of halos that are selected 
            fsel = self.get_galaxy_subsample(galaxy_prop, pops=pops)

            if np.all(fsel == 0):
                print(f"No galaxies found satisfying selection!")
                print(f"z={zbin}", selection_criteria)
                continue

            for i, pop in enumerate(self.pops):

                num_p[i,h] = self.pops[i].get_num_from_fsel(fsel[i], zbin=zbin)
    
                # Honor user-supplied list of populations to include
                if pops is not None:
                    if i not in pops:
                        continue

                for j, popx in enumerate(self.pops):
                    # Avoid double counting.
                    if (j > i):
                        break
                    
                    # Honor user-supplied list of populations to include
                    if pops is not None:
                        if j not in pops:
                            continue
                    
                    ps[i,j,:,h] = pop.get_xs_obs(scales,
                        wave_obs=None, zg=zbin, 
                        field1_is_num=1, field2_is_num=1,
                        fsel1=fsel[i,:,:], fsel2=fsel[j,:,:],
                        pop2=popx, **kwargs)
                    
            ##
            # Done with this redshift bin
            ps[:,:,:,h] /= num_p[:,h].sum(axis=0)**2
        ##
        # Modify PS units before return
        

        #if pops is None:
        #    hist = self.history # poke
        #    self._history['ps_nirb_x_gal'] = scales, scales_inv, waves, ps

        return ps.sum(axis=0).sum(axis=0)
    
    def get_galaxy_number_counts(self, band, magbins, popids=None,
        dlam=10, zmin=None, zmax=None, zbin=0.01):
        """
        Compute the number of galaxies per square degree for each source populations.

        Parameters
        ----------
        band : tuple
            Band edges in microns.
        magbins : np.ndarray
            Array of magnitude bin *centers*.
        """

        # Put band in terms that internal routines understand
        x = np.mean(band) * 1e4
        dx = (band[1] - band[0]) * 1e4
        
        assert dx > 3 * dlam

        if popids is not None:
            if type(popids) not in [list, tuple]:
                popids = [popids]
        
        # Loop over populations and save results for each one separately
        num_by_pop = {}
        for i, pop in enumerate(self.pops):

            if popids is not None:
                if i not in popids:
                    continue

            # Check zmin, zmax values 
            if zmin is None:
                _zmin = max(pop.zdead, pop.halos.tab_z.min())
            else:
                _zmin = zmin
            
            # Check zmin, zmax values 
            if zmax is None:
                _zmax = min(pop.zform, pop.halos.tab_z.max())
            else:
                _zmax = zmax

            # Farm out the real work to the `pop` object.
            num_pop = pop.get_number_counts(magbins, 
                x=x, units='Angstroms', window=dx, dlam=dlam, 
                zmin=_zmin, zmax=_zmax, zbin=zbin)
            
            num_by_pop[i] = num_pop.copy()
        
        return num_by_pop

    @property
    def pops(self):
        return self.sim_gs.medium.field.pops

    #@property
    #def pops(self):
    #    if not hasattr(self, '_pops'):
    #        self._pops = CompositePopulation(pf=self.pf, cosm=self.cosm,
    #            **self._kwargs).pops

    #    return self._pops

    @property
    def grid(self):
        return self.sim_gs.medium.field.grid

    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            if hasattr(self.grid, 'hydr'):
                self._hydr = self.grid.hydr
            else:
                self._hydr = Hydrogen(pf=self.pf, cosm=self.cosm, **self.pf)
        return self._hydr

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            if hasattr(self.grid, 'cosm'):
                self._cosm = self.grid.cosm
            else:
                self._cosm = Cosmology(pf=self.pf, **self.pf)

        return self._cosm

    @property
    def halos(self):
        if not hasattr(self, '_halos'):
            self._halos = self.pops[0].halos
        return self._halos

    def run(self):
        """
        Run everything we can.
        """
        pass

    def run_ebl(self):
        hist = self.mean_intensity.run()
        if not self.mean_intensity._run_complete:
            self.mean_intensity.run()

    def get_21cm_gs(self):
        if '21cm_gs' not in self.history:
            self.sim_gs.run()
            self.history['21cm_gs'] = self.sim_gs.history

        return self.sim_gs

    def get_21cm_ps(self, z=None, k=None):
        if '21cm_ps' not in self.history:
            # Allow user to specify (z, k) if they want
            self.sim_ps.run(z=z, k=k)#(z, k)
            self.history['21cm_ps'] = self.sim_ps.history

        return self.sim_ps
    
    def get_bias(self, z, limit, wave=1600., cut_in_mass=False, absolute=False,
        cut_in_flux=False, intensity_weight=False):
        """
        Compute linear bias of galaxies brighter than (or more massive than)
        some cut-off.

        Parameters
        ----------
        z : int, float
            Redshift of interest.
        limit : int, float
            This parameter controls either the limiting magnitude or the
            limiting halo mass, depending on the value of `cut_in_mass`.
            By default, our approach is to use apparent magnitudes in order to
            connect to observations more explicitly. For example, `limit=26.5`
            is a Roman-like magnitude cut on the galaxy population.
        cut_in_mass : bool
            If True, then `limit` is assumed to be a halo mass in Msun.
        absolute : bool
            Whether `limit` magnitudes are absolute or apparent AB mags.
        cut_in_flux : bool
            Not currently implement. Might be useful for comparing with
            specroscopic surveys which often report sensitivities as a
            limiting line luminosity in [erg/s/cm^2].

        Returns
        -------

        """


        iz = np.argmin(np.abs(z - self.halos.tab_z))

        tab_M = self.halos.tab_M
        tab_b = self.halos.tab_bias[iz,:]
        tab_n = self.halos.tab_dndm[iz,:]
        
        top = np.zeros_like(self.halos.tab_M)
        bot = np.zeros_like(self.halos.tab_M)
        for i, pop in enumerate(self.pops):

            
            tab_f = pop.tab_focc[iz,:]

            if cut_in_flux:
                raise NotImplemented('help')
            elif cut_in_mass:
                if type(limit) in [list, tuple, np.ndarray]:
                    lo, hi = limit
                    ok = np.logical_and(tab_M >= lo, tab_M < hi)
                else:
                    ok = tab_M >= limit
            else:
                _filt, mags = pop.get_mags(z, x=wave, absolute=absolute)
                ok = np.logical_and(mags <= limit, np.isfinite(mags))
    
            # Can add weighting by luminosity
            if intensity_weight:
                L = pop.get_lum(z, x=wave, units_out='erg/s/Hz', total_sat=1)
            else:
                L = np.ones_like(tab_M)

            focc = tab_f if pop.is_central_pop else np.ones_like(tab_M)
    
            integ_top = tab_b[ok==1] * tab_n[ok==1] * focc[ok==1] * L[ok==1]
            integ_bot = tab_n[ok==1] * focc[ok==1] * L[ok==1]
    
            top[ok==1] += integ_top
            bot[ok==1] += integ_bot

        b = np.trapezoid(top[ok==1] * tab_M[ok==1], x=np.log(tab_M[ok==1])) \
          / np.trapezoid(bot[ok==1] * tab_M[ok==1], x=np.log(tab_M[ok==1]))

        return b        

    def save(self, prefix, suffix='pkl', clobber=False, fields=None):
        """
        Save results of calculation. Pickle parameter file dict.

        Notes
        -----
        1) will save files as prefix.history.suffix and prefix.parameters.pkl.
        2) ASCII files will fail if simulation had multiple populations.

        Parameters
        ----------
        prefix : str
            Prefix of save filename
        suffix : str
            Suffix of save filename. Can be hdf5 (or h5) or pkl.
            Anything else will be assumed to be ASCII format (e.g., .txt).
        clobber : bool
            Overwrite pre-existing files of same name?

        """

        self.sim_gs.save(prefix, clobber=clobber, fields=fields)

        fn = '%s.fluctuations.%s' % (prefix, suffix)

        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else:
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)

        if suffix == 'pkl':
            f = open(fn, 'wb')
            pickle.dump(self.history._data, f)
            f.close()

            try:
                f = open('%s.blobs.%s' % (prefix, suffix), 'wb')
                pickle.dump(self.blobs, f)
                f.close()

                if self.pf['verbose']:
                    print('Wrote {}.blobs.{}'.format(prefix, suffix))
            except AttributeError:
                print('Error writing {}.blobs.{}'.format(prefix, suffix))

        elif suffix in ['hdf5', 'h5']:
            import h5py

            f = h5py.File(fn, 'w')
            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                f.create_dataset(key, data=np.array(self.history[key]))
            f.close()

        # ASCII format
        else:
            f = open(fn, 'w')
            print >> f, "#",

            for key in self.history:
                if fields is not None:
                    if key not in fields:
                        continue
                print >> f, '%-18s' % key,

            print >> f, ''

            # Now, the data
            for i in range(len(self.history[key])):
                s = ''

                for key in self.history:
                    if fields is not None:
                        if key not in fields:
                            continue

                    s += '%-20.8e' % (self.history[key][i])

                if not s.strip():
                    continue

                print >> f, s

            f.close()

        if self.pf['verbose']:
            print('Wrote {}.fluctuations.{}'.format(prefix, suffix))

        #write_pf = True
        #if os.path.exists('%s.parameters.pkl' % prefix):
        #    if clobber:
        #        os.remove('%s.parameters.pkl' % prefix)
        #    else:
        #        write_pf = False
        #        print 'WARNING: %s.parameters.pkl exists! Set clobber=True to overwrite.' % prefix

        #if write_pf:
        #
        #    #pf = {}
        #    #for key in self.pf:
        #    #    if key in self.carryover_kwargs():
        #    #        continue
        #    #    pf[key] = self.pf[key]
        #
        #    if 'revision' not in self.pf:
        #        self.pf['revision'] = get_hg_rev()
        #
        #    # Save parameter file
        #    f = open('%s.parameters.pkl' % prefix, 'wb')
        #    pickle.dump(self.pf, f, -1)
        #    f.close()
        #
        #    if self.pf['verbose']:
        #        print 'Wrote %s.parameters.pkl' % prefix
        #
