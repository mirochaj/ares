import os
import copy
import pickle
import numpy as np
from ..util import ProgressBar
from ..util import ParameterFile
from ..util.Stats import bin_c2e
from .Global21cm import Global21cm
from types import FunctionType, NoneType
from ..util.Misc import get_wave_or_equivalent
from .PowerSpectrum21cm import PowerSpectrum21cm
from ..util.Math import integrate_with_subgrid_interp
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

        zarr = self.halos.tab_z
        #ps = np.zeros((len(self.pops), len(scales), len(waves), len(zarr)))
        #px = np.zeros((len(self.pops), len(self.pops), len(scales), len(waves), len(zarr)))
        # Save contributing pieces

        # Save redshift chunks
        ps_z = np.zeros((len(self.pops), len(self.pops),
            len(scales), len(waves), zarr.size))
        
        # Generate masks first
        fmask = self.get_masks(masking_criteria, pops)

        ##
        # Loop over source populations and compute power spectrum.
        for i, pop in enumerate(self.pops):

            # Honor user-supplied list of populations to include
            if pops is not None:
                if i not in pops:
                    continue

            for j, popx in enumerate(self.pops):
                # Don't recompute terms we already have
                # (symmetry about diagonal for intensity autos with wave1=wave2
                # allows us to just record existing results and move on)
                if is_autos and (j > i):
                    ps_z[i,j,:,:,:] = ps_z[j,i,:,:,:]
                    continue

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
                        #px[i,j,:,:] = _px[i,j,:,:] / to_ps_units
                        ps_z[i,j,:,:,:] = _pz[i,j,:,:,:] / to_ps_units
                        continue

                for k, wave in enumerate(waves):

                    if type(masking_criteria) in [dict, NoneType]:
                        fsel1 = 1-fmask[i]
                    else:
                        fsel1 = 1-fmask[k][i]

                    # Will default to 1h + 2h + shot
                    if j == i:
                        ps_z[i,j,:,k,:] = pop.get_ps_obs(scales,
                            wave_obs1=xmic[k], wave_obs2=xmic2[k],
                            fsel1=fsel1,
                            **kwargs)
                        #ps[i,:,k] = px[i,j,:,k]
                        #ps_z[i,i,:,k,:] = px[i,j,:,k]#pop._ps_obs_integrand.copy()
                        continue

                    if not include_inter_pop:
                        continue

                    if type(masking_criteria) in [dict, NoneType]:
                        fsel2 = 1-fmask[j]
                    else:
                        fsel2 = 1-fmask[k][j]

                    ##
                    # Cross terms only from here on
                    ps_z[i,j,:,k,:] = pop.get_ps_obs(scales,
                        wave_obs1=xmic[k], wave_obs2=xmic2[k],
                        fsel1=fsel1, fsel2=fsel2,
                        pop2=popx, **kwargs)
                    # Setting pop2 to None if i == j avoids recomputing
                    # the luminosity etc. inside other get_ps_* functions
                    # However, I don't think we'll ever get to this
                    # line if i == j, but doesn't hurt I guess
                    #ps_z[i,j,:,k,:] = pop._ps_obs_integrand.copy()

                ##
                # Clear out some memory -- u(k|M) tabs can be big.
                #if hasattr(pop.halos, '_tab_u_nfw'):
                #    del pop.halos._tab_u_nfw

        ##
        # Final step: integrate along redshift axis.
        ps, ps_by_pop = self.get_limber_integral(ps_z, waves=waves, waves2=waves2)

        ##
        # Modify PS units before return
        # The 3-D power spectra should always have units of volume [cMpc^3]
        # potentially times intensity or intensity squared for EBL/galaxy 
        # crosses and EBL autos.
        # We get another factor of cMpc^-1 from integrating along the LoS.
        # 
        ps_z *= to_ps_units * to_ps_units2 / cm_per_mpc**4
        ps *= to_ps_units * to_ps_units2 / cm_per_mpc**4
        ps_by_pop *= to_ps_units * to_ps_units2 / cm_per_mpc**4
        
        if pops is None:
            hist = self.history # poke
            self._history['ps_ebl'] = ps

        self.ps_by_pop = ps_by_pop
        self.ps_by_z = ps_z

        return ps
    
    def get_galaxy_subsample(self, selection_criteria, pops=None,
        return_fraction=True, is_mask=0, logic='or'):
        """
        Subject model galaxies to cuts in redshift, magnitude, and/or color.

        .. note :: This is used both for masking and for sample selection. Note 
            that if selection_criteria is None, we return ones (all galaxies are 
            selected). 

        Parameters
        ----------
        selection_criteria : dict

        Returns
        -------
        A 3-D array with dimensions corresponding to (population, z, Mh).
        Each element is the fraction of halos that satisfy the selection criteria.

        """
        
        if is_mask:
            f_sel = np.zeros((len(self.pops), self.halos.tab_z.size, 
                self.halos.tab_M.size, 2))
        else:    
            f_sel = np.ones((len(self.pops), self.halos.tab_z.size, 
                self.halos.tab_M.size, 2))

        for i, pop in enumerate(self.pops):
            if pops is not None:
                if i not in pops:
                    continue

            if pop.is_diffuse:
                continue

            f_sel[i,:,:,:] = pop.get_galaxy_subsample(selection_criteria, 
                return_fraction=return_fraction, logic=logic)
        
        return f_sel
    
    def get_masks(self, masking_criteria, pops=None, mask_logic='or'):
        ##
        # Need to determine fraction of halos that are masked
        if masking_criteria is None:
            print(f"! WARNING: did you mean not to provide a mask?")
            fmask = np.zeros(len(self.pops))
        else:
            if type(masking_criteria) == dict:
                fmask = self.get_galaxy_subsample(masking_criteria, pops=pops, 
                    is_mask=1, logic=mask_logic)
            else:
                fmask = []
                for mask in masking_criteria:
                    fmask.append(self.get_galaxy_subsample(mask, pops=pops, 
                        is_mask=1, logic=mask_logic))

        self._fmask = np.array(fmask)
        return self._fmask

    def get_ebl_x_galaxies(self, scales, waves, zbins, 
        selection_criteria, masking_criteria,
        wave_units='mic', flux_units='SI', pops=None,
        cache_ipop_mtx=None, **kwargs):
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
        selection_criteria : dict
            A dictionary defining the magnitude and/or color and/or redshift
            cuts used to select galaxies. At the moment, this is just a
            magnitude cut provided as, e.g.,
            > selection_criteria={'mag': [('sdss_z', 22)], 'z': (0, 1)}
            Note that you can also pass in an np.ndarray which is the output
            of a previous call of the form 
            > self.get_galaxy_subsample(selection_criteria)
        masking_criteria : dict
            Like `selection_criteria`, but defines the properties of galaxies
            to be masked out. Can also pass an array if you have already
            mapped the criteria into an array via `get_masks`.
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
        if pops is None:
            pops = [i for i in range(len(self.pops))]
        
        assert zbins.ndim == 2, "Must provide 2-D array of redshift bin edges!"

        # Do some error-handling if waves is 2-D: means the user provided
        # bandpasses instead of a set of wavelengths.
        if waves.ndim == 2:
            assert waves.shape[1] == 2, \
                "If `waves` is 2-D, must have shape (num waves, 2)."

        if wave_units.lower().startswith('mic'):
            pass
        else:
            raise NotImplemented('help')
        
        if flux_units.lower() == 'si':
            to_ps_units = cm_per_m**2 / erg_per_s_per_nW
        elif flux_units.lower() == 'mjy':
            to_ps_units = 1e17
        elif flux_units.lower() == 'cgs':
            to_ps_units = 1
        else:
            raise NotImplemented('help')

        ps = np.zeros((len(scales), len(waves), len(zbins)))
        
        # Save contributing pieces

        # [optonal] Save redshift chunks
        zarr = self.halos.tab_z
        ps_z = np.zeros((len(self.pops), len(self.pops),
            len(scales), len(waves), len(zbins), zarr.size))

        # Read-in or generate selection function and mask from scratch.
        if type(masking_criteria) == np.ndarray:
            fmask = masking_criteria
        else:
            fmask = self.get_masks(masking_criteria, pops)

        if type(selection_criteria) == np.ndarray:
            fsel_allz = selection_criteria
        else:
            fsel_allz = self.get_galaxy_subsample(selection_criteria, pops=pops)
        
        # Get full z-dependent number density
        num_pz = np.zeros((len(self.pops), len(waves), len(zarr)))

        ##
        # Loop over source populations and compute number density
        for i, pop in enumerate(self.pops):
            if pops is not None:
                if i not in pops:
                    continue

            if type(masking_criteria) in [dict, NoneType]:
                num_i = self.pops[i].get_num_from_fsel(fsel_allz[i] * (1 - fmask[i]))
                for k in range(len(waves)):
                    num_pz[i,k,:] = num_i.copy()
            else:
                for k in range(len(waves)):
                    num_pz[i,k,:] = self.pops[i].get_num_from_fsel(fsel_allz[i] * (1 - fmask[k,i]))

        ##
        # Now get Limber integrand for each zbin/channel pair.
        num_p = np.zeros((len(self.pops), len(waves), len(zbins)))
        for h, zbin in enumerate(zbins):

            zlo, zhi = zbin
            zlo = max(zlo, self.pf['final_redshift'])
            zhi = min(zhi, self.pf['initial_redshift'])

            ##
            # Need to determine fraction of halos that are selected 
            # Can we just modify fsel_allz?
            # Apply this redshift bin onto existing selection function
            # (which will be z-dep through mag cut and any overarching z cut
            # but won't know about this particular z bin).
            fsel = []
            for popid, pop in enumerate(pops):
                tmp = np.zeros_like(fsel_allz[popid])
                for iz, z in enumerate(zarr):
                    if z < zbin[0]:
                        continue
                    if z > zbin[1]:
                        continue
                    
                    tmp[iz] = fsel_allz[popid,iz,:,:]
                
                fsel.append(tmp)
            fsel = np.array(fsel)

            if np.all(fsel == 0):
                print(f"! No galaxies found satisfying selection!")
                print(f"! z={zbin}, selection:", selection_criteria)
                continue
            else:
                print(f"! Generating crosses in z={zlo:.3f}-{zhi:.3f}...")
            
            for i, pop in enumerate(self.pops):

                # Honor user-supplied list of populations to include
                if pops is not None:
                    if i not in pops:
                        continue

                fsel1 = fsel[i]

                for j, popx in enumerate(self.pops):

                    # Honor user-supplied list of populations to include
                    if pops is not None:
                        if j not in pops:
                            continue

                    for k, wave in enumerate(waves):
                        if type(masking_criteria) in [dict, NoneType]:
                            fsel1b =1 - fmask[i] 
                            fsel2 = 1 - fmask[j]
                        else:
                            fsel1b =1 - fmask[k][i] 
                            fsel2 = 1 - fmask[k][j]


                        # Try to load from cache [optional]
                        if (cache_ipop_mtx is not None):
                            _px, _pz = cache_ipop_mtx
                            _npops = _px.shape[0]
                            # If we're covered by the cache, use it
                            if i < _npops:
                                # Assumes cache_ipop_mtx is in 
                                # same units as requested here!
                                # Could add check later.
                                #px[i,j,:,:] = _px[i,j,:,:] / to_ps_units
                                ps_z[i,j,:,:,:] = _pz[i,j,:,:,:] / to_ps_units
                                continue

                    

                        if j == 0:
                            num_p[i,k,h] = self.pops[i].get_num_from_fsel(
                                fsel1 * fsel1b, 
                                zbin=zbin)
                        
                        # (pops, pops, scales, waves, zbin, zall)
                        ps_z[i,j,:,k,h,:] = pop.get_xs_obs(scales,
                            wave_obs=wave, zg=zbin, 
                            isnum1=1, isnum2=0,
                            fsel1=fsel1, fsel2=fsel2,
                            pop2=popx, 
                            **kwargs)
                        


        ##
        # Final step: integrate along redshift axis.
        ps, ps_by_pop = self.get_limber_integral(ps_z,
            waves=waves, zbins=zbins, num=num_pz)

        # Modify units
        ps_z *= to_ps_units / cm_per_mpc**2
        ps *= to_ps_units / cm_per_mpc**2
        ps_by_pop *= to_ps_units / cm_per_mpc**2
        
        self.num_by_pop = num_p
        self.num_by_pop_z = num_pz

        self.xs_by_pop = ps_by_pop
        self.xs_by_z = ps_z

        self.fsel_by_pop = fsel_allz
        self.fmask_by_pop = fmask

        #if pops is None:
        #    hist = self.history # poke
        #    self._history['ps_nirb_x_gal'] = scales, scales_inv, waves, ps

        return ps
    
    def get_galaxy_ps(self, scales, zbins, selection_criteria, 
        masking_criteria=None, 
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

        zarr = self.pops[0].halos.tab_z
        Hofz = np.array([self.cosm.HubbleParameter(z) for z in zarr])
        ps_z = np.zeros((len(self.pops), len(self.pops), len(scales), len(zbins), len(zarr)))
        ps = np.zeros((len(scales), len(zbins)))
        
        # Save contributing pieces

        # [optonal] Save redshift chunks
        #ps_z = np.zeros((len(self.pops), len(self.pops),
        #    len(scales), len(waves), self.pops[0].halos.tab_z.size))

        # Get full z-dependent number density
        num_pz = np.zeros((len(self.pops), len(zarr)))
        fsel_allz = self.get_galaxy_subsample(selection_criteria, pops=pops)
        
        for i, pop in enumerate(self.pops):
            if pops is not None:
                if i not in pops:
                    continue
            num_pz[i,:] = self.pops[i].get_num_from_fsel(fsel_allz[i])

        
        # Loop over source populations and compute cross spectrum.
        num_p = np.zeros((len(self.pops), len(zbins)))
        for h, zbin in enumerate(zbins):

            zlo, zhi = zbin
            zlo = max(zlo, self.pf['final_redshift'])
            zhi = min(zhi, self.pf['initial_redshift'])

            galaxy_prop = {'z': (zlo, zhi)}
            galaxy_prop.update(selection_criteria)
            
            ##
            # Need to determine fraction of halos that are selected 
            fsel = self.get_galaxy_subsample(galaxy_prop, pops=pops)

            if np.all(fsel == 0):
                print(f"No galaxies found satisfying selection!")
                print(f"z={zbin}", selection_criteria)
                continue

            for i, pop in enumerate(self.pops):

                # Honor user-supplied list of populations to include
                if pops is not None:
                    if i not in pops:
                        continue
                    
                num_p[i,h] = self.pops[i].get_num_from_fsel(fsel[i], zbin=zbin)
    
                for j, popx in enumerate(self.pops):
                    # Avoid double counting.
                    if (j > i):
                        break
                    
                    # Honor user-supplied list of populations to include
                    if pops is not None:
                        if j not in pops:
                            continue
                    
                    ps_z[i,j,:,h,:] = pop.get_xs_obs(scales,
                        wave_obs=None, zg=zbin, 
                        isnum1=1, isnum2=1,
                        fsel1=fsel[i,:,:], fsel2=fsel[j,:,:],
                        pop2=popx, **kwargs)
            
        ##
        #
        ps, ps_by_pop = self.get_limber_integral(ps_z, zbins=zbins, num=num_pz)
        
        self.gg_by_pop = ps_by_pop

        #if pops is None:
        #    hist = self.history # poke
        #    self._history['ps_nirb_x_gal'] = scales, scales_inv, waves, ps

        return ps#.sum(axis=0).sum(axis=0)
    
    def get_limber_integral(self, ps3d, waves=None, zbins=None, num=None, waves2=None):
        """
        Take a 3-D power spectrum (along lightcone) and perform integration
        along z axis, i.e., perform the Limber integral.

        .. note :: Used internally in the get_ps_* routines.

        Parameters
        ----------
        ps3d : np.ndarray
        """

        zarr = self.halos.tab_z
        dzarr = self.halos.tab_dz
        
        d = np.array([self.cosm.get_dist_los_comoving(0., z) \
            for z in zarr]) / cm_per_mpc
        Hofz = np.array([self.cosm.HubbleParameter(z) \
            for z in zarr])

        dchi_dz_dsq = (c / cm_per_mpc / Hofz) / d**2 

        is_galaxy_auto = waves is None
        is_ebl_auto = (waves is not None) and (zbins is None)
        is_galaxy_ebl_cross = (waves is not None) and (zbins is not None)

        if waves is not None:
            freqs_2d = c / (np.array(waves) * 1e-4)
            dnu = np.abs(np.diff(freqs_2d, axis=1))
            freqs = np.mean(freqs_2d, axis=1)

            if waves2 is not None:
                freqs_2d2 = c / (np.array(waves2) * 1e-4)
                dnu2 = np.abs(np.diff(freqs_2d2, axis=1))
                freqs2 = np.mean(freqs_2d2, axis=1)
            else:
                dnu2 = dnu
                freqs2 = freqs

        ##
        # Shape of ps3d:
        # EBL autos or internal cross: (pops, pops, scales, waves, zarr)
        # Galaxy/EBL crosses: (pops, pops, scales, waves, zbins, zarr)
        # Galaxy autos: (pops, pops, scales, zbins, zarr)

        #npops, npops, nell, nwaves, nzbins, nz = ps3d.shape
        assert ps3d.shape[-1] == zarr.size

        ps_2d = np.zeros((ps3d.shape[2:-1]))
        ps_2d_by_pop = np.zeros((ps3d.shape[0:-1]))

        ##
        # Dimensions of `ps3d` are different for different cases,
        # handle one by one for now. This could be condensed for
        # sure but probably at the expense of clarity.
        if is_ebl_auto:
            zlo = self.pf['final_redshift']
            zhi = self.pf['initial_redshift']

            # Loop over waves
            for j in range(ps3d.shape[3]):
                W_I_sq = (freqs[j] / dnu[j]) * (freqs2[j] / dnu2[j]) \
                     / (4. * np.pi)**2 / (1 + zarr)**4
                # The None slicing here is to match the first axis of
                # `ps3d` aftering summing over populations, which is ell.
                limber_integ = dchi_dz_dsq[None,:] * W_I_sq[None,:] \
                    * ps3d.sum(axis=0).sum(axis=0)[:,j,:]
                
                # Loop over modes
                for k in range(ps3d.shape[2]):
                    ps_2d[k,j] = integrate_with_subgrid_interp(zarr, 
                        limber_integ[k,:], zlo, zhi)
                    
                    ##
                    # Store by population as well
                    for p1 in range(len(self.pops)):
                        for p2 in range(len(self.pops)):
                            limber_integ_bypop = dchi_dz_dsq * W_I_sq \
                                * ps3d[p1,p2,k,j,:]
                            
                            ps_2d_by_pop[p1,p2,k,j] = \
                                integrate_with_subgrid_interp(zarr, 
                                    limber_integ_bypop, zlo, zhi)

        elif is_galaxy_auto:
            # Loop over redshift bins
            for i in range(ps3d.shape[3]):
                zlo, zhi = zbins[i]
                zlo = max(zlo, self.pf['final_redshift'])
                zhi = min(zhi, self.pf['initial_redshift'])
        
                # N(z), though integral will be truncated to 
                # (zlo, zhi) interval below (don't worry)
                n_vs_zall = num[:,:].sum(axis=0)
                
                # Total number of galaxies in this particular redshift bin,
                # summed over source populations.
                n_in_zbin = integrate_with_subgrid_interp(zarr, 
                        num[:,:].sum(axis=0), zlo, zhi)
                
                W_g = n_vs_zall / n_in_zbin / ((c / cm_per_mpc) / Hofz)

                # The None slicing here is to match the first axis of
                # `ps3d` which is ell.
                limber_integ = dchi_dz_dsq[None,:] * W_g[None,:]**2 \
                    * ps3d.sum(axis=0).sum(axis=0)[:,i,:] \
                    / n_vs_zall[None,:]**2
            
                for k in range(ps3d.shape[2]):
                    ps_2d[k,i] = integrate_with_subgrid_interp(zarr, 
                        limber_integ[k,:], zlo, zhi)
                    
                    ##
                    # Save by population too
                    for p1 in range(len(self.pops)):
                        for p2 in range(len(self.pops)):
                            limber_integ_bypop = dchi_dz_dsq \
                                    * W_g**2 \
                                    * ps3d[p1,p2,k,i,:] \
                                    / n_vs_zall**2
                            
                            if np.all(limber_integ_bypop == 0):
                                continue

                            ps_2d_by_pop[p1,p2,k,i] = \
                                integrate_with_subgrid_interp(zarr,
                                    limber_integ_bypop, zlo, zhi)  

        # Galaxy/EBL cross correlation
        elif is_galaxy_ebl_cross:

            # Loop over redshift bins
            for i in range(ps3d.shape[4]):
                zlo, zhi = zbins[i]
                zlo = max(zlo, self.pf['final_redshift'])
                zhi = min(zhi, self.pf['initial_redshift'])
        
                # Loop over waves
                for j in range(ps3d.shape[3]):
                    # N(z), though integral will be truncated to 
                    # (zlo, zhi) interval below (don't worry)
                    n_vs_zall = num[:,j,:].sum(axis=0)
                    
                    # Total number of galaxies in this particular redshift bin,
                    # summed over source populations.
                    n_in_zbin = integrate_with_subgrid_interp(zarr, 
                            num[:,j,:].sum(axis=0), zlo, zhi)
                    
                    W_g = n_vs_zall / n_in_zbin / ((c / cm_per_mpc) / Hofz)
                    W_I = (freqs[j] / dnu[j]) / (4. * np.pi) / (1 + zarr)**2 

                    # The None slicing here is to match the first axis of
                    # `ps3d` which is ell.
                    limber_integ = dchi_dz_dsq[None,:] * W_g[None,:] * W_I[None,:] \
                        * ps3d.sum(axis=0).sum(axis=0)[:,j,i,:] \
                        / n_vs_zall[None,:]
                    
                    # This loop is over ell
                    for k in range(ps3d.shape[2]):
                        # The reason we use this integrator is to be 
                        # as accurate as possible when zbin edges 
                        # (provided by user) don't line up exactly 
                        # with the redshift points in our grid, which
                        # is essentially always since the z gridding 
                        # is not even (usually in fixed time or logx)

                        limb = np.ma.array(limber_integ[k], mask=n_vs_zall==0,
                            fill_value=0)
                        ps_2d[k,j,i] = integrate_with_subgrid_interp(zarr, 
                            limb, zlo, zhi)
                        
                        ##
                        # Save by population too
                        for p1 in range(len(self.pops)):
                            for p2 in range(len(self.pops)):
                                limber_integ_bypop = dchi_dz_dsq \
                                        * W_g * W_I \
                                        * ps3d[p1,p2,k,j,i,:] \
                                        / n_vs_zall
                                
                                limb = np.ma.array(limber_integ_bypop, 
                                    mask=n_vs_zall==0, fill_value=0)
                                
                                ps_2d_by_pop[p1,p2,k,j,i] = \
                                    integrate_with_subgrid_interp(zarr,
                                        limb, zlo, zhi)                        
                                    
        ##
        # Done
        return ps_2d, ps_2d_by_pop
    
    def get_galaxy_number_counts(self, band, magbins, pops=None,
        dlam=10, zmin=None, zmax=None, zbin=0.01, selection_criteria=None,
        volume_density=False):
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

        if pops is not None:
            if type(pops) not in [list, tuple]:
                pops = [pops]
        
        ##
        # Front-load selection function calculation
        #fsel = self.get_galaxy_subsample(selection_criteria, pops=pops)

        # Loop over populations and save results for each one separately
        num_by_pop = {}
        for i, pop in enumerate(self.pops):

            if pops is not None:
                if i not in pops:
                    num_by_pop[i] = 0
                    continue

            if pop.is_diffuse:
                num_by_pop[i] = 0
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
                zmin=_zmin, zmax=_zmax, zbin=zbin,
                selection_criteria=selection_criteria,
                volume_density=volume_density)
            
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
