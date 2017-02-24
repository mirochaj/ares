"""

MetaGalacticBackground.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:43:06 MST 2015

Description: 

"""

import os
import time
import scipy
import pickle
import numpy as np
from ..static import Grid
from types import FunctionType
from ..util import ParameterFile
from scipy.interpolate import interp1d
from ..solvers import UniformBackground
from ..analysis.MetaGalacticBackground import MetaGalacticBackground \
    as AnalyzeMGB
from ..physics.Constants import E_LyA, E_LL, ev_per_hz, erg_per_ev
from ..util.ReadData import _sort_history, flatten_energies, flatten_flux

_scipy_ver = scipy.__version__.split('.')

# This keyword didn't exist until version 0.14 
if float(_scipy_ver[1]) >= 0.14:
    _interp1d_kwargs = {'assume_sorted': True}
else:
    _interp1d_kwargs = {}
    
    
def NH2(z, Mh, fH2=3.5e-4):
    return 1e17 * (fH2 / 3.5e-4) * (Mh / 1e6)**(1. / 3.) * ((1. + z) / 20.)**2.

def f_sh_fl13(zz, Mmin):
    return np.minimum(1, (NH2(zz, Mh=Mmin) / 1e14)**-0.75)    

def get_Mmin_func(zarr, Jlw, Mmin_prev, **kwargs):
    
    # Self-shielding. Function of redshift only!
    if kwargs['feedback_LW_fsh'] is not None:
        f_sh = f_sh_fl13
    elif type(kwargs['feedback_LW_fsh']) is FunctionType:
        f_sh = kwargs['feedback_LW_fsh']    
    else:
        f_sh = lambda z, Mmin: 1.0
    
    # Interpolants for Jlw, Mmin, for this iteration
    # This is Eq. 4 from Visbal+ 2014
    Mmin = lambda zz: np.interp(zz, zarr, Mmin_prev)
    f_J = lambda zz: f_sh(zz, Mmin(zz)) * np.interp(zz, zarr, Jlw)
    
    if kwargs['feedback_LW_Mmin'] is 'visbal2015':
        f_M = lambda zz: 2.5 * 1e5 * pow(((1. + zz) / 26.), -1.5) \
            * (1. + 6.96 * pow(4 * np.pi * f_J(zz), 0.47))
    elif type(kwargs['feedback_LW_Mmin']) is FunctionType:
        f_M = kwargs['feedback_LW_Mmin']
    else:
        raise NotImplementedError('Unrecognized Mmin option: %s' % kwargs['feedback_LW_Mmin'])
        
    return f_M
        
class MetaGalacticBackground(AnalyzeMGB):
    def __init__(self, grid=None, **kwargs):
        """
        Initialize a MetaGalacticBackground object.    
        """
        
        self._grid = grid
        self._has_fluxes = False
        self._has_coeff = False
        
        self.kwargs = kwargs
        
        if not hasattr(self, '_suite'):
            self._suite = []
            
    @property
    def pf(self):
        if not hasattr(self, '_pf'):
            self._pf = ParameterFile(**self.kwargs)
        return self._pf        
            
    @property
    def solver(self):
        if not hasattr(self, '_solver'):
            self._solver = UniformBackground(grid=self.grid, **self.kwargs)
        return self._solver
        
    @property
    def pops(self):
        return self.solver.pops
            
    @property 
    def count(self):
        if not hasattr(self, '_count'):
            self._count = 0
        return self._count     
        
    @property
    def grid(self):
        if self._grid is None:
            self._grid = Grid(
                grid_cells=self.pf['grid_cells'], 
                length_units=self.pf['length_units'], 
                start_radius=self.pf['start_radius'],
                approx_Salpha=self.pf['approx_Salpha'],
                logarithmic_grid=self.pf['logarithmic_grid'],
                cosmological_ics=self.pf['cosmological_ics'],
                )
            self._grid.set_properties(**self.pf)    
                
        return self._grid
    
    def run(self, include_pops=None):
        
        if include_pops is None:
            include_pops = range(self.solver.Npops)
        
        data = {}
        for i in include_pops:
            t1 = time.time()
            z, fluxes = self.run_pop(popid=i)
            t2 = time.time()

            print "pop %i, iteration #%i took %.2g sec" % (i, self.count, t2 -t1)

            data[i] = fluxes
                    
        # Each element of the history is series of lists.
        # The first axis corresponds to redshift, while the second
        # dimension has as many chunks as there are sub-bands for RTE solutions.    
        # Also: redshifts are *descending* at this point
        self._history = data.copy()
        self._suite.append(data.copy())

        count = self.count   # Just to make sure attribute exists
        self._count += 1

        ## 
        # Feedback
        ##
        if not self._is_Mmin_converged(include_pops):
            self.reboot()
            self.run(include_pops=include_pops)

        self._has_fluxes = True

        self._f_Ja = lambda z: np.interp(z, self._zarr, self._Ja)
        self._f_Jlw = lambda z: np.interp(z,self._zarr, self._Jlw)
        
    @property    
    def _lwb_sources(self):
        if not hasattr(self, '_lwb_sources_'):
            self._lwb_sources_ = []
            for i, pop in enumerate(self.pops):
                if pop.is_lw_src:
                    self._lwb_sources_.append(i)
        
        return self._lwb_sources_
        
    def run_pop(self, popid=0):
        """
        Evolve radiation background in time.

        .. note:: Assumes we're using the generator, otherwise the time 
            evolution must be controlled manually.

        Returns
        -------
        Nothing: sets `history` attribute containing the entire evolution
        of the background for each population.

        """
        
        if self.solver.approx_all_pops:
            return None, None
        
        t = 0.0
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
                
        all_fluxes = []
        for z in self.solver.redshifts[popid][-1::-1]:
            _z, fluxes = self.update_fluxes(popid=popid)            
            all_fluxes.append(fluxes)
            
        return self.solver.redshifts[popid][-1::-1], all_fluxes

    def reboot(self):
        delattr(self, '_history')
        delattr(self.solver, '_generators')
        delattr(self.solver, '_emissivities')
        
        # All quantities that depend on Mmin.
        to_del = ['_tab_Mmin_', '_Mmin', '_tab_sfrd_total_',
            '_tab_sfrd_at_threshold_', '_tab_fstar_at_Mmin_',
            '_tab_nh_at_Mmin_', '_tab_MAR_at_Mmin_']

        ## Reset Mmin for feedback-susceptible populations
        for popid in self.pf['feedback_LW_felt_by']:
            pop = self.pops[popid]

            #for key in to_del:
            #    try:
            #        delattr(pop, key)
            #    except AttributeError:
            #        print "Attribute %s didn't exist." % key
            #        continue

            pop._tab_Mmin = np.interp(pop.halos.z, self._zarr, self._Mmin_now)

            self.kwargs['pop_Mmin{%i}' % popid] = self._Mmin_now

            # Only want to re-compute emissivity of LW populations!
            #ehat = self.solver.TabulateEmissivity(self.solver.redshifts[popid],
            #    self.solver.energies[popid], pop)
            #    
            #k = range(self.solver.Npops).index(popid)
            #self.solver._emissivities[k] = ehat
            
        # May not need to do this -- just execute loop just above?
        self.__init__(**self.kwargs)

    #def _init_stepping(self):
    #    """
    #    Initialize lists which bracket radiation background fluxes.
    #
    #    The structure of these lists is as follows:
    #    (1) Each list contains one element per source population.
    #    (2) If that population will approximate the RTE, this entry will be 
    #        None.
    #    (3) The redshift lists, _zlo and _zhi, will just be a sequences of 
    #        floats. 
    #    (4) The flux entries, if not None, will be lists, since in general an
    #        emission band can be broken up into several pieces. In this case,
    #        the number of entries (for each source population) will be equal
    #        to the number of bands, which you can find in self.bands_by_pop.
    #
    #    Sets
    #    ----
    #    Several attributes:
    #    (1) _zhi, _zlo
    #    (2) _fhi, _flo
    #
    #    """
    #
    #    # For "smart" time-stepping
    #    self._zhi = []; self._zlo = []
    #    self._fhi = []; self._flo = []
    #    
    #    # Looping over populations.
    #    z_by_pop = []
    #    for i, generator in enumerate(self.generators):
    #
    #        # Recall that each generator may actually be a list of generators,
    #        # one for each (sub-)band.
    #        
    #        if (generator == [None]) or (generator is None):
    #            self._zhi.append(None)
    #            self._zlo.append(None)
    #            self._fhi.append(None)
    #            self._flo.append(None)
    #            continue
    #
    #        # Only make it here when real RT is happenin'
    #
    #        # Setup arrays (or lists) for flux solutions
    #        _fhi = []
    #        _flo = []
    #        for j, gen in enumerate(generator):
    #            if gen.__name__ == '_flux_generator_generic':
    #                _fhi.append(np.zeros_like(self.energies[i][j]))
    #                _flo.append(np.zeros_like(self.energies[i][j]))
    #                continue
    #
    #            # Otherwise, there are sub-bands (i.e., sawtooth)
    #            _fhi.append(np.zeros_like(np.concatenate(self.energies[i][j])))
    #            _flo.append(np.zeros_like(np.concatenate(self.energies[i][j])))
    #
    #        # Loop over sub-bands and retrieve fluxes
    #        for j, gen in enumerate(generator):
    #
    #            # Tap generator, grab fluxes
    #            zhi, flux = gen.next()
    #
    #            # Increment the flux
    #            _fhi[j] += flatten_flux(flux).copy()
    #            
    #            # Tap generator, grab fluxes (again)
    #            zlo, flux = gen.next()
    #                                                                
    #            # Increment the flux (again)
    #            _flo[j] += flatten_flux(flux).copy()
    #
    #        # Save fluxes for this population
    #        self._zhi.append([zhi for k in range(len(generator))])
    #        self._zlo.append([zlo for k in range(len(generator))])
    #        
    #        self._fhi.append(_fhi)
    #        self._flo.append(_flo)
    #
    #        z_by_pop.append(zlo)
    #
    #    # Set the redshift based on whichever population took the smallest
    #    # step. Other populations will interpolate to find flux.
    #    self.update_redshift(max(z_by_pop))

    #def step(self, popid=0):
    #    """
    #    Initialize generator for the meta-galactic radiation background.
    #    
    #    ..note:: This can run asynchronously with a MultiPhaseMedium object.
    #
    #    Returns
    #    -------
    #    Generator for the background radiation field. Yields the flux for 
    #    each population.
    #
    #    """
    #
    #    t = 0.0
    #    z = self.pf['initial_redshift']
    #    zf = self.pf['final_redshift']
    #            
    #    # Start the generator
    #    while z > zf:     
    #        z, fluxes = self.update_fluxes(popid=popid)            
    #                    
    #        yield z, fluxes

    #def update_redshift(self, z):
    #    self.z = z

    @property
    def history(self):
        if hasattr(self, '_history'):
            pass
        elif hasattr(self, 'all_fluxes'):
            self._history = _sort_history(self.all_fluxes)
        else:
            raise NotImplemented('help!')
    
        return self._history
        
    def update_fluxes(self, popid=0):
        """
        Loop over flux generators and retrieve the next values.

        ..note:: Populations need not have identical redshift sampling.

        Returns
        -------
        Current redshift and dictionary of fluxes. Each element of the flux
        dictionary corresponds to a single population, and within that, there
        are separate lists for each sub-band over which we solve the RTE.

        """

        pop_generator = self.solver.generators[popid]

        # Skip approximate (or non-contributing) backgrounds
        if pop_generator is None:
            return None, None

        fluxes_by_band = []

        # For each population, the band is broken up into pieces
        for j, generator in enumerate(pop_generator):
            z, f = generator.next()
            fluxes_by_band.append(flatten_flux(f))

        return z, fluxes_by_band

    def update_rate_coefficients(self, z, **kwargs):
        """
        Compute ionization and heating rate coefficients.

        Parameters
        ----------
        z : float
            Current redshift.

        Returns
        -------
        Dictionary of rate coefficients.

        """
        
        # Must compute rate coefficients from fluxes     
        if self.solver.approx_all_pops:
            kwargs['fluxes'] = [None] * self.solver.Npops
            return self.solver.update_rate_coefficients(z, **kwargs)
                
        if not self._has_fluxes:
            self.run()
        
        if not self._has_coeff:
            
            kw = kwargs.copy()
            
            kw['return_rc'] = True
            
            # This chunk only gets executed once
            self._rc_tabs = [{} for i in range(self.solver.Npops)]
            
            # very similar chunk of code lives in update_fluxes...
            # could structure better, but i'm tired.
            
            fluxes = {i:None for i in range(self.solver.Npops)}
            for i, pop_generator in enumerate(self.solver.generators):
                
                kw['zone'] = self.pops[i].zone
                also = {}
                for sp in self.grid.absorbers:
                    also['%s_%s' % (self.pops[i].zone, sp)] = 1.0
                kw.update(also)     
                
                zarr = self.solver.redshifts[i]
                Nz = len(zarr)

                self._rc_tabs[i]['k_ion'] = np.zeros((Nz,
                        self.grid.N_absorbers))
                self._rc_tabs[i]['k_ion2'] = np.zeros((Nz,
                        self.grid.N_absorbers, self.grid.N_absorbers))
                self._rc_tabs[i]['k_heat'] = np.zeros((Nz,
                        self.grid.N_absorbers))
                self._rc_tabs[i]['Ja'] = np.zeros(Nz)
                self._rc_tabs[i]['Jlw'] = np.zeros(Nz)
            
                # Need to cycle through redshift here
                for _iz, redshift in enumerate(zarr):
                          
                    if redshift < self.pf['final_redshift']:
                        break
                        
                    if redshift < self.pf['kill_redshift']:
                        break    
                    
                    # Get fluxes if need be
                    if pop_generator is None:
                        kw['fluxes'] = None
                    else:
                        self._rc_tabs[i]['Ja'][_iz] = self._f_Ja(redshift)
                        self._rc_tabs[i]['Jlw'][_iz] = self._f_Jlw(redshift)
                        
                        # Fluxes are in order of *descending* redshift!
                        fluxes[i] = self.history[i][Nz-_iz-1]
                        kw['fluxes'] = fluxes
                        
                    # This routine expects fluxes to be a dictionary, with
                    # the keys being population ID # and the elements lists
                    # of fluxes in each (sub-) band.    
                    coeff = self.solver.update_rate_coefficients(redshift, 
                        popid=i, **kw)
                    
                    self._rc_tabs[i]['k_ion'][_iz,:] = \
                        coeff['k_ion'].copy()
                    self._rc_tabs[i]['k_ion2'][_iz,:] = \
                        coeff['k_ion2'].copy()
                    self._rc_tabs[i]['k_heat'][_iz,:] = \
                        coeff['k_heat'].copy()
                        
            self._interp = [{} for i in range(self.solver.Npops)]
            for i, pop in enumerate(self.pops):
                zarr = self.solver.redshifts[i]

                # Create functions
                self._interp[i]['k_ion'] = \
                    [None for _i in range(self.grid.N_absorbers)]
                self._interp[i]['k_ion2'] = \
                    [[None,None,None] for _i in range(self.grid.N_absorbers)]
                self._interp[i]['k_heat'] = \
                    [None for _i in range(self.grid.N_absorbers)]
            
                self._interp[i]['Ja'] = interp1d(zarr, 
                    self._rc_tabs[i]['Ja'], 
                    bounds_error=False, fill_value=0.0)
                self._interp[i]['Jlw'] = interp1d(zarr, 
                    self._rc_tabs[i]['Jlw'], 
                    bounds_error=False, fill_value=0.0)    
            
                for j in range(self.grid.N_absorbers):
                    self._interp[i]['k_ion'][j] = \
                        interp1d(zarr, self._rc_tabs[i]['k_ion'][:,j], 
                            bounds_error=False, fill_value=0.0)    
                    self._interp[i]['k_heat'][j] = \
                        interp1d(zarr, self._rc_tabs[i]['k_heat'][:,j], 
                            bounds_error=False, fill_value=0.0)    
                    
                    for k in range(self.grid.N_absorbers):
                        self._interp[i]['k_ion2'][j][k] = \
                            interp1d(zarr, self._rc_tabs[i]['k_ion2'][:,j,k],
                                bounds_error=False, fill_value=0.0)
                     
            self._has_coeff = True  
                           
            return self.update_rate_coefficients(z, **kwargs)                    
            
        else:
                        
            to_return = \
            {
             'k_ion': np.zeros((1,self.grid.N_absorbers)),
             'k_ion2': np.zeros((1,self.grid.N_absorbers, self.grid.N_absorbers)),
             'k_heat': np.zeros((1,self.grid.N_absorbers)),
            }
        
            for i, pop in enumerate(self.pops):
        
                if pop.zone != kwargs['zone']:
                    continue
        
                fset = self._interp[i]             
        
                # Call interpolants, add 'em all up.
                this_pop = \
                {
                 'k_ion':  np.array([[fset['k_ion'][j](z) \
                    for j in range(self.grid.N_absorbers)]]),
                 'k_heat': np.array([[fset['k_heat'][j](z) \
                    for j in range(self.grid.N_absorbers)]]),
                 'Ja': fset['Ja'](z),
                 'Jlw': fset['Jlw'](z),
                }
                                
                # Convert to rate coefficient                
                for j, absorber in enumerate(self.grid.absorbers):
                    x = kwargs['%s_%s' % (pop.zone, absorber)]
                    this_pop['k_ion'][0][j] /= x
                    
                    # No helium for cgm, at least not this carefully
                    if pop.zone == 'cgm':
                        break

                tmp = np.zeros((self.grid.N_absorbers, self.grid.N_absorbers))
                for j in range(self.grid.N_absorbers):
                    for k in range(self.grid.N_absorbers):
                        tmp[j,k] = fset['k_ion2'][j][k](z)

                this_pop['k_ion2'] = np.array([tmp])

                for key in to_return:
                    to_return[key] += this_pop[key]

            return to_return
                          
    @property
    def z_unique(self):
        if not hasattr(self, '_z_unique'):
            _allz = [self.solver.redshifts[i] for i in range(self.solver.Npops)]
            self._z_unique = np.unique(np.concatenate(_allz))
        return self._z_unique        
                
    def get_uvb_tot(self, include_pops=None):
        
        if include_pops is None:
            include_pops = range(self.solver.Npops)
        
        # Compute JLW to get estimate for Mmin^(k+1) 
        _allz = []
        _f_Ja = []
        _f_Jlw = []
        for i, popid in enumerate(include_pops):
            
            _z, _Ja, _Jlw = self.get_uvb(popid)
            
            _allz.append(_z)
            _f_Jlw.append(interp1d(_z, _Jlw, kind='linear'))
            _f_Ja.append(interp1d(_z, _Ja, kind='linear'))
        
        zarr = np.unique(np.concatenate(_allz))
        
        Jlw = np.zeros_like(zarr)
        Ja = np.zeros_like(zarr)
        for i, popid in enumerate(include_pops):
            Jlw += _f_Jlw[i](zarr)
            Ja += _f_Ja[i](zarr)
            
        return zarr, Ja, Jlw

    def _is_Mmin_converged(self, include_pops):

        if include_pops is None:
            include_pops = range(self.solver.Npops)

        # Otherwise, grab all the fluxes
        zarr, Ja, Jlw = self.get_uvb_tot(include_pops)
        self._zarr = zarr
        self._Ja = Ja
        self._Jlw = Jlw
        
        # Return right away if feedback is OFF.
        if self.pf['feedback_LW_Mmin'] is None:
            return True

        # Instance of the population that "feels" the feedback.
        # Need for (1) initial _Mmin_pre value, and (2) setting ceiling
        pop_fb = self.pops[self.pf['feedback_LW_felt_by'][0]]

        if self.count == 1:
            self._Mmin_pre = pop_fb.Mmin(zarr)
            
        # Function to compute the next Mmin(z) curve
        f_M = get_Mmin_func(zarr, Jlw / 1e-21, self._Mmin_pre, **self.pf)

        # Use this on the next iteration
        if self.count > 1:
            _Mmin_next = np.mean([f_M(zarr), self._Mmin_pre], axis=0)
        else:
            _Mmin_next = f_M(zarr)

        # Potentially impose ceiling on Mmin
        Tcut = self.pf['feedback_LW_Tcut']

        # Instance of the population that "feels" the feedback.
        pop_fb = self.pops[self.pf['feedback_LW_felt_by'][0]]
        Mmin_ceil = pop_fb.halos.VirialMass(Tcut, zarr)

        # Final answer.
        Mmin = np.minimum(_Mmin_next, Mmin_ceil)

        ##
        # Setup interpolant
        ##
        #f_Mmin = lambda zz: 10**np.interp(zz, zarr, np.log10(Mmin))
        
        if self.count == 1:
            self._Mmin_pre = pop_fb.Mmin(zarr)
            self._Mmin_bank = [self._Mmin_pre.copy()]
        else:    
            self._Mmin_pre = self._Mmin_now.copy()
            
        self._Mmin_now = Mmin.copy()
        
        # Save for prosperity
        self._Mmin_bank.append(self._Mmin_now.copy())
        
        # Compare Mmin of last two iterations.
        # Can't be converged after 1 iteration!
        if self.count == 1:
            return False
        
        if self.count >= self.pf['feedback_LW_maxiter']:
            return True 
        
        rtol = self.pf['feedback_LW_Mmin_rtol']
        atol = self.pf['feedback_LW_Mmin_atol']       
        
        # Less stringent requirement, that mean error meet tolerance.
        if self.pf['feedback_LW_mean_err']:
            err_rel = np.abs((self._Mmin_pre - self._Mmin_now) \
                / self._Mmin_now)
            err_abs = np.abs(self._Mmin_now - self._Mmin_pre)
        
            if rtol > 0:
                if err_rel.mean() > rtol:
                    return False
                elif err_rel.mean() < rtol and (atol == 0):
                    return True
        
            # Only make it here if rtol is satisfied or irrelevant
        
            if atol > 0:
                if err_abs.mean() < atol:
                    return True    
        # More stringent: that all Mmin values must have converged independently            
        else:
            converged = np.allclose(self._Mmin_pre, self._Mmin_now,
                rtol=rtol, atol=atol)    
        
        return converged
            
    def get_uvb(self, popid):
        """
        Return Ly-a and LW background flux in units of erg/s/cm^2/Hz/sr.
        """
        
        k = range(self.solver.Npops).index(popid)

        # Approximate sources
        if np.any(self.solver.solve_rte[k]):
            z, E, flux = self.get_history(popid=popid, flatten=True)
            
            l = np.argmin(np.abs(E - E_LyA))     # should be 0
                                                            
            # Redshift is first dimension!
            Ja = flux[:,l]
                    
            # Find photons in LW band    
            is_LW = np.logical_and(E >= 11.18, E <= E_LL)
            
            dnu = (E_LL - 11.18) / ev_per_hz

            # Need to do an integral to finish this one.
            Jlw = np.zeros_like(z)
        
        else:
            # Need a redshift array!
            dz = 0.1
            z = np.arange(self.pf['final_redshift'], 
                self.pf['initial_redshift']+dz, dz)
            Ja = np.zeros_like(z)
            Jlw = np.zeros_like(z)
        
        ##
        # Loop over redshift
        ##
        for i, redshift in enumerate(z): 
            
            if not np.any(self.solver.solve_rte[k]):
                Ja[i] = self.solver.LymanAlphaFlux(redshift, popid=popid)
                if self.pf['feedback_LW_Mmin'] is not None:   
                    Jlw[i] = self.solver.LymanWernerFlux(redshift, popid=popid)
                continue
                       
            LW_flux = flux[i,is_LW]
            
            # Convert to energy units, and per eV to prep for integral
            LW_flux *= E[is_LW] * erg_per_ev / ev_per_hz
            
            Jlw[i] = np.trapz(LW_flux, x=E[is_LW]) / dnu
        
        return z, Ja, Jlw
            
    def update_Jlw(self, popid, bandid, fluxes_in, fluxes_nested=True):
        """
        Get new Ja and Jlw.
        """
        
        i, j = popid, bandid
        
        Earr = np.concatenate(self.solver.energies[i][j])
        l = np.argmin(np.abs(Earr - E_LyA))     # should be 0
        
        if fluxes_nested:
            fluxes = fluxes_in[i][j]
        else:
            fluxes = flatten_flux(fluxes_in)
        
        Ja = fluxes[l]
        
        ##
        # Compute JLW
        ##
        
        # Find photons in LW band    
        is_LW = np.logical_and(Earr >= 11.18, Earr <= E_LL)
        
        # And corresponding fluxes
        flux = fluxes[is_LW]
        
        # Convert to energy units, and per eV to prep for integral
        flux *= Earr[is_LW] * erg_per_ev / ev_per_hz
        
        dnu = (E_LL - 11.18) / ev_per_hz
        Jlw = np.trapz(flux, x=Earr[is_LW]) / dnu
        
        return Ja, Jlw
        
    def get_integrated_flux(self, band, popid=0):
        """
        Return integrated flux in supplied (Emin, Emax) band at all redshifts.
        """
        
        zarr, Earr, flux = self.get_history(popid, True, True)
        
        i1 = np.argmin(np.abs(Earr - band[0]))
        i2 = np.argmin(np.abs(Earr - band[1]))
                
        return zarr, np.trapz(flux[:,i1:i2], x=Earr[i1:i2], axis=1)

    def get_history(self, popid=0, flatten=False):
        """
        Grab data associated with a single population.

        Parameters
        ----------
        popid : int
            ID number for population of interest.
        flatten : bool
            For sawtooth calculations, the energies are broken apart into 
            different bands which have different sizes. Set this to true if
            you just want a single array, rather than having the energies
            and fluxes broken apart by their band.

        Returns
        -------
        Tuple containing the redshifts, energies, and fluxes for the given
        population, in that order.
        
        if flatten == True:
            The energy array is 1-D.
            The flux array will have shape (z, E)
        else:
            The energies are stored as a list. The number of elements will
            be determined by how many sub-bands there are. Each element will
            be a list or an array, depending on whether or not there is a 
            sawtooth component to that particular background.
        
        """
        
        hist = self.history[popid]
            
        # First, get redshifts. If not run "thru run", then they will
        # be in descending order so flip 'em.
        #if self._is_thru_run or self.pf['compute_fluxes_at_start']:
        z = self.solver.redshifts[popid]
        #else:
        #    # This may change on the fly due to sub-cycling and such
        #    z = np.array(self.all_z).T[popid][-1::-1]

        if flatten:
            E = flatten_energies(self.solver.energies[popid])

            f = np.zeros([len(z), E.size])
            for i, flux in enumerate(hist):
                fzflat = []
                for j in range(len(self.solver.energies[popid])):
                    fzflat.extend(flux[j])

                f[i] = np.array(fzflat)

            # "tr" = "to return"
            z_tr = z
            E_tr = E
            f_tr = np.array(f)[-1::-1,:]
        else:
            z_tr = z
            E_tr = self.solver.energies[popid]
            f_tr = hist[-1::-1][:]
            
        # We've flipped the fluxes too since they are inherently in 
        # order of descending redshift.    
        return z_tr, E_tr, f_tr

    def save(self, prefix, suffix='pkl', clobber=False):
        """
        Save results of calculation.

        Notes
        -----
        1) will save files as prefix.rfield.suffix.
        2) ASCII files will fail if simulation had multiple populations.

        Parameters
        ----------
        prefix : str
            Prefix of save filename
        suffix : str
            Suffix of save filename. Can be hdf5 (or h5), pkl, or npz. 
            Anything else will be assumed to be ASCII format (e.g., .txt).
        clobber : bool
            Overwrite pre-existing files of same name?
    
        """

        fn_1 = '%s.fluxes.%s' % (prefix, suffix)
        fn_2 = '%s.emissivities.%s' % (prefix, suffix)

        all_fn = [fn_1, fn_2]

        f_data = [self.get_history(i, flatten=True) for i in range(self.solver.Npops)]
        z = [f_data[i][0] for i in range(self.solver.Npops)]
        E = [f_data[i][1] for i in range(self.solver.Npops)]
        fl = [f_data[i][2] for i in range(self.solver.Npops)]

        all_data = [(z, E, fl), 
            (self.solver.redshifts, self.solver.energies, self.solver.emissivities)]

        for i, data in enumerate(all_data):
            fn = all_fn[i]

            if os.path.exists(fn):
                if clobber:
                    os.remove(fn)
                else:
                    print '%s exists! Set clobber=True to overwrite.' % fn
                    continue

            if suffix == 'pkl':
                f = open(fn, 'wb')
                pickle.dump(data, f)
                f.close()

            elif suffix in ['hdf5', 'h5']:
                raise NotImplementedError('no hdf5 support for this yet.')

            elif suffix == 'npz':
                f = open(fn, 'w')
                np.savez(f, **data)
                f.close()
            
            # ASCII format
            else:  
                raise NotImplementedError('No ASCII support for this.')          
                
            print 'Wrote %s.' % fn
    
    
    
    