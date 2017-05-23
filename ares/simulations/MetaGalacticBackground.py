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
from ..physics.Constants import E_LyA, E_LL, ev_per_hz, erg_per_ev, \
    sqdeg_per_std
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
    if kwargs['feedback_LW_fsh']:
        f_sh = f_sh_fl13
    #elif type(kwargs['feedback_LW_fsh']) is FunctionType:
    #    f_sh = kwargs['feedback_LW_fsh']    
    else:
        f_sh = lambda z, Mmin: 1.0
    
    # Interpolants for Jlw, Mmin, for this iteration
    # This is Eq. 4 from Visbal+ 2014
    Mmin = lambda zz: np.interp(zz, zarr, Mmin_prev)
    f_J = lambda zz: f_sh(zz, Mmin(zz)) * np.interp(zz, zarr, Jlw)
    
    if kwargs['feedback_LW_Mmin'] == 'visbal2015':
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
        if not hasattr(self, '_pops'):
            self._pops = self.solver.pops
        return self._pops
            
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
            
    @property
    def rank(self):
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.rank
        except ImportError:
            rank = 0
    
        return rank
    
    def run(self, include_pops=None):
        """
        Loop over populations, determine background intensity.
        
        Parameters
        ----------
        include_pops : list
            For internal use only!
            
        """    
                    
        # In this case, we can evolve all LW sources first, and wait 
        # to do other populations until the very end.
        if self.pf['feedback_LW'] and (include_pops is None):
            # This means it's the first iteration
            include_pops = self._lwb_sources
        elif include_pops is None:
            include_pops = range(self.solver.Npops)
        
        if not hasattr(self, '_data_'):
            self._data_ = {}
        
        for i, popid in enumerate(include_pops):            
            z, fluxes = self.run_pop(popid=popid)
            self._data_[popid] = fluxes
                    
        # Each element of the history is series of lists.
        # The first axis corresponds to redshift, while the second
        # dimension has as many chunks as there are sub-bands for RTE solutions.    
        # Also: redshifts are *descending* at this point
        self._history = self._data_.copy()
        self._suite.append(self._data_.copy())

        count = self.count   # Just to make sure attribute exists
        self._count += 1

        ## 
        # Feedback
        ##
        if self._is_Mmin_converged(self._lwb_sources):
            self._has_fluxes = True
            self._f_Ja = lambda z: np.interp(z, self._zarr, self._Ja)
            self._f_Jlw = lambda z: np.interp(z, self._zarr, self._Jlw)
            
            # Now that feedback is done, evolve all non-LW sources to get
            # final background.
            if include_pops == self._lwb_sources:
                self.reboot(include_pops=self._not_lwb_sources)
                self.run(include_pops=self._not_lwb_sources)
            
        else:
            if self.pf['verbose']:
                print "LWB: iteration #%i..." % (self.count+1)
                
            self.reboot()
            self.run(include_pops=self._lwb_sources)
    
    @property        
    def today(self):
        """
        Take background intensity at final redshift and evolve to z=0.
        
        This is just the second term of Eq. 25 in Mirocha (2014).
        """
        
        _fluxes_today = []
        _energies_today = []
        
        for popid, pop in enumerate(self.pops):
            if not self.solver.solve_rte[popid]:
                _fluxes_today.append(None)
                _energies_today.append(None)
                continue
                
            z, E, flux = self.get_history(popid=popid, flatten=True)    
                
            Et = E / (1. + z[0])
            ft = flux[0] / (1. + z[0])**2
            
            _energies_today.append(Et)
            _fluxes_today.append(ft)
            
        return _energies_today, _fluxes_today
                           
    @property        
    def jsxb(self):
        if not hasattr(self, '_jsxb'):
            self._jsxb = self.jxrb(band='soft')
        return self._jsxb

    @property
    def jhxb(self):
        if not hasattr(self, '_jhxb'):
            self._jhxb = self.jxrb(band='hard')
        return self._jhxb

    def jxrb(self, band='soft'):
        """
        Compute soft X-ray background flux at z=0.
        """

        jx = 0.0
        Ef, ff = self.today
        for popid, pop in enumerate(self.pops):
            if Ef[popid] is None:
                continue
                
            flux_today = ff[popid] * Ef[popid] \
                * erg_per_ev / sqdeg_per_std / ev_per_hz
                
            if band == 'soft':
                Eok = np.logical_and(Ef[popid] >= 5e2, Ef[popid] <= 2e3)
            elif band == 'hard':
                Eok = np.logical_and(Ef[popid] >= 2e3, Ef[popid] <= 1e4)
            else:
                raise ValueError('Unrecognized band! Only know \'hard\' and \'soft\'')
        
            Earr = Ef[popid][Eok]
            # Find integrated 0.5-2 keV flux
            dlogE = np.diff(np.log10(Earr))
            
            jx += np.trapz(flux_today[Eok] * Earr, dx=dlogE) * np.log(10.)
                          
        return jx          
                           
    @property
    def _not_lwb_sources(self):
        if not hasattr(self, '_not_lwb_sources_'):
            self._not_lwb_sources_ = []
            for i, pop in enumerate(self.pops):
                if pop.is_lw_src:
                    continue
                self._not_lwb_sources_.append(i)
        
        return self._not_lwb_sources_
        
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

    def reboot(self, include_pops=None):
        delattr(self, '_history')
        delattr(self, '_pf')
                
        if include_pops is None:
            include_pops = range(self.solver.Npops)
        
        ## All quantities that depend on Mmin.
        #to_del = ['_tab_Mmin_', '_Mmin', '_tab_sfrd_total_',
        #    '_tab_sfrd_at_threshold_', '_tab_fstar_at_Mmin_',
        #    '_tab_nh_at_Mmin_', '_tab_MAR_at_Mmin_']

        # Reset Mmin for feedback-susceptible populations
        for popid in include_pops:
            #pop = self.pops[popid]
            #
            #if not pop.feels_feedback:
            #    continue
            #    
            #if self.pf['pop_Tmin{%i}' % popid] is not None:
            #    if self.pf['pop_Tmin{%i}' % popid] >= 1e4:
            #        continue

            #for key in to_del:
            #    try:
            #        delattr(pop, key)
            #    except AttributeError:
            #        print "Attribute %s didn't exist." % key
            #        continue
            
            # Linked populations will get 
            if type(self.pf['pop_Mmin{%i}' % popid]) is str:
                continue
                        
            self.kwargs['pop_Mmin{%i}' % popid] = \
                np.interp(self.pops[popid].halos.z, self.z_unique, self._Mmin_now)
                            
            # Need to make sure, if any populations are linked to this Mmin,
            # that they get updated too.

            #if not self.pf['feedback_clear_solver']:
            #    pop._tab_Mmin = np.interp(pop.halos.z, self._zarr, self._Mmin_now)
            #    bands = self.solver.bands_by_pop[popid]
            #    z, nrg, tau, ehat = self.solver._set_grid(pop, bands, 
            #        compute_emissivities=True)
            #    
            #    k = range(self.solver.Npops).index(popid)
            #    self.solver._emissivities[k] = ehat
                
        if self.pf['feedback_clear_solver']:
            delattr(self, '_solver')
            delattr(self, '_pops')
        else:
            delattr(self.solver, '_generators')        

        # May not need to do this -- just execute loop just above?
        self.__init__(**self.kwargs)
                
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
                          
                    # The use of "continue", rather than "break", is
                    # VERY important because redshifts are in ascending order
                    # at this point.
                    if redshift < self.pf['final_redshift']:
                        # This shouldn't ever happen...
                        continue
                        
                    if redshift < self.pf['kill_redshift']:
                        continue    
                    
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
            if sum([item is not None for item in _allz]) == 0:
                dz = self.pf['fallback_dz']
                self._z_unique = np.arange(self.pf['final_redshift'],
                    self.pf['initial_redshift']+dz, dz)
            else:
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
        
        zarr = self.z_unique
        
        Jlw = np.zeros_like(zarr)
        Ja = np.zeros_like(zarr)
        for i, popid in enumerate(include_pops):
            Jlw += _f_Jlw[i](zarr)
            Ja += _f_Ja[i](zarr)
            
        return zarr, Ja, Jlw
        
    @property
    def _LW_felt_by(self):
        if not hasattr(self, '_LW_felt_by_'):
            self._LW_felt_by_ = []
            for popid, pop in enumerate(self.pops):
                
                Mmin = pop.pf['pop_Mmin']
                
                if type(Mmin) is str:
                    self._LW_felt_by_.append(popid)
                    continue
                                
                Tmin = pop.pf['pop_Tmin']
                if type(Tmin) is str:
                    pass
                
                if Mmin is None: 
                    T = Tmin
                else:
                    T = pop.halos.VirialTemperature(Mmin, pop.halos.z,
                        self.pf['mu'])
                
                if type(T) in [float, int, np.float64]:
                    if T < 1e4:
                        self._LW_felt_by_.append(popid)
                else:
                    if np.any(T < 1e4):
                        self._LW_felt_by_.append(popid)  
                        
        return self._LW_felt_by_                                      

    def _is_Mmin_converged(self, include_pops):

        # Need better long-term fix: Lya sources aren't necessarily LW 
        # sources, if (for example) approx_all_pops = True. 
        if not self.pf['feedback_LW']:
            # Will use all then
            include_pops = None
        elif include_pops is None:
            include_pops = range(self.solver.Npops)

        # Otherwise, grab all the fluxes
        zarr, Ja, Jlw = self.get_uvb_tot(include_pops)
        self._zarr = zarr
        
        Ja = np.maximum(Ja, 0.)
        Jlw = np.maximum(Jlw, 0.)
        
        self._Ja = Ja
        self._Jlw = Jlw
        
        if not self.pf['feedback_LW']:
            return True
        
        # Instance of a population that "feels" the feedback.
        # Need for (1) initial _Mmin_pre value, and (2) setting ceiling
        pop_fb = self.pops[self._lwb_sources[0]]

        # Save last iteration's solution for Mmin(z)
        if self.count == 1:
            # For a guess, take the minimum Mmin of all LW-producing pops
            self._Mmin_pre = np.min([self.pops[idnum].Mmin(zarr) \
                for idnum in self._lwb_sources], axis=0)
                
            self._Mmin_bank = [self._Mmin_pre.copy()]
            self._Jlw_bank = [Jlw]
        else:
            self._Mmin_pre = self._Mmin_now.copy()
            
        self._Mmin_pre = np.maximum(self._Mmin_pre, 
            pop_fb.halos.Mmin_floor(zarr))    
            
        if np.any(np.isnan(Jlw)):
            Jlw[np.argwhere(np.isnan(Jlw))] = 0.0
                    
        # Function to compute the next Mmin(z) curve
        # Use of Mmin here is for shielding prescription (if used), 
        # but implicitly used since it set Jlw for this iteration.
        f_M = get_Mmin_func(zarr, Jlw / 1e-21, self._Mmin_pre, **self.pf)

        # Use this on the next iteration
        Mnext = f_M(zarr)
        if (self.count > 1) and self.pf['feedback_LW_softening'] is not None:
            if self.pf['feedback_LW_softening'] == 'sqrt':
                _Mmin_next = np.sqrt(Mnext * self._Mmin_pre)
            elif self.pf['feedback_LW_softening'] == 'mean':
                _Mmin_next = np.mean([Mnext, self._Mmin_pre], axis=0)
            elif self.pf['feedback_LW_softening'] == 'log10_mean':
                _Mmin_next = 10**np.mean([np.log10(Mnext), 
                    np.log10(self._Mmin_pre)], axis=0)
            else:
                raise NotImplementedError('help')
        else:
            _Mmin_next = Mnext
            
        # Need to apply Mmin floor
        _Mmin_next = np.maximum(_Mmin_next, pop_fb.halos.Mmin_floor(zarr))

        # Potentially impose ceiling on Mmin
        Tcut = self.pf['feedback_LW_Tcut']

        # Instance of a population that "feels" the feedback.
        # Just need access to a few HMF routines.
        Mmin_ceil = pop_fb.halos.VirialMass(Tcut, zarr)

        # Final answer.
        Mmin = np.minimum(_Mmin_next, Mmin_ceil)

        # Set new solution
        self._Mmin_now = Mmin.copy()
        # Save for our records (read: debugging)
        self._Mmin_bank.append(self._Mmin_now.copy())
        self._Jlw_bank.append(Jlw)

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
                
            perfect = np.all(np.equal(self._Mmin_pre, self._Mmin_now))
            
            # Nobody is perfect. Something is weird. Keep on keepin' on.
            # This is happening at iteration 3 for some reason...
            if perfect:
                converged = False    
                
        return converged
            
    def get_uvb(self, popid):
        """
        Return Ly-a and LW background flux in units of erg/s/cm^2/Hz/sr.
        """
        
        # Approximate sources
        if np.any(self.solver.solve_rte[popid]):
            z, E, flux = self.get_history(popid=popid, flatten=True)
            
            if self.pf['secondary_lya'] and self.pops[popid].is_ion_src_igm:
                Ja = np.zeros_like(z) # placeholder
            else:
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
            z = self.z_unique
            Ja = np.zeros_like(z)
            Jlw = np.zeros_like(z)

        ##
        # Loop over redshift
        ##
        for i, redshift in enumerate(z):
            if not np.any(self.solver.solve_rte[popid]):
                Ja[i] = self.solver.LymanAlphaFlux(redshift, popid=popid)
                if self.pf['feedback_LW']:
                    Jlw[i] = self.solver.LymanWernerFlux(redshift, popid=popid)

                continue
            elif self.pf['secondary_lya'] and (self.pops[popid].is_ion_src_igm):
                Ja[i] = self.solver.volume.SecondaryLymanAlphaFlux(redshift, 
                    popid=popid, fluxes={popid:flux[i]})

            # Convert to energy units, and per eV to prep for integral
            LW_flux = flux[i,is_LW] * E[is_LW] * erg_per_ev / ev_per_hz

            Jlw[i] = np.trapz(LW_flux, x=E[is_LW]) / dnu

        return z, Ja, Jlw

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
    
    
    
    