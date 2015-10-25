"""

MultiPhaseMedium.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:46:28 MST 2015

Description: 

"""

import numpy as np
from .GasParcel import GasParcel
from ..util import ParameterFile, ProgressBar
from ..util.ReadData import _sort_history, _load_inits
from .MetaGalacticBackground import MetaGalacticBackground
from ..util.SetDefaultParameterValues import MultiPhaseParameters

# These should go in ProblemTypes?
igm_pars = \
{
 'grid_cells': 1,
 'isothermal': False,
 'expansion': True,
 'initial_ionization': [1.-1.2e-3, 1.2e-3],
 'cosmological_ics': False,
}

cgm_pars = \
{
 'grid_cells': 1,
 'isothermal': True,
 'initial_ionization': [1. - 1e-8, 1e-8],
 'initial_temperature': [1e4],
 'expansion': True,
 'cosmological_ics': True,
 'recombination': 'A',
 'include_He': False,
}

_mpm_defs = MultiPhaseParameters()

class MultiPhaseMedium(object):
    def __init__(self, **kwargs):
        """
        Initialize a MultiPhaseMedium object.
        
        By default, this is a two-zone model, consisting of a  "bulk IGM"
        grid patch and an "HII regions" grid patch, dubbed "igm" and "cgm", 
        respectively. To perform a single-zone calculation, simply set 
        ``include_cgm=False``.
        """

        if 'load_ics' not in kwargs:
            kwargs['load_ics'] = True

        self.pf = ParameterFile(**kwargs)
                
        # Load in initial conditions, interpolate to initial_redshift
        if self.pf['load_ics']:
            
            inits = self.inits = _load_inits()

            zi = self.pf['initial_redshift']
            if not np.all(np.diff(inits['z']) > 0):
                raise ValueError('Redshifts in ICs must be in ascending order!')
                
            Ti = np.interp(zi, inits['z'], inits['Tk'])
            xi = np.interp(zi, inits['z'], inits['xe'])
            new_pars = {'cosmological_ics': False,
                        'igm_initial_temperature': Ti,
                        'igm_initial_ionization': [1. - xi, xi]}
            kwargs.update(new_pars)

            #if self.pf['include_He']:
            #    igm_pars.update({'include_He': True,
            #        'initial_ionization': [1. - xi, xi, 1.-xi, xi, 1e-10]})         

        self._initialize_zones(**kwargs)
        self._insert_inits()

    def _initialize_zones(self, **kwargs):
        """
        Initialize (up to two) GasParcels.
        """
        
        # Reset stop time based on final redshift.
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
                
        self.parcels = []
        for zone in ['igm', 'cgm']:
            if not self.pf['include_%s' % zone]:
                continue
                
            kw = self.pf.copy()
            
            # Loop over defaults, pull out the ones for this zone                
            for key in _mpm_defs:
                if key[0:4] != '%s_' % zone:
                    continue

                # Have to rename variables so Grid class will know them
                grid_key = key.replace('%s_' % zone, '')
                
                if key in kwargs:
                    kw[grid_key] = kwargs[key]
                else:
                    kw[grid_key] = _mpm_defs[key]
                            
            if zone == 'igm':
                self.kw_igm = kw.copy()
                self.parcel_igm = GasParcel(**self.kw_igm)
                
                self.gen_igm = self.parcel_igm.step()

                self.field = MetaGalacticBackground(grid=self.parcel_igm.grid, 
                    **kwargs)

                # Set initial values for rate coefficients
                self.parcel_igm.update_rate_coefficients(self.parcel_igm.grid.data, 
                    **self.field.volume.rates_no_RT)
                    
                self.parcels.append(self.parcel_igm)
                    
            else:
                self.kw_cgm = kw.copy()
                self.parcel_cgm = GasParcel(**self.kw_cgm)
                self.parcel_cgm.grid.set_recombination_rate(True)
                self.parcel_cgm._set_chemistry()
                self.gen_cgm = self.parcel_cgm.step()
                
                self.parcel_cgm.chem.chemnet.monotonic_EoR = \
                    self.pf['monotonic_EoR']
                
                if not hasattr(self, 'field'):
                    self.field = MetaGalacticBackground(grid=self.parcel_cgm.grid, 
                        **kwargs)

                self.parcel_cgm.update_rate_coefficients(self.parcel_cgm.grid.data, 
                    **self.field.volume.rates_no_RT)
                
                self.parcels.append(self.parcel_cgm)
                        
            if not hasattr(self, 'tf'):
                self.tf = self.default_parcel.grid.cosm.LookbackTime(zf, z)
                self.pf['stop_time'] = self.tf / self.pf['time_units']    

            self.parcels[-1].pf['stop_time'] = self.pf['stop_time']
             
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
    def zones(self):
        if not hasattr(self, '_zones'):
            self._zones = int(self.pf['include_igm']) \
                + int(self.pf['include_cgm'])
        
        return self._zones

    @property
    def default_parcel(self):
        if not hasattr(self, '_default_parcel'):
            self._default_parcel = self.parcel_igm if self.pf['include_igm'] \
                else self.parcel_cgm
                
        return self._default_parcel

    @property
    def dynamic_tau(self):
        return self.pf['tau_dynamic']

    def update_optical_depth(self):
        """
        Dynamically update optical depth as simulation runs.
        """
        
        # Recall that self.field.tau is a list with as many elements as there
        # are distinct populations
        
        
        tau = []
        for i in range(self.field.Npops):
            pass
            
        
        self.field.tau = tau
        

    def subcycle(self):
        """
        See if we need to re-do the previous timestep.
        
        This mean:
            (1) Re-compute the IGM optical depth.
            (2)
        """

        return False

        # Check IGM ionization state between last two steps. 
        # Converged to desired tolerance?
        
        #self.
        
        
    def run(self):
        """
        Run simulation from start to finish.

        Returns
        -------
        Nothing: sets `history` attribute.

        """

        pb = ProgressBar(self.tf, use=self.pf['progress_bar'])
        pb.start()
                                    
        # Evolve in time
        for t, z, data_igm, data_cgm, RC_igm, RC_cgm in self.step():
            
            pb.update(t)
                        
            # Save data
            self.all_z.append(z)
            self.all_t.append(t)
            
            if self.pf['include_cgm']:    
                self.all_data_cgm.append(data_cgm.copy())
            
            if self.pf['include_igm']:
                self.all_data_igm.append(data_igm.copy())  
                
            if self.pf['save_rate_coefficients']:
                if self.pf['include_cgm']:     
                    self.all_RCs_cgm.append(RC_cgm.copy())
                if self.pf['include_igm']:
                    self.all_RCs_igm.append(RC_igm.copy())

        pb.finish()          

        # Sort everything by time
        if self.pf['include_igm']:
            self.history_igm = \
                _sort_history(self.all_data_igm, prefix='igm_', squeeze=True)
            self.history = self.history_igm.copy()
        else:
            self.history = {}
            
        if self.pf['include_cgm']:    
            self.history_cgm = \
                _sort_history(self.all_data_cgm, prefix='cgm_', squeeze=True)        
            self.history.update(self.history_cgm)

        # Save rate coefficients [optional]
        if self.pf['save_rate_coefficients']:
            if self.pf['include_igm']:
                self.rates_igm = \
                    _sort_history(self.all_RCs_igm, prefix='igm_', squeeze=True)
                self.history.update(self.rates_igm)
            
            if self.pf['include_cgm']:    
                self.rates_cgm = \
                    _sort_history(self.all_RCs_cgm, prefix='cgm_', squeeze=True)
                self.history.update(self.rates_cgm)

        self.history['t'] = np.array(self.all_t)
        self.history['z'] = np.array(self.all_z)
        
    def step(self):
        """
        Generator for a two-phase intergalactic medium.
        
        Returns
        -------
        Tuple containing the current time, redshift, and dictionaries for the
        IGM and CGM data at a single snapshot.
        
        """

        t = 0.0
        z = self.pf['initial_redshift']
        dt = self.pf['time_units'] * self.pf['initial_timestep']
        zf = self.pf['final_redshift']
        
        # Read initial conditions
        if self.pf['include_igm']:  
            data_igm = self.parcel_igm.grid.data.copy()
        
        if self.pf['include_cgm']:
            data_cgm = self.parcel_cgm.grid.data.copy()

        # Evolve in time!
        while z > zf:

            # Increment time / redshift
            dtdz = self.default_parcel.grid.cosm.dtdz(z)
            t += dt
            z -= dt / dtdz
                        
            # The (potential) generators need this
            self.field.update_redshift(z)
                        
            # IGM rate coefficients
            if self.pf['include_igm']:
                RC_igm = self.field.update_rate_coefficients(z, 
                    zone='igm', return_rc=True, igm_h_1=data_igm['h_1'])
                                                                
                # Now, update IGM parcel
                t1, dt1, data_igm = self.gen_igm.next()
                                
                # Pass rate coefficients off to the IGM parcel
                self.parcel_igm.update_rate_coefficients(data_igm, **RC_igm)
            else:
                dt1 = 1e50
                RC_igm = data_igm = None
                data_igm = {'h_1': 1.0}
                
            if self.pf['include_cgm']:
                # CGM rate coefficients
                RC_cgm = self.field.update_rate_coefficients(z,
                    zone='cgm', return_rc=True, cgm_h_1=data_cgm['h_1'])
                
                # Pass rate coefficients off to the CGM parcel
                self.parcel_cgm.update_rate_coefficients(data_cgm, **RC_cgm)
                
                # Now, update CGM parcel
                t2, dt2, data_cgm = self.gen_cgm.next()
            else:
                dt2 = 1e50
                RC_cgm = data_cgm = None

            # Must update timesteps in unison
            dt_pre = dt * 1.
            dt = min(dt1, dt2)

            # Might need these...
            if self.pf['include_igm']:
                data_igm_pre = data_igm.copy()
            if self.pf['include_cgm']:    
                data_cgm_pre = data_cgm.copy()

            # If we're computing the IGM optical depth dynamically, we may
            # need to "re-do" this step to ensure convergence.

            redo = self.subcycle()
            
            if not redo:    
                
                # Changing attribute! A little scary, but we must make sure
                # these parcels are evolved in unison
                if self.pf['include_igm']:
                    self.parcel_igm.dt = dt
                if self.pf['include_cgm']:
                    self.parcel_cgm.dt = dt

                yield t, z, data_igm, data_cgm, RC_igm, RC_cgm
                
                continue

            # If we've made it here, we need to trick our generators a bit
            
            # "undo" this time-step
            t -= dt_pre
            z += dt_pre / dtdz

            self.update_optical_depth()

    def _insert_inits(self):
        """
        Prepend provided initial conditions to the data storage lists.
        """
        
        if not self.pf['load_ics']:
            self.all_t, self.all_z, self.all_data_igm, self.all_data_cgm = \
                [], [], [], []
            if self.pf['save_rate_coefficients']:    
                self.all_RCs_igm, self.all_RCs_cgm = [], []
                
            if not self.pf['include_cgm']:
                del self.all_RCs_cgm, self.all_data_cgm    
            return
            
        # Flip to descending order (in redshift)
        z_inits = self.inits['z'][-1::-1]
        Tk_inits = self.inits['Tk'][-1::-1]
        xe_inits = self.inits['xe'][-1::-1]
        inits_all = {'z': z_inits, 'Tk': Tk_inits, 'xe': xe_inits}

        # Stop pre-pending once we hit the first light redshift
        i_trunc = np.argmin(np.abs(z_inits - self.pf['initial_redshift']))    
        if z_inits[i_trunc] <= self.pf['initial_redshift']:
            i_trunc += 1

        self.all_t = []
        self.all_data_igm = []
        self.all_z = list(z_inits[0:i_trunc])
        self.all_RCs_igm = [self.field.volume.rates_no_RT] * len(self.all_z)
        self.all_RCs_cgm = [self.field.volume.rates_no_RT] * len(self.all_z)

        # Don't mess with the CGM (much)
        if self.pf['include_cgm']:
            tmp = self.parcel_cgm.grid.data
            self.all_data_cgm = [tmp.copy() for i in range(len(self.all_z))]
            for i, cgm_data in enumerate(self.all_data_cgm):
                self.all_data_cgm[i]['rho'] = \
                    self.parcel_cgm.grid.cosm.MeanBaryonDensity(self.all_z[i])
                
                self.all_data_cgm[i]['n'] = \
                    self.parcel_cgm.grid.particle_density(cgm_data, self.all_z[i])
        
        if not self.pf['include_igm']:
            return
        
        # Loop over redshift and derive things for the IGM
        for i, red in enumerate(self.all_z): 

            snapshot = {}
            for key in self.parcel_igm.grid.data.keys():
                if key in self.inits.keys():
                    snapshot[key] = inits_all[key][i]
                    continue
            
            # Electron fraction
            snapshot['e'] = inits_all['xe'][i]
  
            # Hydrogen neutral fraction
            xe = inits_all['xe'][i]
            
            if 2 not in self.parcel_igm.grid.Z:
                xe = min(xe, 1.0)
                
            xi = xe / (1. + self.parcel_igm.grid.cosm.y)

            snapshot['h_1'] = 1. - xi
            snapshot['h_2'] = xi
            
            # Add helium, assuming xHeII = xHII, and xHeIII << 1
            if self.parcel_igm.pf['include_He']:
                snapshot['he_1'] = 1. - xi
                snapshot['he_2'] = xi
                snapshot['he_3'] = 1e-10
                
            snapshot['rho'] = self.parcel_igm.grid.cosm.MeanBaryonDensity(red)
            snapshot['n'] = \
                self.parcel_igm.grid.particle_density(snapshot.copy(), red)

            self.all_t.append(np.zeros_like(self.all_z))
            self.all_data_igm.append(snapshot.copy())


            