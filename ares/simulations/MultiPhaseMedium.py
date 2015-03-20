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

defaults = \
{
 'load_ics': True,
}

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
}

class MultiPhaseMedium:
    def __init__(self, **kwargs):
        """
        Initialize a MultiPhaseMedium object.
        """

        kwargs.update(defaults)
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
                        'initial_temperature': Ti,
                        'initial_ionization': [1. - xi, xi]}
            igm_pars.update(new_pars)

        # Initialize two GasParcels
        self.kw_igm = self.pf.copy()
        self.kw_igm.update(igm_pars)

        self.kw_cgm = self.pf.copy()
        self.kw_cgm.update(cgm_pars)
                    
        self.parcel_igm = GasParcel(**self.kw_igm)
        self.parcel_cgm = GasParcel(**self.kw_cgm)
        
        self._model_specific_patches()

        # Reset!
        self.parcel_cgm._set_chemistry()
        #del self.parcel_cgm._rate_coefficients
        
        # Initialize generators
        self.gen_igm = self.parcel_igm.step()
        self.gen_cgm = self.parcel_cgm.step()
    
        # Intialize radiation background
        self.field = MetaGalacticBackground(grid=self.parcel_igm.grid, 
            **self.pf)
        
        # Set initial values for rate coefficients
        self.parcel_igm.update_rate_coefficients(self.parcel_igm.grid.data, 
            **self.field.volume.rates_no_RT)
        self.parcel_cgm.update_rate_coefficients(self.parcel_cgm.grid.data, 
            **self.field.volume.rates_no_RT)        
    
        self._insert_inits()

    def _model_specific_patches(self):
        """
        A few modifications to parameter file required by this formalism.
        """

        # Reset stop time based on final redshift.
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
        self.tf = self.parcel_igm.grid.cosm.LookbackTime(zf, z)
        self.pf['stop_time'] = self.tf / self.pf['time_units']
        self.parcel_igm.pf['stop_time'] = self.pf['stop_time']
        self.parcel_cgm.pf['stop_time'] = self.pf['stop_time']

        # Fix CGM parcel 
        self.parcel_cgm.grid.data['Tk'] = np.array([1e4])
        self.parcel_cgm.grid.set_recombination_rate(in_bubbles=True)

    def update_background(self):
        pass
        
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
            self.all_data_igm.append(data_igm.copy())  
            self.all_data_cgm.append(data_cgm.copy())
            
            if self.pf['save_rate_coefficients']:
                self.all_RCs_igm.append(RC_igm.copy())  
                self.all_RCs_cgm.append(RC_cgm.copy())

        pb.finish()          

        # Sort everything by time
        self.history_igm = \
            _sort_history(self.all_data_igm, prefix='igm_', squeeze=True)
        self.history_cgm = \
            _sort_history(self.all_data_cgm, prefix='cgm_', squeeze=True)        
        
        self.history = self.history_igm.copy()
        self.history.update(self.history_cgm)
        
        # Save rate coefficients [optional]
        if self.pf['save_rate_coefficients']:
            self.rates_igm = \
                _sort_history(self.all_RCs_igm, prefix='igm_', squeeze=True)
            self.rates_cgm = \
                _sort_history(self.all_RCs_cgm, prefix='cgm_', squeeze=True)
                
            self.history.update(self.rates_igm)
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
        data_igm = self.parcel_igm.grid.data.copy()
        data_cgm = self.parcel_cgm.grid.data.copy()

        # Evolve in time!        
        while z > zf:

            # Increment time / redshift
            dtdz = self.parcel_igm.grid.cosm.dtdz(z)
            t += dt
            z -= dt / dtdz

            # IGM rate coefficients
            RC_igm = self.field.update_rate_coefficients(z, 
                zone='igm', return_rc=True, igm_h_1=data_igm['h_1'])

            # Now, update IGM parcel
            t1, dt1, data_igm = self.gen_igm.next()

            # Re-compute rate coefficients
            self.parcel_igm.update_rate_coefficients(data_igm, **RC_igm)

            # CGM rate coefficients
            RC_cgm = self.field.update_rate_coefficients(z, 
                zone='cgm', return_rc=True, igm_h_1=data_igm['h_1'])

            # Re-compute rate coefficients
            self.parcel_cgm.update_rate_coefficients(data_cgm, **RC_cgm)

            # Now, update CGM parcel
            t2, dt2, data_cgm = self.gen_cgm.next()

            # Must update timesteps in unison
            dt = min(dt1, dt2)
            self.parcel_igm.dt = dt
            self.parcel_cgm.dt = dt

            yield t, z, data_igm, data_cgm, RC_igm, RC_cgm

    def _insert_inits(self):
        """
        Prepend provided initial conditions to the data storage lists.
        """
        
        if not self.pf['load_ics']:
            self.all_t, self.all_z, self.all_data_igm, self.all_data_cgm = \
                [], [], [], []
            if self.pf['save_rate_coefficients']:    
                self.all_RCs_igm, self.all_RCs_cgm = [], []
            return
            
        # Flip to descending order (in redshift)
        z_inits = self.inits['z'][-1::-1]
        Tk_inits = self.inits['Tk'][-1::-1]
        xe_inits = self.inits['xe'][-1::-1]
        inits_all = {'z': z_inits, 'Tk': Tk_inits, 'xe': xe_inits}
        
        # Stop pre-pending once we hit the first light redshift
        i_trunc = np.argmin(np.abs(z_inits - self.pf['first_light_redshift']))    
        if z_inits[i_trunc] <= self.pf['first_light_redshift']:
            i_trunc += 1

        self.all_t = []
        self.all_data_igm = []
        self.all_z = list(z_inits[0:i_trunc])
        self.all_RCs_igm = [self.field.volume.rates_no_RT] * len(self.all_z)
        self.all_RCs_cgm = [self.field.volume.rates_no_RT] * len(self.all_z)
        
        # Don't mess with the CGM (much)
        tmp = self.parcel_cgm.grid.data
        self.all_data_cgm = [tmp.copy() for i in range(len(self.all_z))]
        for i, cgm_data in enumerate(self.all_data_cgm):
            self.all_data_cgm[i]['rho'] = \
                self.parcel_igm.grid.cosm.MeanBaryonDensity(self.all_z[i])
            self.all_data_cgm[i]['n'] = \
                self.parcel_cgm.grid.particle_density(cgm_data, self.all_z[i])
        
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
            snapshot['rho'] = self.parcel_igm.grid.cosm.MeanBaryonDensity(red)
            snapshot['n'] = \
                self.parcel_igm.grid.particle_density(snapshot.copy(), red)

            self.all_t.append(np.zeros_like(self.all_z))
            self.all_data_igm.append(snapshot.copy())


            