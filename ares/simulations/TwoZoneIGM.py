"""

TwoZoneIGM.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 16 12:46:28 MST 2015

Description: 

"""

import numpy as np
from .GasParcel import GasParcel
from ..util.ReadData import _sort_data
from ..util import ParameterFile, ProgressBar
from ..populations import CompositePopulation
#from .MetaGalacticBackground import MetaGalacticBackground

igm_pars = \
{
 'grid_cells': 1,
 'isothermal': False,
 'expansion': True,
 'initial_ionization': [1.-1.2e-3, 1.2e-3],
 'cosmological_ics': True,
}

cgm_pars = \
{
 'grid_cells': 1,
 'isothermal': True,
 'initial_ionization': [1. - 1e-8, 1e-8, ],
 'initial_temperature': 1e4,
 'expansion': True,
 'cosmological_ics': True,
}

class TwoZoneIGM:
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
        # Initialize two GasParcels
        self.kw_igm = self.pf.copy()
        self.kw_igm.update(igm_pars)
        
        self.kw_cgm = self.pf.copy()
        self.kw_cgm.update(cgm_pars)        
        
        self.parcel_igm = GasParcel(**self.kw_igm)
        self.parcel_cgm = GasParcel(**self.kw_cgm)
        
        # Fix CGM parcel to have negligible volume filling factor
        self.parcel_cgm.grid.data['Tk'] = 1e4

        # Initialize generators
        self.gen_igm = self.parcel_igm.step()
        self.gen_cgm = self.parcel_cgm.step()
        
        # Initialize radiation backgrounds?
        self.mgb = MetaGalacticBackground(**self.pf)
            
    def run(self):
        """
        Run simulation from start to finish.
        
        Returns
        -------
        Nothing: sets `history` attribute.
        
        """
        
        t = 0.0
        dt = self.pf['initial_timestep'] * self.pf['time_units']
        
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
        zfl = self.pf['first_light_redshift']
        tf = self.parcel_igm.grid.cosm.LookbackTime(zf, z)
        
        dz = dt / self.parcel_igm.grid.cosm.dtdz(z)
        
        pb = ProgressBar(tf, use=self.pf['progress_bar'])
        pb.start()
        
        # Rate coefficients for initial conditions
        self.parcel_igm.set_rate_coefficients(self.parcel_igm.grid.data)
        #self.parcel._set_backgrounds()

        all_t = []
        all_z = []
        all_data = []
        for t, dt, data_igm in self.gen_igm:
            
            dtdz = self.parcel_igm.grid.cosm.dtdz(z)
            z -= dt / dtdz
            
            # Re-compute rate coefficients
            self.parcel_igm.set_rate_coefficients(data_igm)
                        
            self.update_backgrounds()            
                        
            # Now, update CGM parcel
            t2, dt2, data_cgm = self.gen_cgm.next() 
            
            if t >= tf:
                break

            pb.update(t)

            # Save data
            all_z.append(z)
            all_t.append(t)
            all_data.append(data.copy())  
            
        pb.finish()          

        self.history = _sort_data(all_data)
        self.history['t'] = np.array(all_t)
        self.history['z'] = np.array(all_z)    
        
    def step(self, t, dt):
        pass    
    
    def update_backgrounds(self):
        """
        Compute ionization and heating rates.
        """
        
        return
        
        # If doing fancy backgrounds, compute them now
        if self.mgb.approx_all_xrb:
            pass
        else:
            raise NotImplemented('no fancy backgrounds yet!')
        
        if self.mgb.approx_all_lwb:
            pass
        else:
            raise NotImplemented('no fancy backgrounds yet!')
            
        


        