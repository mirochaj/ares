"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description: 

"""

import numpy as np
from ..static import Grid
import copy, os, re, pickle
from ..sources import DiffuseSource
from ..util.Misc import parse_kwargs
from ..util.ReadData import load_inits
from ..util.WriteData import CheckPoints
from ..util.ManageHistory import WriteData
from ..util.PrintInfo import print_21cm_sim
from ..populations import CompositePopulation
from ..util import ProgressBar, RestrictTimestep
from ..solvers.RadiationField import RadiationField
from ..solvers.UniformBackground import UniformBackground
from ..util.SetDefaultParameterValues import SetAllDefaults
from ..physics.Constants import k_B, m_p, G, g_per_msun, c, sigma_T, \
    erg_per_ev, nu_0_mhz
    
try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False
    
try:
    from scipy.interpolate import interp1d
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1   
    
HOME = os.environ.get('HOME')
ARES = os.environ.get('ARES')

class Global21cm:
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.

        See Also
        --------
        Set of all acceptable kwargs in:
            ares/util/SetDefaultParameterValues.py
        
        If you'd prefer not to run from recombination each time, make high-z
        initial conditions once and read-in from then on:
            ares/input/generate_initial_conditions.py
        
        If you'd prefer to generate lookup tables for the halo mass function,
        rather than re-calculating each time, run:
            ares/input/generate_hmf_tables.py
            
        Same deal for IGM optical depth:
            ares/input/generate_optical_depth_tables.py
            
        """
        
        if kwargs:
            
            if 'tanh_model' not in kwargs:
                self.pf = parse_kwargs(**kwargs)
            else:
                if kwargs['tanh_model']:
                    from ..util.TanhModel import TanhModel

                    tanh_model = TanhModel(**kwargs)
                    self.pf = tanh_model.pf
                    
                    if self.pf['tanh_nu'] is not None:
                        nu = self.pf['tanh_nu']
                        z = nu_0_mhz / nu - 1.
                    else:
                        z = np.arange(self.pf['final_redshift'] + self.pf['tanh_dz'],
                            self.pf['initial_redshift'], self.pf['tanh_dz'])[-1::-1]
                    
                    self.history = tanh_model(z, **self.pf).data
                    
                    if self.pf['inline_analysis'] is not None:
                        self.run_inline_analysis()

                    return
                    
                else:
                    self.pf = parse_kwargs(**kwargs)

        else:
            self.pf = SetAllDefaults()
                    
        # Check for identical realization
        self.found_sim = False
        if self.pf['load_sim'] and os.path.exists('%s/.ares' % HOME):
            for fn in os.listdir('%s/.ares' % HOME):
                if not re.search('.pkl', fn):
                    continue
                
                f = open('%s/.ares/%s' % (HOME, fn), 'rb')
                pf = pickle.load(f)
                f.close()
                
                if pf == self.pf:
                    break
            
            self.found_sim = True
            
            prefix = fn.partition('.')[0]
            self.history = dict(np.load('%s/.ares/%s.npz' % (HOME, prefix)))
            
            if rank == 0:
                print "\nFound identical realization! Loaded %s/.ares/%s.npz" \
                    % (HOME, prefix)
            return 
            
        self._check_for_conflicts()

        self.fullcalc = (self.pf['approx_xray'] + self.pf['approx_lya']) < 2
            
        # Initialize two grid patches   
        self.grid_igm = Grid(dims=1, approx_Salpha=self.pf['approx_Salpha'])
        self.grid_cgm = Grid(dims=1)
        self.grids = [self.grid_igm, self.grid_cgm]
            
        # If we just need access to cosmology, use igm grid (arbitrarily)
        self.grid = self.grid_igm    
            
        # Set physics
        self.grid_igm.set_physics(isothermal=0, compton_scattering=1, 
            secondary_ionization=0, expansion=1, recombination='B')
        self.grid_cgm.set_physics(isothermal=1, compton_scattering=0, 
            secondary_ionization=0, expansion=1, recombination='A',
            clumping_factor=self.pf['clumping_factor'])        
            
        # Read in initial conditions (if they exist)  
        if self.pf['initial_temperature'] is None:  
            T0 = self.grid.cosm.TCMB(self.pf['initial_redshift'])
        else:
            T0 = self.pf['initial_temperature']
            
        x0 = self.pf['initial_ionization']
        if ARES and self.pf['load_ics']:
            if have_h5py:
                inits_path = '%s/input/inits/initial_conditions.hdf5' % ARES
            else:
                inits_path = '%s/input/inits/initial_conditions.npz' % ARES
                
            if os.path.exists(inits_path):
                inits = self.inits = load_inits(inits_path)
                
                T0 = np.interp(self.pf['initial_redshift'], inits['z'],
                    inits['Tk'])
                xe = np.interp(self.pf['initial_redshift'], inits['z'],
                    inits['xe'])
                   
                if len(self.pf['Z']) > 1:
                    x0 = [xe, 0.0]
                else:
                    x0 = [min(xe, 1.0)]

                self.inits_path = inits_path
        else:
            if len(x0) != len(self.pf['Z']):
                x0 = [self.pf['initial_ionization'][0]] * 2 
                            
        # Set cosmological initial conditions  
        for grid in self.grids:  
            grid.set_cosmology(initial_redshift=self.pf['initial_redshift'], 
                omega_m_0=self.pf["omega_m_0"], 
                omega_l_0=self.pf["omega_l_0"], 
                omega_b_0=self.pf["omega_b_0"], 
                hubble_0=self.pf["hubble_0"], 
                helium_by_number=self.pf['helium_by_number'], 
                cmb_temp_0=self.pf["cmb_temp_0"], 
                approx_highz=self.pf["approx_highz"])
                
            grid.set_chemistry(Z=self.pf['Z'])
            grid.set_density(grid.cosm.rho_b_z0 \
                * (1. + self.pf['initial_redshift'])**3)

        self.helium = 2 in self.pf['Z']

        self.grid_igm.set_temperature(T0)
        self.grid_cgm.set_temperature(1.e4)
        self.grid_cgm.set_recombination_rate(in_bubbles=True)

        for i, Z in enumerate(self.pf['Z']):
            self.grid_igm.set_ionization(Z=Z, x=x0[i])
            self.grid_cgm.set_ionization(Z=Z, x=1e-8)
                        
        self.grid_igm.data['n'] = \
            self.grid_igm.particle_density(self.grid_igm.data, 
            z=self.pf['initial_redshift'])
        self.grid_cgm.data['n'] = \
            self.grid_cgm.particle_density(self.grid_cgm.data, 
            z=self.pf['initial_redshift'])

        # To compute timestep
        self.timestep_igm = RestrictTimestep(self.grid_igm, 
            self.pf['epsilon_dt'])
        self.timestep_cgm = RestrictTimestep(self.grid_cgm, 
            self.pf['epsilon_dt'])    
            
        # For regulating time/redshift steps
        self.checkpoints = CheckPoints(pf=self.pf, 
            grid=self.grid,
            time_units=self.pf['time_units'],
            initial_redshift=self.pf['initial_redshift'],
            final_redshift=self.pf['final_redshift'],
            dzDataDump=self.pf['dzDataDump'],
            dtDataDump=self.pf['dtDataDump'],
            )
            
        ##
        # PRINT STUFF!
        ##
        print_21cm_sim(self)    

        # Initialize radiation sources / populations
        if self.pf["radiative_transfer"]:
            self._init_RT(self.pf)
        else:
            self.approx_all_xray = 1  # will be updated in _init_RT
            self.srcs = None
            
        if type(self.pf['feedback']) in [bool, int]:
            try:
                self.feedback = [self.pf['feedback']] * len(self.pops.pops)    
            except AttributeError:
                self.feedback = [self.pf['feedback']]
        else:
            self.feedback = self.pf['feedback']
            
        self.feedback_ON = sum(self.feedback) > 0    
        
        # Initialize radiative transfer solver
        self.rt_igm = RadiationField(self.grid_igm, self.srcs, **self.pf)
        self.rt_cgm = RadiationField(self.grid_cgm, self.srcs, **self.pf)
            
        self.rt_cgm.chem.chemnet.SourceIndependentCoefficients(T=1.e4)    
            
        # Set up X-ray flux generator
        if not self.approx_all_xray:
            if self.pf['EoR_xavg'] == 0:
                raise NotImplemented('This needs work (EoR_xavg == 0)')
                self._init_XRB(pre_EoR=True)
                self._init_XRB(pre_EoR=False)
            else:
                self._init_XRB()
            
        if self.pf['track_extrema']:
            from ..analysis.TurningPoints import TurningPoints
            self.track = TurningPoints(inline=True, **self.pf)    
        
        # should raise error if different tau tables passed to each source.    
    
        self.write = WriteData(self)
    
    def _init_RT(self, pf, use_tab=True):
        """
        Initialize astrophysical populations & radiation backgrounds.
        """
        
        self.pops = CompositePopulation(**pf)

        if len(self.pops.pops) == 1:
            self.pop = self.pops.pops[0]
                        
        # Loop over populations, make separate RB and RS instances for each
        self.rbs = [UniformBackground(pop) for pop in self.pops.pops]
        
        self.Nrbs = len(self.rbs)
        
        self.approx_all_xray = 1
        
        for rb in self.rbs:
            self.approx_all_xray *= rb.pf['approx_xray']
        
        # Don't have to do this...could just stick necessary attributes in RB
        self.srcs = [DiffuseSource(rb) for rb in self.rbs]

    def _init_XRB(self, pre_EoR=True, **kwargs):
        """ Setup cosmic X-ray background calculation. """
                
        if pre_EoR:
            self.pre_EoR = True
            
            # Store XrayFluxGenerators
            self.cxrb_gen = [None for i in range(self.Nrbs)]
            for i, rb in enumerate(self.rbs):
                if rb.pf['approx_xray']:
                    continue

                self.cxrb_gen[i] = rb.XrayFluxGenerator(rb.igm.tau)

                # All UniformBackgrounds must share these properties
                if not hasattr(self, 'cxrb_shape'):
                    self.cxrb_shape = (rb.igm.L, rb.igm.N)
                if not hasattr(self, 'zmin_igm'):
                    self.zmin_igm = rb.igm.z[0]
                    self.cxrb_zall = rb.igm.z.copy()

            # Save X-ray background incrementally
            self.xray_flux = [[] for i in range(self.Nrbs)]
            self.xray_heat = [[] for i in range(self.Nrbs)]
                
            # Generate fluxes at first two redshifts
            fluxes_lo = []; fluxes_hi = []
            for cxrb in self.cxrb_gen:
                                
                if cxrb is None:
                    fluxes_hi.append(0.0)
                    continue
            
                fluxes_hi.append(cxrb.next())  # this line halting parallel calculations
            
            for cxrb in self.cxrb_gen:
                if cxrb is None:
                    fluxes_lo.append(0.0)
                    continue
                    
                fluxes_lo.append(cxrb.next())
                    
            self.cxrb_flo = fluxes_lo
            self.cxrb_lhi = self.cxrb_shape[0] - 1
            self.cxrb_llo = self.cxrb_shape[0] - 2
            
            self.cxrb_fhi = fluxes_hi
            
            # Figure out first two redshifts
            for i, rb in enumerate(self.rbs):
                if rb.pop.pf['approx_xray']:
                    continue

                self.cxrb_zhi = rb.igm.z[-1]
                self.cxrb_zlo = rb.igm.z[-2]
                
                break
            
            # Store first two fluxes, separate by rb instance    
            for i, rb in enumerate(self.rbs):
                self.xray_flux[i].extend([fluxes_hi[i], fluxes_lo[i]])
        else:
            for i, rb in enumerate(self.rbs):
                if rb.pf['approx_xray']:
                    continue
                    
                self.cxrb_shape = (rb.igm.L, rb.igm.N)     
                
                self.zmin_igm = rb.igm.z[0]
                self.cxrb_zall = rb.igm.z.copy()
                
                self.cxrb_zhi = rb.igm.z[-1]
                self.cxrb_zlo = rb.igm.z[-2]
                self.cxrb_lhi = self.cxrb_shape[0] - 1
                self.cxrb_llo = self.cxrb_shape[0] - 2
                
                # Done in ComputeXRB upon switch to EoR
                #self.cxrb_flo = fluxes_lo
                #self.cxrb_fhi = fluxes_hi
                
                break
                
            # Save X-ray background incrementally
            if not hasattr(self, 'xray_flux'):
                self.xray_flux = [[] for i in range(self.Nrbs)]
            if not hasattr(self, 'xray_heat'):
                self.xray_heat = [[] for i in range(self.Nrbs)]            

        # Store rate coefficients by (source, absorber)
        self.cxrb_hlo = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G1lo = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G2lo = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_hhi = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G1hi = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G2hi = np.zeros([self.Nrbs, self.grid.N_absorbers])
        
        for i, rb in enumerate(self.rbs):
            
            self.xray_heat[i].extend([self.cxrb_hhi[i], self.cxrb_hlo[i]])
            
            for j, absorber in enumerate(self.grid.absorbers):
            
                if rb.pop.pf['approx_xray']:
                    self.cxrb_hlo[i,j] = \
                        rb.igm.HeatingRate(self.cxrb_zlo, return_rc=True, **kwargs)
                    self.cxrb_hhi[i,j] = \
                        rb.igm.HeatingRate(self.cxrb_zhi, return_rc=True, **kwargs)
                    self.cxrb_G1lo[i,j] = \
                        rb.igm.IonizationRateIGM(self.cxrb_zlo, return_rc=True, **kwargs)
                    self.cxrb_G1hi[i,j] = \
                        rb.igm.IonizationRateIGM(self.cxrb_zhi, return_rc=True, **kwargs)
                    self.cxrb_G2lo[i,j] = \
                        rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo, return_rc=True, **kwargs)
                    self.cxrb_G2hi[i,j] = \
                        rb.igm.SecondaryIonizationRateIGM(self.cxrb_zhi, return_rc=True, **kwargs)
                    
                    continue
                    
                # Otherwise, compute heating etc. from background intensity    
                self.cxrb_hlo[i,j] = \
                    rb.igm.HeatingRate(self.cxrb_zlo, species=j,
                    xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                self.cxrb_hhi[i,j] = \
                    rb.igm.HeatingRate(self.cxrb_zhi, species=j,
                    xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                
                # IGM ionization
                self.cxrb_G1lo[i,j] = \
                    rb.igm.IonizationRateIGM(self.cxrb_zlo, species=j,
                    xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                self.cxrb_G1hi[i,j] = \
                    rb.igm.IonizationRateIGM(self.cxrb_zhi, species=j,
                    xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                                
                if self.pf['secondary_ionization'] > 0:
                    self.cxrb_G2lo[i,j] = \
                        rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo,
                        xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                    self.cxrb_G2hi[i,j] = \
                        rb.igm.SecondaryIonizationRateIGM(self.cxrb_zhi,
                        xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                
    def run(self):
        """Just another pathway to our __call__ method. Runs simulation."""
        self.__call__()
     
    def __call__(self):
        """ Evolve chemistry and radiation background. """
        
        if hasattr(self, 'history'):
            return
        
        t = 0.0
        dt = self.pf['initial_timestep'] * self.pf['time_units']
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
        zfl = self.zfl = self.pf['first_light_redshift']
        
        if self.pf["dzDataDump"] is not None:
            dz = self.pf["dzDataDump"]
        else:
            dz = dt / self.grid.cosm.dtdz(z)
        
        # Create dictionary for output - store initial conditions
        self.history = self.write._initialize_history()
        
        # Read initial conditions
        data_igm = self.grid_igm.data.copy()
        data_cgm = self.grid_cgm.data.copy()

        # Initialize progressbar
        self.tf = self.grid.cosm.LookbackTime(zf, z)
        if self.pf['progress_bar']:
            print ""
        self.pb = ProgressBar(self.tf, '21-cm (pre-EoR)',
            use=self.pf['progress_bar'])
        self.pb.start()

        # Feedback
        self.feedback_last_z = z

        self.step = 0

        fields = ['h_1', 'h_2', 'e']
        if self.helium:
            fields.extend(['he_1', 'he_2', 'he_3'])

        # Evolve to final redshift
        while z > zf:
                        
            # kwargs specific to bulk IGM grid patch
            kwargs = {'igm': True, 'return_rc': True}
            
            # Make temporary arrays in order of ascending redshift 
            # (required by interpolation routines we use)
            ztmp = self.history['z'][-1::-1]
            xtmp = self.history['xavg'][-1::-1]

            # Need to pass ionization state to both grids
            to_rt1d = {}

            # Add all hydrogen (and possibly helium) fractions
            for field in fields:
                to_rt1d['cgm_%s' % field] = self.history['cgm_%s' % field][-1]
                to_rt1d['igm_%s' % field] = self.history['igm_%s' % field][-1]

            # Compute X-ray background flux
            flux_x = self.ComputeXRB(z, ztmp, xtmp, **to_rt1d)
            
            to_rt1d.update({'xray_flux': flux_x})
            kwargs.update(to_rt1d)
            
            # Grab ionization/heating rates from full X-ray background calculation
            if not self.approx_all_xray:
                
                # Sum over sources, but keep sorted by absorbing species
                hlo = np.sum(self.cxrb_hlo, axis=0)
                hhi = np.sum(self.cxrb_hhi, axis=0)
                G1lo = np.sum(self.cxrb_G1lo, axis=0)
                G1hi = np.sum(self.cxrb_G1hi, axis=0)
                                
                # Interpolate to current time
                H = [np.interp(z, [self.cxrb_zlo, self.cxrb_zhi], 
                    [hlo[i], hhi[i]]) for i in range(self.grid.N_absorbers)]
                G1 = [np.interp(z, [self.cxrb_zlo, self.cxrb_zhi], 
                    [G1lo[i], G1hi[i]]) for i in range(self.grid.N_absorbers)]
                
                if self.pf['secondary_ionization'] > 0:
                    G2lo = np.sum(self.cxrb_G2lo, axis=0)
                    G2hi = np.sum(self.cxrb_G2hi, axis=0)
                    
                    G2 = [np.interp(z, [self.cxrb_zlo, self.cxrb_zhi], 
                        [G2lo[i], G2hi[i]]) for i in range(self.grid.N_absorbers)]
                    
                else:
                    G2 = [0.0] * self.grid.N_absorbers
                                                                              
                kwargs.update({'epsilon_X': np.array(H), 
                  'Gamma': np.array(G1), 
                  'gamma': np.array(G2)})

            # Solve for xe and Tk in the bulk IGM
            data_igm = self.rt_igm.Evolve(data_igm, t=t, dt=dt, z=z, **kwargs)

            # Next up: bubbles
            kwargs.update({'igm': False})

            # Gamma etc. are only for the bulk IGM - lose them for bubbles!
            if 'Gamma' in kwargs:
                kwargs.pop('Gamma')
            if 'gamma' in kwargs:
                kwargs.pop('gamma')
            if 'epsilon_X' in kwargs:
                kwargs.pop('epsilon_X')
                
            # Solve for the volume filling factor of HII regions
            if self.pf['radiative_transfer'] and (z <= zfl):
                data_cgm = self.rt_cgm.Evolve(data_cgm, t=t, dt=dt, z=z, **kwargs)

            # Increment time and redshift
            zpre = z
            
            t += dt
            z -= dt / self.grid.cosm.dtdz(z)
            
            # SAVE RESULTS
            self.write._update_history(z, zpre, data_igm, data_cgm)
                        
            if z <= zf:
                break

            ##
            # FEEDBACK: Modify Tmin depending on IGM temperature and/or LW flux
            ##
            #self.Feedback(zpre, data_igm_fl['Tk'], Jlw=Ja)
            
            # Inline analysis: possibly kill calculation if we've passed
            # a turning point, zero-crossing, or critical ionization fraction.
            # See "stop" parameter for more information.             
            if self.pf['track_extrema']:
                stop = self.track.is_stopping_point(self.history['z'],
                    self.history['dTb'])
                if stop:
                    break

            ##
            # TIME-STEPPING FROM HERE ON OUT
            ##                 
                                            
            # Figure out next dt based on max allowed change in evolving fields
            new_dt_igm = \
                self.timestep_igm.Limit(self.rt_igm.chem.q_grid.squeeze(),
                self.rt_igm.chem.dqdt_grid.squeeze(), z=z,
                method=self.pf['restricted_timestep'])
            
            if (z + dt / self.grid.cosm.dtdz(z)) <= self.pf['first_light_redshift'] and \
                self.pf['radiative_transfer']:
                
                new_dt_cgm = \
                    self.timestep_cgm.Limit(self.rt_cgm.chem.q_grid.squeeze(), 
                    self.rt_cgm.chem.dqdt_grid.squeeze(), z=z,
                    method=self.pf['restricted_timestep'])
            else:
                new_dt_cgm = 1e50
                            
            # Limit timestep further based on next DD and max allowed increase
            dt = min(min(new_dt_igm, new_dt_cgm), 2*dt)
            dt = min(dt, self.checkpoints.next_dt(t, dt))
            dt = min(dt, self.pf['max_dt'] * self.pf['time_units'])

            # Limit timestep based on next RD
            if self.checkpoints.redshift_dumps:
                dz = min(self.checkpoints.next_dz(z, dz), self.pf['max_dz'])
                dt = min(dt, dz*self.grid.cosm.dtdz(z))
                
            if self.pf['max_dz'] is not None:
                dt = min(dt, self.pf['max_dz']*self.grid.cosm.dtdz(z))    
                
            # Limit redshift step by next element in flux generator
            if not self.approx_all_xray and (z > self.zmin_igm):                
                dtdz = self.grid.cosm.dtdz(z)
                
                if (z - dt / dtdz) < self.cxrb_zall[max(self.cxrb_llo-1, 0)]:
                    dz = (z - self.cxrb_zall[self.cxrb_llo-1]) * 0.5
                    
                    # Means we're on the last step
                    if dz < 0:
                        dz = z - self.zmin_igm
                                                
                    dt = dz*dtdz
            
            self.pb.update(t)

            # Quit if reionization is ~complete (xavg = 0.9999 by default)
            if self.history['xavg'][-1] >= self.pf['stop_xavg']:            
                break
                
            self.step += 1

        self.pb.finish()

        tmp = {}    
        for key in self.history:
            tmp[key] = np.array(self.history[key])

        self.history = tmp
            
        if self.pf['track_extrema']:
            self.turning_points = self.track.turning_points
        
        if self.pf['inline_analysis'] is not None:
            self.run_inline_analysis()
            
    @property
    def blob_shape(self):
        if not hasattr(self, '_blob_shape'):
            if self.pf['inline_analysis']:
                self._blob_shape = map(len, self.pf['inline_analysis'])[-1::-1]
            else:
                self._blob_shape = None
                
        return self._blob_shape
        
    def run_inline_analysis(self):
        """
        Compute some quantities of interest.
        
        Example
        -------
        sim = ares.simulations.Global21cm(track_extrema=True, 
            inline_analysis=(['dTb'], list('BCD'))
        
        sim.run()
        
        zip(*sim.blobs)[0]  # are the brightness temperatures of B, C, and D
        sim.ztps            # redshifts
        
        """
        
        if not hasattr(self, 'turning_points'):
            
            from ..analysis.TurningPoints import TurningPoints
            self._track = TurningPoints(inline=True, **self.pf)
            
            # Otherwise, find them. Not the most efficient, but it gets the job done
            if self.history['z'].max() < 70 and 'A' not in self._track.TPs:
                self._track.TPs.append('A')
            
            delay = self.pf['stop_delay']
            
            for i in range(len(self.history['z'])):
                if i < (delay + 2):
                    continue
                    
                stop = self._track.is_stopping_point(self.history['z'][i-delay-1:i],
                    self.history['dTb'][i-delay-1:i])
                                    
            self.turning_points = self._track.turning_points
                
        fields, ztmp = self.pf['inline_analysis']
        
        zmin = self.history['z'].min()
        zmax = self.history['z'].max()        
        
        # Convert turning points to actual redshifts
        redshift = []
        ztps = []
        for element in ztmp:
            if type(element) is str:                
                if element not in self.turning_points:
                    redshift.append(np.inf)
                    ztps.append(np.inf)
                else:
                    redshift.append(self.turning_points[element][0])
                    ztps.append((element, self.turning_points[element][0]))
            else:
                redshift.append(element)
            
        # Redshift x blobs (x species)
        output = []
        for j, field in enumerate(fields):
            
            if field in self.history:                
                interp = interp1d(self.history['z'][-1::-1],
                    self.history[field][-1::-1])
            elif field == 'curvature':
                tmp = []
                for element in ztmp:
                    
                    if element not in self.turning_points:
                        tmp.append(np.inf)
                        continue
                    
                    if (type(element)) == str and (element != 'trans'):
                        tmp.append(self.turning_points[element][-1])
                    else:
                        tmp.append(np.inf)

                output.append(tmp)
                continue
            
            tmp = []
            for i, z in enumerate(redshift):

                if z is None:
                    tmp.append(np.inf)
                    continue

                if zmin <= z <= zmax:
                    tmp.append(float(interp(z)))
                else:
                    tmp.append(np.inf)
            
            output.append(tmp)
        
        # Reshape output so it's (redshift x blobs)
        self.blobs = np.array(zip(*output))
        self.blob_names, self.blob_redshifts = self.pf['inline_analysis']
        self.ztps = ztps
        
    def electron_density(self, hist, zone='igm'):            
        """ Compute electron density given simulation history. """
                
        # First, compute electron density
        nH = self.grid.cosm.nH(hist['z'][-1])
        nHe = self.grid.cosm.nHe(hist['z'][-1])
           
        n_e = nH * hist['%s_h_2' % zone][-1]           
              
        if self.helium:
            if self.pf['approx_helium']:
                n_e += nHe * hist['%s_h_2' % zone][-1]
            else:
                n_e += nHe * hist['%s_he_2' % zone][-1]
                n_e += 2 * nHe * hist['%s_he_3' % zone][-1]
                         
        return n_e
                         
    def tau_CMB(self, hist):
        """
        Compute electron-scattering optical depth between last two steps.
        """
    
        integrand = lambda z, de: de * sigma_T * self.grid.cosm.dldz(z)
        
        # Integrate via trapezoidal rule
        dz = hist['z'][-2] - hist['z'][-1]
        
        n_H = self.grid.cosm.nH(np.array(hist['z'][-2:]))
        
        tau = 0.0
        for zone in ['igm', 'cgm']:
            de = np.array(hist['%s_e' % zone])[-3:-1] * n_H
            
            if zone == 'igm':
                de *= (1. - np.array(hist['cgm_h_2'])[-3:-1])
            else:
                de *= np.array(hist['cgm_h_2'])[-3:-1]
            
            tau += 0.5 * dz * (integrand(hist['z'][-1], de[-1]) 
                + integrand(hist['z'][-2], de[-2]))
        
        return tau
        
    def Feedback(self, z, Tigm, Jlw=0.0, mu=0.6):
        """
        Modify the minimum virial temperature of star-forming haloes based
        on IGM temperature and/or strength of LW background.

        Parameters
        ----------
        z : float
            Current redshift.

        """
        
        if sum(self.feedback) == 0:
            return
        
        if (self.feedback_last_z - z < self.pf['feedback_dz']):
            return

        M_J = self.history['M_J'][-1]
        M_F = self.history['M_F'][-1]
        
        Tmin = [pop.Tmin for pop in self.pops.pops]
        for p, pop in enumerate(self.pops.pops):
            if not self.feedback[p]:
                continue
                
            if 'jeans_mass' in self.pf['feedback_method']:
                Tmin[p] = max(pop.halos.VirialTemperature(M_J, z, mu=mu),
                    Tmin[p])
                
            if 'filtering_mass' in self.pf['feedback_method']:        
                Tmin[p] = max(pop.halos.VirialTemperature(M_F, z, mu=mu), 
                    Tmin[p])
                
            if self.pf['feedback_Tmin_of_T'] is not None:
                Tmin = max(self.pf['feedback_Tmin_of_T'](Tigm, z), 
                    Tmin[p])
                
            if 'critical_Jlw' in self.pf['feedback_method']:
                #if self.pf['feedback_Tmin_of_T'] is None:
                #    Tmin_J = 
                #else:
                
                #Tmin_J = self.pf['feedback_Tmin_of_J'](Jlw, z)
                
                flux = Jlw * 10.2 * erg_per_ev / 1e-21
                
                # Machacek et al. (2001) Equation 25
                if self.pf['feedback_analytic']:
                    Tmin_J = 1e3 * 0.36 \
                        * ((4. * np.pi * flux) \
                        * ((1. + z) / 20.)**1.5 / self.grid.cosm.Omh2)**0.22
                        
                    Tmin[p] = max(Tmin_J, Tmin[p])
                
                else:
                    
                    Mcrit = 2.5e5 * ((1. + z) / 26.)**-1.5 \
                        * (1. + 6.96 * ((4. * np.pi * flux))**0.47)
                        
                    Tcrit = pop.halos.VirialTemperature(Mcrit, z)
                    
                    Tmin[p] = max(Tcrit, Tmin[p])
                
        for p, pop in enumerate(self.pops.pops):   
            pop._set_fcoll(Tmin[p], mu)
                
        self.feedback_last_z = z
                
    def JeansMass(self, z, Tigm, mu=0.6):
        rhob = self.grid.cosm.MeanBaryonDensity(z)
        
        # Cosmological Jeans mass
        return (5. * k_B * Tigm / G / mu / m_p)**1.5 \
            * (3. / 4. / np.pi / rhob)**0.5 / g_per_msun
            
    def FilteringMass(self):
        """ Eq. 6 in Gnedin (2000). """
        
        a_all = 1. / (1. + np.array(self.history['z']))
        a_now = a_all[-1]
        M_J = np.array(self.history['M_J'])
        integrand = M_J**(2./3.) * (1. - np.sqrt(a_all / a_now))
        
        return ((3. / a_now) * np.trapz(integrand, x=a_all))**1.5
        
    def tabulate_blobs(self, z):
        """
        Print blobs at a particular redshift (nicely).
        """
        
        print "-" * 25
        print "par        value "
        print "-" * 25
        
        for k, par in enumerate(self.blob_names):
            i = self.blob_redshifts.index(z)
            j = list(self.blob_names).index(par)
                        
            print "%-10s %-8.4g" % (par, self.blobs[i,j])
        
    def ComputeXRB(self, z, ztmp, xtmp, **kwargs):
        """
        Compute cosmic X-ray background flux.

        Parameters
        ----------
        z : float
            Current (observer) redshift
        ztmp : numpy.ndarray
            Array of redshifts (in ascending redshift).
        xtmp : numpy.ndarray
            Array of mean ionized fractions correspond to redshifts in ztmp.
        """
        
        if self.approx_all_xray:
            return None

        if not self.pf['radiative_transfer']:
            return None
            
        if z > self.pf['first_light_redshift']:
            return None
        
        switch = False
                
        # Check to make sure optical depth tables are still valid. 
        # If not, re-initialize UniformBackground instance(s) and
        # prepare to compute the IGM optical depth on-the-fly.        
        if self.pre_EoR:
            
            # Keep using optical depth table? If not, proceed to indented block
            if (xtmp[0] > self.pf['EoR_xavg']) or (z < self.cxrb_zall[0]):
                
                self.pb.finish()
                
                if rank == 0 and self.pf['verbose']:
                    if z <= self.zmin_igm:
                        print "\nEoR has begun (@ z=%.4g, x=%.4g) because we've reached the end of the optical depth table." \
                            % (z, xtmp[0])
                    else:
                        print "\nEoR has begun (@ z=%.4g, x=%.4g) by way of xavg > %.4g criterion." \
                            % (z, xtmp[0], self.pf['EoR_xavg'])
                        
                self.pre_EoR = False
                
                # Update parameter file
                Nz = int((np.log10(1.+self.cxrb_zall[self.cxrb_llo]) \
                        - np.log10(1.+self.pf['final_redshift'])) \
                        / self.pf['EoR_dlogx']) + 1
                        
                new_pars = {'initial_redshift': self.cxrb_zall[self.cxrb_llo], 
                    'redshift_bins': Nz, 'load_tau': False}
                
                # Loop over sources and re-initialize
                ct = 0
                self.rbs_old = []
                
                self.cxrb_fhi_EoR = []; self.cxrb_flo_EoR = []
                for i, rb in enumerate(self.rbs):
                    if self.cxrb_gen[i] is None:
                        self.cxrb_fhi_EoR.append(0.0)
                        self.cxrb_flo_EoR.append(0.0)
                        self.rbs_old.append(None)
                        continue
                    
                    # Store last two "pre-EoR" fluxes
                    E_pre, flux_lo = rb.igm.E.copy(), self.cxrb_flo[i].copy()
                    flux_hi = self.cxrb_fhi[i].copy()
                    zlo, zhi = self.cxrb_zlo, self.cxrb_zhi
                    
                    pop = self.pops.pops[i]
                    pop.pf.update(new_pars)
                    
                    self.rbs_old.append(copy.deepcopy(rb))
                    rb.__init__(pop=pop, use_tab=False)

                    fhi_interp = np.interp(rb.igm.E, E_pre, flux_hi)
                    flo_interp = np.interp(rb.igm.E, E_pre, flux_lo)

                    self.cxrb_lhi = rb.igm.L - 1
                    self.cxrb_zhi = rb.igm.z[-1]

                    self.cxrb_llo = rb.igm.L - 2
                    self.cxrb_zlo = rb.igm.z[-2]

                    self.cxrb_fhi_EoR.append(fhi_interp)

                    # Interpolate to current redshift
                    tmp = np.zeros_like(rb.igm.E)
                    for j, nrg in enumerate(rb.igm.E):
                        tmp[j] = np.interp(self.cxrb_zlo, [zlo, zhi], 
                            [flo_interp[j], fhi_interp[j]])

                    self.cxrb_flo_EoR.append(tmp)

                    # Optical depth "on-the-fly"
                    if ct == 0:
                        self.tau_otf_all = np.zeros([rb.igm.L, rb.igm.N])
                        self.tau_otf = np.zeros(rb.igm.N)
                    
                    self.cxrb_gen[i] = rb.XrayFluxGenerator(self.tau_otf,
                        flux0=fhi_interp)

                    # Poke generator once 
                    # (the first redshift is guaranteed to overlap with p-EoR)
                    self.cxrb_gen[i].next()
                    
                    ct += 1
                                        
                    # Poke generator again if redshift resolution very good?
                    
                del self.cxrb_flo, self.cxrb_fhi
                self.cxrb_fhi = self.cxrb_fhi_EoR
                self.cxrb_flo = self.cxrb_flo_EoR

                self._init_XRB(pre_EoR=False, **kwargs)
                
                if self.pf['progress_bar']:
                    print ""

                self.pb = ProgressBar(self.tf, '21-cm (EoR)',
                    use=self.pf['progress_bar'])
                self.pb.start()

        # Loop over UniformBackground instances, sum fluxes
        ct = 0
        new_fluxes = []
        for i, rb in enumerate(self.rbs):
            if self.cxrb_gen[i] is None:
                new_fluxes.append(0.0)
                continue

            # If we don't have fluxes for this redshift yet, poke the
            # generator to get the next set
            if z < self.cxrb_zlo:
                switch = True
                
                # If in EoR, update optical depth reference
                if not self.pre_EoR and ct == 0:

                    # Optical depth between zlo and zlo-dz
                    this_tau = self.ComputeTauOTF(rb)
                    self.tau_otf_all[self.cxrb_llo] = this_tau.copy()
                    self.tau_otf[:] = this_tau
                    
                    if np.any(np.isinf(this_tau)):
                        raise ValueError('infinite optical depth')

                # Now, compute flux
                new_fluxes.append(self.cxrb_gen[i].next())
                
            ct += 1

        # If we just retrieved new fluxes, update attributes accordingly
        if switch:
            znow = self.cxrb_zall[self.cxrb_llo]
            znext = self.cxrb_zall[self.cxrb_llo - 1]
            
            self.cxrb_lhi -= 1
            self.cxrb_llo -= 1

            self.cxrb_zhi = self.cxrb_zlo
            self.cxrb_zlo = znext
            
            self.cxrb_fhi = copy.deepcopy(self.cxrb_flo)
            self.cxrb_flo = copy.deepcopy(new_fluxes)
            
            # Heat and ionization
            self.cxrb_hhi = self.cxrb_hlo
            self.cxrb_G1hi = self.cxrb_G1lo
            self.cxrb_G2hi = self.cxrb_G2lo
                        
            self.cxrb_hlo = np.zeros([self.Nrbs, self.grid.N_absorbers])
            self.cxrb_G1lo = np.zeros([self.Nrbs, self.grid.N_absorbers])
            self.cxrb_G2lo = np.zeros([self.Nrbs, self.grid.N_absorbers])
            
            for i, rb in enumerate(self.rbs):
                
                for j, absorber in enumerate(self.grid.absorbers):
                    
                    if j > 0 and self.pf['approx_helium']:
                        continue
                                        
                    if rb.pop.pf['approx_xray']:
                        
                        heat_rb = rb.igm.HeatingRate(self.cxrb_zlo, 
                            species=j, return_rc=True, **kwargs)
                        G1_rb = rb.igm.IonizationRateIGM(self.cxrb_zlo, 
                            species=j, return_rc=True, **kwargs)
                        
                        self.cxrb_hlo[i,j] = heat_rb
                        self.cxrb_G1lo[i,j] = G1_rb
                        
                        self.xray_flux[i].append(0.0)
                        self.xray_heat[i].append(self.cxrb_hlo[i])
                        
                        if self.pf['secondary_ionization'] == 0:
                            self.cxrb_G2lo[i,j] = np.zeros(self.grid.N_absorbers)
                            continue
                        
                        G2_rb = rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo, 
                            species=j, return_rc=True, **kwargs)

                        self.cxrb_G2lo[i,:] = G2_rb

                    self.xray_flux[i].append(self.cxrb_flo[i])
                    
                    self.cxrb_hlo[i,j] = rb.igm.HeatingRate(self.cxrb_zlo, 
                        species=j, xray_flux=self.cxrb_flo[i], return_rc=True, 
                        **kwargs)
                    
                    self.xray_heat[i].append(self.cxrb_hlo[i])    
                    
                    self.cxrb_G1lo[i,j] = rb.igm.IonizationRateIGM(self.cxrb_zlo, 
                        species=j, xray_flux=self.cxrb_flo[i], return_rc=True,
                        **kwargs)
                    
                    if self.pf['secondary_ionization'] > 0:
                        self.cxrb_G2lo[i,j] = rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo, 
                            species=j, xray_flux=self.cxrb_flo[i], 
                            return_rc=True, **kwargs)

        return None    
              
    def ComputeTauOTF(self, rb):
        """
        Compute IGM optical depth on-the-fly (OTF).
        
        Must extrapolate to determine IGM ionization state at next z.
        """

        if self.cxrb_llo <= 1:
            znow = rb.igm.z[1]
            znext = rb.igm.z[0]
        else:
            znow = rb.igm.z[self.cxrb_llo]
            znext = rb.igm.z[self.cxrb_llo-1]

        # Redshift in order of increasing redshift
        zz = np.array(self.history['z'][-1:-4:-1])
        
        # Compute mean hydrogen ionized fraction
        xHI_igm = np.array(self.history['igm_h_1'])[-1:-4:-1]
        xHI_cgm = np.array(self.history['cgm_h_1'])[-1:-4:-1]
        
        xHI_avg = xHI_cgm * xHI_igm
                
        xx = [xHI_avg]
        nf = [rb.igm.cosm.nH]
        
        if self.helium:
            xHeI_igm = np.array(self.history['igm_he_1'][-1:-4:-1])
            xHeI_cgm = np.array(self.history['cgm_he_1'][-1:-4:-1])
            xHeI_avg = xHeI_cgm * xHeI_igm
                                                                 
            xHeII_igm = np.array(self.history['igm_he_2'][-1:-4:-1])
            xHeII_cgm = np.array(self.history['cgm_he_2'][-1:-4:-1])
            xHeII_avg = xHeII_cgm * xHeII_igm

            xx.extend([xHeI_avg, xHeII_avg])
            nf.extend([rb.igm.cosm.nHe]*2)
        elif self.pf['approx_helium']:
            xx.extend([xHI_avg, np.zeros_like(xHI_avg)])
            nf.extend([rb.igm.cosm.nHe]*2)

        tau = np.zeros_like(rb.igm.E)
        for k in range(3):  # absorbers

            if k > 0 and (not self.helium) and (not self.pf['approx_helium']):
                continue
                
            if self.pf['approx_helium'] and k == 2:
                continue

            # Interpolate to get current neutral fraction
            xnow = np.interp(znow, zz, xx[k])
            
            # Extrapolate to find neutral fractions at these two redshifts
            m = (xx[k][1] - xx[k][0]) / (zz[1] - zz[0])
            xnext = m * (znext - zz[0]) + xx[k][0]
            
            # Get cross-sections
            snext = rb.igm.sigma_E[k]
            snow = np.roll(rb.igm.sigma_E[k], -1)
            
            # Neutral densities
            nnow = nf[k](znow) * xnow
            nnext = nf[k](znext) * xnext
                                    
            tau += 0.5 * (znow - znext) \
                * (nnext * rb.igm.cosm.dldz(znext) * snext \
                +  nnow  * rb.igm.cosm.dldz(znow)  * snow)

        tau[-1] = 0.0

        return tau
                      
    def save(self, prefix, suffix='pkl', clobber=False):
        self.write.save(prefix, suffix, clobber)
        
    def _check_for_conflicts(self):
        if not self.pf['radiative_transfer']:
            return
            
        if self.pf['approx_lya'] == 0 and np.all(self.pf['spectrum_Emin'] > 13.6):
            raise ValueError('Must supply Lyman series spectrum!')
        
        if self.pf['approx_xray'] == 0 and self.pf['load_tau'] == 0 \
            and self.pf['tau_table'] is None:
            raise ValueError('Supply tau_table or set load_tau=True when approx_xray=False')

