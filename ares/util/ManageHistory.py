"""

WriteData.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Aug 26 10:36:50 MDT 2014

Description: 

"""

import pickle, os
import numpy as np
from scipy.interpolate import interp1d

try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1

zones = ['igm', 'cgm']

# Each zone knows about this stuff - they are just numbers
states = ['h_1', 'h_2', 'he_1', 'he_2', 'he_3', 'e', 'Tk']
derived = ['xavg', 'Ja', 'tau_e', 'dTb', 'Ts']

# Each zone knows about this stuff too - but the dimensions are different
rates = ['Gamma', 'gamma', 'heat']

class WriteData():
    def __init__(self, sim):
        self.sim = sim

    def _initialize_history(self):
        """
        Store initial conditions in 'history' dictionary which will store
        entire, well, history.
        """

        history = {}

        n_H = self.sim.grid.cosm.nH(self.sim.pf['initial_redshift'])
        history['z'] = [self.sim.pf['initial_redshift']]
    
        grids = [self.sim.grid_igm, self.sim.grid_cgm]
        for i, zone in enumerate(zones):
    
            grid = grids[i]
    
            for j, state in enumerate(states):
                
                if (not self.sim.helium) and state in ['he_1', 'he_2', 'he_3']:
                    continue
                
                history['%s_%s' % (zone, state)] = [grid.data[state][0]]

            history['%s_gamma' % zone] = [np.zeros([self.sim.grid.N_absorbers]*2)]
            history['%s_Gamma' % zone] = [np.zeros(self.sim.grid.N_absorbers)]
            history['%s_heat' % zone] = [np.zeros(self.sim.grid.N_absorbers)]
    
        # Keep track of mean hydrogen ionization fractions
        history['xavg'] = [self.sim.grid_cgm.data['h_2'][0] \
                + (1. - self.sim.grid_cgm.data['h_2'][0]) \
                * self.sim.grid_igm.data['h_2'][0]]
    
        history['Ja'] = [0.0]
    
        history['tau_e'] = [0.0]
    
        if self.sim.feedback_ON:
            history['M_J'] = [self.sim.JeansMass(self.sim.pf['initial_redshift'], self.sim.grid_igm.data['Tk'][0])]
            history['M_F'] = [history['M_J'][0]]
    
        history['Ts'] = [self.sim.grid.hydr.SpinTemperature(history['z'][0], 
            history['igm_Tk'][0], history['Ja'][0], 
            history['igm_h_2'][0], history['igm_e'][0] * n_H)]
    
        history['dTb'] = \
            [self.sim.grid.hydr.DifferentialBrightnessTemperature(history['z'][0],
                history['xavg'][0], history['Ts'][0])]
    
        # Insert initial conditions data into history                
        if hasattr(self.sim, 'inits'):
    
            i = np.argmin(np.abs(self.sim.inits['z'] - self.sim.pf['initial_redshift']))
            if self.sim.inits['z'][i] <= self.sim.pf['initial_redshift']:
                i += 1
    
            for j, red in enumerate(self.sim.inits['z']):    
    
                if j < i:
                    continue
    
                for key in history:
    
                    if key == 'igm_h_2':  
                        xe = self.sim.inits['xe'][j]
                        
                        if 2 not in self.sim.grid.Z:
                            xe = min(xe, 1.0)
                        
                        xi = xe / (1. + self.sim.grid.cosm.y)
                        history[key].insert(0, xi)
                    elif key == 'igm_h_1':
                        xe = self.sim.inits['xe'][j]
                        
                        if 2 not in self.sim.grid.Z:
                            xe = min(xe, 1.0)
                            
                        xi = xe / (1. + self.sim.grid.cosm.y)
                        history[key].insert(0, 1. - xi)
                    elif key == 'Tk':
                        history[key].insert(0, self.sim.inits[key][j])
                    elif key == 'z':
                        history[key].insert(0, self.sim.inits[key][j]) 
    
                    elif key == 'Ts':   
                        xe = self.sim.inits['xe'][j]
                        
                        if 2 not in self.sim.grid.Z:
                            xe = min(xe, 1.0)
                        
                        xi = xe / (1. + self.sim.grid.cosm.y)
                        
                        val = self.sim.grid.hydr.SpinTemperature(red, 
                            self.sim.inits['Tk'][j], 0.0, 
                            xi, self.sim.inits['xe'][j] * self.sim.grid.cosm.nH(red))
                        history['Ts'].insert(0, val)
                    elif key == 'dTb':    
                        xe = self.sim.inits['xe'][j]
                        
                        if 2 not in self.sim.grid.Z:
                            xe = min(xe, 1.0)
                        
                        xi = xe / (1. + self.sim.grid.cosm.y)
                        Ts = self.sim.grid.hydr.SpinTemperature(red, 
                            self.sim.inits['Tk'][j], 0.0, 
                            xi, self.sim.inits['xe'][j] * self.sim.grid.cosm.nH(red))
                        val = self.sim.grid.hydr.DifferentialBrightnessTemperature(red,
                            xi, Ts)    
                        history['dTb'].insert(0, val)
    
                    elif key in ['igm_gamma', 'cgm_gamma']:
                        history[key].insert(0, np.zeros([self.sim.grid.N_absorbers]*2))
                    elif key in ['igm_Gamma', 'cgm_Gamma', 'cgm_heat', 'igm_heat']:
                        history[key].insert(0, np.zeros([self.sim.grid.N_absorbers]))
                    else:
                        history[key].insert(0, 0.0)
    
        return history
    
    def _update_history(self, z, zpre, data_igm, data_cgm):
        """ Save stuff to history dictionary """
        
        # Make data values floats rather than single-element arrays
        data_igm_fl = self._strip(data_igm)
        data_cgm_fl = self._strip(data_cgm)
        
        # Condense data from each grid to single dictionary
        to_keep = {'z': z}
        
        to_keep['xavg'] = data_cgm_fl['h_2'] \
            + (1. - data_cgm_fl['h_2']) * data_igm_fl['h_2']
        
        ##    
        # Save data! Could make a separate routine for this.
        ##
        grids = [data_igm_fl, data_cgm_fl]
        rts = [self.sim.rt_igm, self.sim.rt_cgm]
        for j, zone in enumerate(zones):
            
            rt = rts[j]
            grid = grids[j]
            
            # Save ion fractions and electron density
            for key in states:
                if (not self.sim.helium) and key in ['he_1', 'he_2', 'he_3']:
                    continue

                to_keep.update({'%s_%s' % (zone, key): grid[key]})

            # REMEMBER: these are rate coefficients, not actual rates
            if self.sim.pf['radiative_transfer'] and (zpre <= self.sim.zfl):
                to_keep['%s_heat' % zone] = rt.kwargs['Heat'].squeeze()
                to_keep['%s_Gamma' % zone] = rt.kwargs['Gamma'].squeeze()
                to_keep['%s_gamma' % zone] = rt.kwargs['gamma'].squeeze()
            else:
                to_keep['%s_heat' % zone] = np.zeros(self.sim.grid.N_absorbers)
                to_keep['%s_Gamma' % zone] = np.zeros(self.sim.grid.N_absorbers)
                to_keep['%s_gamma' % zone] = np.zeros(self.sim.grid.N_absorbers)

        # Store to_keep items in history
        for key in to_keep:
            self.sim.history[key].append(to_keep[key])

        ## 
        # DERIVED QUANTITIES
        ##

        # Compute Lyman-alpha background for WF coupling
        if self.sim.pf['radiative_transfer'] and z < self.sim.zfl:
            Ja = np.sum([rb.LymanAlphaFlux(z) for rb in self.sim.rbs])
        else:
            Ja = 0.0

        # z, Tk, Ja, nH, ne
        Ts = self.sim.grid.hydr.SpinTemperature(z, to_keep['igm_Tk'], Ja,
            to_keep['igm_h_2'], to_keep['igm_e'] * self.sim.grid_igm.cosm.nH(z))
        dTb = self.sim.grid.hydr.DifferentialBrightnessTemperature(z,
            to_keep['xavg'], Ts)

        # CMB optical depth
        if self.sim.step > 1:
            tau_e = self.sim.history['tau_e'][-1] + self.sim.tau_CMB(self.sim.history)
        else:
            tau_e = 0.0

        # Store derived fields
        self.sim.history['Ja'].append(Ja)
        self.sim.history['Ts'].append(Ts)
        self.sim.history['dTb'].append(dTb)

        self.sim.history['tau_e'].append(tau_e)

        if self.sim.feedback_ON:
            self.sim.history['M_J'].append(self.JeansMass(z, data_igm_fl['igm_Tk']))
            self.sim.history['M_F'].append(self.FilteringMass())

    def _strip(self, data):
        """ Convert single element arrays to floats in all entries. """
    
        new = {}
        for key in data:
            new[key] = data[key][0]
    
        return new
    
    def save(self, prefix, suffix='pkl', clobber=False):
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
            Suffix of save filename. Can be hdf5 (or h5), pkl, or npz. 
            Anything else will be assumed to be ASCII format (e.g., .txt).
        clobber : bool
            Overwrite pre-existing files of same name?
    
        """
        
        fn = '%s.history.%s' % (prefix, suffix)
    
        if os.path.exists(fn):
            if clobber:
                os.remove(fn)
            else: 
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    
        if suffix == 'pkl':            
            f = open(fn, 'wb')
            pickle.dump(self.sim.history, f)
            f.close()
            
        elif suffix in ['hdf5', 'h5']:
            f = h5py.File(fn, 'w')
            for key in self.sim.history:
                print key
                f.create_dataset(key, data=np.array(self.sim.history[key]))
            f.close()
    
        elif suffix == 'npz':
            f = open(fn, 'w')
            np.savez(f, **self.sim.history)
            f.close()
    
        # ASCII format
        else:            
            f = open(fn, 'w')
            print >> f, "#",
        
            for key in self.sim.history:
                print >> f, '%-18s' % key,
        
            print >> f, '\n'
        
            # Now, the data
            for i in range(len(self.sim.history[key])):
                s = ''
        
                for key in self.sim.history:
                    s += '%-20.8e' % self.sim.history[key][i]
        
                if not s.strip():
                    continue
        
                print >> f, s
        
            f.close()

        if os.path.exists('%s.pars.pkl' % prefix):
            if clobber:
                os.remove('%s.pars.pkl' % prefix)
            else: 
                raise IOError('%s exists! Set clobber=True to overwrite.' % fn)
    
        # Save parameter file
        f = open('%s.parameters.pkl' % prefix, 'wb')
        pickle.dump(self.sim.pf, f)
        f.close()
        
        if rank == 0:
            print 'Wrote %s and %s.parameters.pkl' % (fn, prefix)     

    def get_blobs(self):
        """
        By default, this will compute values of all elements in history dict
        at turning points.
        
        Assumes simulation at least ran to completion.
        """
        
        if not self.sim.pf['track_extrema']:
            return None

        results = {feature:{} for feature in list('BCD')}
        
        for i, item in enumerate(self.sim.history):
            
            if item[0:3] in zones:
                if item[4:] in rates:
                    blob = self._get_Nd_blob(item)
                    if blob is not None:
                        for key in blob:
                            results[key].update(blob[key].copy())
                        continue
                
            if item[4:] == 'gamma':
                continue    
                
            interp = interp1d(self.sim.history['z'][-1::-1],
                self.sim.history[item][-1::-1])
                
            for j, feature in enumerate(list('BCD')):
            
                try:
                    results[feature][item] = \
                        float(interp(self.sim.turning_points[feature][0]))
                except KeyError:
                    results[feature][item] = -77777
                except IndexError:
                    results[feature][item] = -88888
                except AttributeError:
                    results[feature][item] = -99999
                    
        return results
        
    def _get_Nd_blob(self, item):
        """
        Some things in sim.history will be arrays! Need to handle that differently.
        """

        shape = []
        
        if self.sim.helium:
            shape.append(3)
            
        if self.sim.pops.N > 1:
            shape.append(self.sim.pops.N)
        
        if not shape:
            return None
            
        if item[4:] == 'gamma':
            print 'Havent worried about gamma yet.'
            return None    

        results = {feature:{} for feature in list('BCD')}
        for j, feature in enumerate(list('BCD')):
            
            result_arr = np.zeros(shape) 
            
            if self.sim.helium and (self.sim.pops.N > 1):
                for k in range(3):
                    for l in range(self.sim.pops.N):
                        interp = interp1d(self.sim.history['z'][-1::-1],
                            self.sim.history[item][-1::-1, k, l])
                            
                        result_arr[k,l] = \
                            float(interp(self.sim.turning_points[feature][0]))
                            
            elif self.sim.helium:
                for k in range(3):
                    interp = interp1d(self.sim.history['z'][-1::-1],
                        self.sim.history[item][-1::-1, k])
                    result_arr[k] = \
                        float(interp(self.sim.turning_points[feature][0]))
                        
            elif self.sim.pops.N > 1:
                for l in range(self.sim.pops.N):
                    interp = interp1d(self.sim.history['z'][-1::-1],
                        self.sim.history[item][-1::-1, l])
                    
                    result_arr[l] = \
                        float(interp(self.sim.turning_points[feature][0]))
                        
            try:
                results[feature][item] = result_arr.copy()
            except KeyError:
                results[feature][item] = -77777
            except IndexError:
                results[feature][item] = -88888
            except AttributeError:
                results[feature][item] = -99999
                        
        return results                
                        
    def arrayify_blobs(self, blobs):
        """
        For 
        """
        pass

    def get_population_data(self):
        """
        Utility for grabbing SFRD, for instance.
        """
        pass
        
        
        