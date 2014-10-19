"""

OptimizeSpectrum.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Mon Feb 25 09:00:27 2013

Description: 

"""

import copy
import numpy as np
from ..static import Grid
from ..sources import RadiationSource
from ..physics.Constants import erg_per_ev 
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

try:
    import ndmin
except ImportError:
    pass
    
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
    
class SpectrumOptimization:
    def __init__(self, logN=None, Z=[1], nfreq=1, rs=None, fn=None, 
        thinlimit=False, isothermal=False, secondary_ionization=0, mcmc=False,
        loglikelihood=None):
        self.logN = logN
        self.Z = Z
        self.nfreq = nfreq
        self.rs = rs
        self.fn = fn
        self.thinlimit = thinlimit
        self.isothermal = isothermal
        self.secondary_ionization = secondary_ionization
        self.mcmc = mcmc
        
        # Use Grid class to carry info about absorbing species
        self.grid = Grid()
        self.grid.set_physics(isothermal=self.isothermal, 
            secondary_ionization=self.secondary_ionization)
        self.grid.set_chemistry(Z=self.Z)
        
        # Initialize radiation source
        if self.rs is None:
            self.rs = RadiationSource(self.grid, 
                self.logN, **{'spectrum_file': self.fn})
        elif type(self.rs) is dict:
            self.rs = RadiationSource(self.grid, 
                self.logN, **self.rs)        
        
        # What integrals are we comparing to?
        self.integrals = copy.deepcopy(self.rs.tab.IntegralList)
        self.integrals.pop(self.integrals.index('Tau'))
        
        # Figure out size of optical depth array [np.prod(logN.shape) x nfreq]
        self.tau_dims = list(self.rs.tab.dimsN.copy())
        self.tau_dims.append(len(self.Z))
        self.tau_dims.append(self.nfreq)
        
        self.tab_dims = list(self.rs.tab.dimsN.copy())
        #self.tab_dims.insert(0, len(self.Z))
        #self.tab_dims.insert(0, len(self.integrals))
    
    def run(self, steps, limits=None, step=None, burn=None, guess=None, err=0.1):
        self.__call__(steps, guess=guess, limits=limits, step=step, 
            burn=burn, err=err)
    
    def __call__(self, steps, guess=None, limits=None, step=None, burn=None, 
        err=0.1, gamma=0.99, afreq=50):
        """
        Construct optimal discrete spectrum for a given radiation source.
        
        Need: Column density range for all species.
            logN = list of arrays, each element should be logN for corresponding
                entry in Z.
        """
        
        if rank == 0:
            print 'Finding optimal %i-bin discrete SED...' % self.nfreq

        # Initialize sampler - generous control parameters
        if limits is None:
            limits = [(13.6, 1e2)] * self.nfreq
            limits.extend([(0.0, 1.0)] * self.nfreq)
        if step is None:    
            step = [(lim[1] - lim[0]) for lim in limits]
        if guess is None:
            guess = []
            for i in xrange(self.nfreq * 2):
                guess.append(np.random.rand() * \
                    (limits[i][1] - limits[i][0]) + limits[i][0])
            
        if self.mcmc:
            self.sampler = ndmin.MarkovChain(lambda p: self.cost(p, err), 
                cov=np.diag(step), limits=limits)
            self.sampler.burn_in(burn, guess=guess)
            
            if steps > 0:
                self.sampler.run(steps, guess=self.sampler.xarr_ML, 
                    eigval=self.sampler.eigenvals, eigvec=self.sampler.eigenvecs)    
        else:    
            self.sampler = ndmin.Annealer(self.cost, limits=limits, 
                step=step, afreq=afreq, gamma=gamma)
            
            self.sampler.run(steps)
         
        if rank == 0:
            print 'Optimization complete.'    
        
    def cost(self, pars, err=0.1):
        E, LE = ndmin.util.halve_list(pars)
        #bfx = sigma_E(E)        
                
        # Compute optical depth for all combinations of column densities
        if self.thinlimit:
            tau = np.zeros(self.tau_dims)
        else:
            tau = self.tau(E)
                        
        # Compute discrete versions of phi & psi
        # NOTE: if ionization thresholds all below smallest emission energy,
        # integrals for all species are identical.
        discrete_tables = self.discrete_tabs(E, LE, tau)
                
        # Keep only one element if we're doing optically thin optimization
        if self.thinlimit:
            mask = 0
        else:
            mask = Ellipsis
            
        # Compute cost
        cost = 0.0
        for i, integral in enumerate(self.integrals):
            for j, absorber in enumerate(self.grid.absorbers):
                name = self.tab_name(integral, absorber)

                cont = self.rs.tabs[name]
                disc = discrete_tables[name]

                if self.mcmc: # really lnL in this case
                                                            
                    #if integral == 'Phi':
                    #    var = (10**self.logN[j])**2 * \
                    #        np.sum(LE**2 * np.exp(-2. * tau[...,j,:]) \
                    #        * (0.01 * bfx)**2 / E**2, axis = -1) / erg_per_ev**2
                    #else:
                    #    var = (10**self.logN[j])**2 * \
                    #        np.sum(LE**2 * np.exp(-2. * tau[...,j,:]) \
                    #        * (0.01 * bfx)**2, axis = -1)

                    cost -= np.sum((disc - cont)[mask]**2 / err**2)
                else:
                    cost += np.max(np.abs(disc - cont)[mask])
                    
                    if not self.thinlimit:
                        cost += np.mean(np.abs(disc - cont)[mask])
        
        return cost
    
    def tab_name(self, integral, absorber):
        return 'log%s_%s' % (integral, absorber)

    def discrete_tabs(self, E, LE, tau=None):
        """
        Compute values of integral quantities assuming discrete emission.
        """
        
        if tau is None:
            tau = self.tau(E)
        
        discrete_tables = {}
        for i, integral in enumerate(self.integrals):
            for j, absorber in enumerate(self.grid.absorbers):
                Eth = self.grid.ioniz_thresholds[absorber]
                
                tmp = np.zeros(self.tab_dims)
                if integral == 'Phi':
                    tmp = self.DiscretePhi(E, LE, tau[:,j,:], Eth)
                if integral == 'Psi':
                    tmp = self.DiscretePsi(E, LE, tau[:,j,:], Eth)
                        
                discrete_tables[self.tab_name(integral, absorber)] = tmp.copy()        
                            
        return discrete_tables   
        
    def DiscretePhi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E) / E
        to_sum[E < Eth] = 0.0
        summed = np.sum(to_sum, axis = -1)
        return np.log10(summed / erg_per_ev)

    def DiscretePsi(self, E, LE, tau_E, Eth):
        to_sum = LE * np.exp(-tau_E)
        to_sum[E < Eth] = 0.0
        summed = np.sum(to_sum, axis = -1)
        return np.log10(summed)
    
    def tau(self, E):
        """
        Compute total optical depth (over all species) as a function of 
        discrete emission energy E (eV).
        """
        
        tau_E = np.zeros(self.tau_dims)
        
        for i in xrange(self.rs.tab.elements_per_table):
            for j, absorber in enumerate(self.grid.absorbers):
                
                loc = list(self.rs.tab.indices_N[i])
                loc.append(j)
                                    
                tau_E[tuple(loc)] = \
                    np.array(self.rs.tab.PartialOpticalDepth(E, 
                    self.rs.tab.Nall[i], absorber))
                    
        return tau_E
        
    def dump(self, fn):
        """
        Write optimization result to HDF5 file.
        """    
        
        if rank > 0:
            return
            
        f = h5py.File(fn, 'w')
        f.create_dataset('chain', data=self.sampler.chain)
        
        if self.mcmc:
            f.create_dataset('post', data=self.sampler.post)
        else:
            f.create_dataset('cost', data=self.sampler.cost)
        
        print 'Wrote chain to %s.' % fn    
        f.close()
        
            
                