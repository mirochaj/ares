"""

OpticalDepth.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Feb 21 11:26:50 MST 2015

Description: 

"""

import numpy as np
from ..util import ProgressBar, ParameterFile

class OpticalDepth:
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)
        
    def ClumpyOpticalDepth(self):    
        pass
    
    def OpticalDepth(self):
        return self.DiffuseOpticalDepth()    
        
    def DiffuseOpticalDepth(self, z1, z2, E, **kwargs):
        """
        Compute the optical depth between two redshifts.
    
        If no keyword arguments are supplied, assumes the IGM is neutral.
    
        Parameters
        ----------
        z1 : float
            observer redshift
        z2 : float
            emission redshift
        E : float
            observed photon energy (eV)  
    
        Notes
        -----
        If keyword argument 'xavg' is supplied, it must be a function of 
        redshift.
    
        Returns
        -------
        Optical depth between z1 and z2 at observed energy E.
    
        """
    
        kw = self._fix_kwargs(functionify=True, **kwargs)
    
        # Compute normalization factor to help numerical integrator
        norm = self.cosm.hubble_0 / c / self.sigma0
    
        # Temporary function to compute emission energy of observed photon
        Erest = lambda z: self.RestFrameEnergy(z1, E, z)
    
        # Always have hydrogen
        sHI = lambda z: self.sigma(Erest(z), species=0)
    
        # Figure out number densities and cross sections of everything
        if self.approx_He:
            nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
            nHeI = lambda z: nHI(z) * self.cosm.y
            sHeI = lambda z: self.sigma(Erest(z), species=1)
            nHeII = lambda z: 0.0
            sHeII = lambda z: 0.0
        elif self.self_consistent_He:
            if type(kw['xavg']) is not list:
                raise TypeError('hey! fix me')
    
            nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
            nHeI = lambda z: self.cosm.nHe(z) \
                * (1. - kw['xavg'](z) - kw['xavg'](z))
            sHeI = lambda z: self.sigma(Erest(z), species=1)
            nHeII = lambda z: self.cosm.nHe(z) * kw['xavg'](z)
            sHeII = lambda z: self.sigma(Erest(z), species=2)
        else:
            nHI = lambda z: self.cosm.nH(z) * (1. - kw['xavg'](z))
            nHeI = sHeI = nHeII = sHeII = lambda z: 0.0
    
        tau_integrand = lambda z: norm * self.cosm.dldz(z) \
            * (nHI(z) * sHI(z) + nHeI(z) * sHeI(z) + nHeII(z) * sHeII(z))
    
        # Integrate using adaptive Gaussian quadrature
        tau = quad(tau_integrand, z1, z2, epsrel=self.rtol, 
            epsabs=self.atol, limit=self.divmax)[0] / norm
    
        return tau
    
    def TabulateOpticalDepth(self, xavg=lambda z: 0.0):
        """
        Compute optical depth as a function of (redshift, photon energy).
    
        Parameters
        ----------
        xavg : function
            Mean ionized fraction as a function of redshift.
    
        Notes
        -----
        Assumes logarithmic grid in variable x = 1 + z. Corresponding 
        grid in photon energy determined in _init_xrb.    
    
        Returns
        -------
        Optical depth table.
    
        """
    
        # Create array for each processor
        tau_proc = np.zeros([self.L, self.N])
    
        pb = ProgressBar(self.L * self.N, 'tau')
        pb.start()     
    
        # Loop over redshift, photon energy
        for l in range(self.L):
    
            for n in range(self.N):
                m = l * self.N + n + 1
    
                if m % size != rank:
                    continue
    
                # Compute optical depth
                if l == (self.L - 1):
                    tau_proc[l,n] = 0.0
                else:
                    tau_proc[l,n] = self.OpticalDepth(self.z[l], 
                        self.z[l+1], self.E[n], xavg=xavg)
    
                pb.update(m)
    
        pb.finish()
    
        # Communicate results
        if size > 1:
            tau = np.zeros_like(tau_proc)       
            nothing = MPI.COMM_WORLD.Allreduce(tau_proc, tau)            
        else:
            tau = tau_proc
    
        return tau
    