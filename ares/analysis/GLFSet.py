"""

GLFSet.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct 24 10:42:45 PDT 2015

Description: 

"""

import re
import numpy as np
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..populations.Galaxy import param_redshift
from ..util.SetDefaultParameterValues import SetAllDefaults

class GLFSet(ModelSet):
    """
    Basically a ModelSet instance with routines specific to the high-z
    galaxy luminosity function.
    """
    
    def get_data(self, z):
        i = self.data['z'].index(z)
        
        return self.data['x'][i], self.data['y'][i], self.data['err'][i]
    
    def ReconstructedLF(self, z, ax=None, fig=1, N=100, 
        resample=True, **kwargs):
        """
        Plot constraints on the luminosity function at given z.
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
        
        # Find relevant elements in chain
        samples = {}
        for i, key in enumerate(self.parameters):
            if not re.search('_lf_', key):
                continue
                
            prefix, redshift = param_redshift(key)
            if z != redshift:
                continue
                
            samples[prefix[0:prefix.find('{')]] = self.chain[:,i]
        
        i = self.data['z'].index(z)
        
        if resample:
            mi, ma = self.data['x'][i][0], self.data['x'][i][-1]
            M = np.linspace(mi-2, ma+2, 200)
        else:
            M = np.array(self.data['x'][i])
        
        pst = 10**samples['pop_lf_pstar']
        Mst = samples['pop_lf_Mstar']            
        a = samples['pop_lf_alpha']
        
        for i in range(N):
            phi = 0.4 * np.log(10.) * pst[i] \
                * (10**(0.4 * (Mst[i] - M)))**(1. + a[i]) \
                * np.exp(-10**(0.4 * (Mst[i] - M)))
            
            ax.semilogy(M, phi, **kwargs)
        
        pl.draw()
        return ax
            
    def PlotData(self, z, ax=None, fig=1, **kwargs):
        M, phi, err = self.get_data(z)
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        ax.errorbar(M, phi, yerr=err, fmt='none', **kwargs)
        ax.set_xlabel(r'$M$')
        ax.set_ylabel(r'$\phi \ (\mathrm{cMpc}^{-3} \ \mathrm{mag}^{-1})$')
        ax.set_yscale('log')
        pl.draw()
        
        return ax
            
    def SlopeEvolution(self):
        pass
        
    def MstarEvolution(self):
        pass
    
    def LstarEvolution(self):
        pass
        
        
        