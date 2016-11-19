"""

GLFSet.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Oct 24 10:42:45 PDT 2015

Description: 

"""

import re
import ares
import numpy as np
from ..util import read_lit
import matplotlib.pyplot as pl
from .ModelSet import ModelSet
from ..phenom.DustCorrection import DustCorrection
from ..util.SetDefaultParameterValues import SetAllDefaults

ln10 = np.log(10.)

phi_of_M = lambda M, pstar, Mstar, alpha: 0.4 * ln10 * pstar \
    * (10**(0.4 * (Mstar - M)*(1. + alpha))) \
    * np.exp(-10**(0.4 * (Mstar - M)))

class ModelSetLF(ModelSet):
    """
    Basically a ModelSet instance with routines specific to the high-z
    galaxy luminosity function.
    """
    
    @property
    def dc(self):
        if not hasattr(self, '_dc'):
            self._dc = DustCorrection(**self.base_kwargs)
        return self._dc
    
    def get_data(self, z):
        i = self.data['z'].index(z)
        
        return self.data['x'][i], self.data['y'][i], self.data['err'][i]
    
    def SFE(self, z, ax=None, fig=1, name='fstar', shade_by_like=False, 
        like=0.685, scatter_kwargs={}, take_log=False, un_log=False,
        multiplier=1, skip=0, stop=None, **kwargs):
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        if shade_by_like:
            q1 = 0.5 * 100 * (1. - like)    
            q2 = 100 * like + q1    
            
        info = self.blob_info(name)
        ivars = self.blob_ivars[info[0]]
        
        # We assume that ivars are [redshift, magnitude]
        M = ivars[1]
        
        loc = np.argmax(self.logL[skip:stop])
        
        sfe = []
        for i, mass in enumerate(M):
            data = self.ExtractData(name, ivar=[z, mass],
                take_log=take_log, un_log=un_log, multiplier=multiplier)
        
            if not shade_by_like:
                sfe.append(data[name][skip:stop][loc])
            else:
                lo, hi = np.percentile(data[name][skip:stop].compressed(), 
                    (q1, q2))
                sfe.append((lo, hi))    
        
        if shade_by_like:
            sfe = np.array(sfe).T
        
            if take_log:
                sfe = 10**sfe
            else:
                zeros = np.argwhere(sfe == 0)
                for element in zeros:
                    sfe[element[0],element[1]] = 1e-15
        
            ax.fill_between(M, sfe[0], sfe[1], **kwargs)
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            if take_log:
                sfe = 10**sfe
            ax.loglog(M, sfe, **kwargs)
        
        ax.set_xlabel(r'$M_h / M_{\odot}$')
        ax.set_ylabel(r'$f_{\ast}(M)$')
        ax.set_ylim(1e-4, 1)
        ax.set_xlim(1e7, 1e14)
        pl.draw()

        return ax
        
    def LuminosityFunction(self, z, ax=None, fig=1, compare_to=None, popid=0, 
        name='galaxy_lf', shade_by_like=False, like=0.685, scatter_kwargs={}, 
        Mlim=(-24, -10), samples=1, take_log=False, un_log=False,
        multiplier=1, skip=0, stop=None, **kwargs):
        """
        Plot the luminosity function used to train the SFE.
        
        """
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        if shade_by_like:
            q1 = 0.5 * 100 * (1. - like)    
            q2 = 100 * like + q1
            
        # Plot fits compared to observational data
        M = np.arange(Mlim[0], Mlim[1], 0.05)
        lit = read_lit(compare_to)
    
        if (compare_to is not None) and (z in lit.redshifts) and (not gotax):
            phi = np.array(lit.data['lf'][z]['phi'])
            err = np.array(lit.data['lf'][z]['err'])
            uplims = phi - err <= 0.0
            
            ax.errorbar(lit.data['lf'][z]['M'], lit.data['lf'][z]['phi'],
                yerr=lit.data['lf'][z]['err'], fmt='o', zorder=10,
                uplims=uplims, **scatter_kwargs)

        info = self.blob_info(name)
        ivars = self.blob_ivars[info[0]]

        # We assume that ivars are [redshift, magnitude]
        mags_disk = ivars[1]
        
        # Apply dust correction
        mags_w_dc = mags_disk - self.dc.AUV(z, ivars[1])
        
        loc = np.argmax(self.logL[skip:stop])

        phi = []
        for i, mag in enumerate(mags_disk):
            data = self.ExtractData(name, ivar=[z, mag],
                take_log=take_log, un_log=un_log, multiplier=multiplier)

            if not shade_by_like:
                #if samples > 1:
                #    size = len(data[name][skip:stop])
                #    phi = [data[name][skip:stop][kk] \
                #        for kk in np.random.randint(0, high=samples, size=size)]
                #else:
                phi.append(data[name][skip:stop][loc])
            else:
                lo, hi = np.percentile(data[name][skip:stop], (q1, q2))
                phi.append((lo, hi))    

        if shade_by_like:
            phi = np.array(phi).T

            if take_log:
                phi = 10**phi
            else:
                zeros = np.argwhere(phi == 0)
                for element in zeros:
                    phi[element[0],element[1]] = 1e-15

            ax.fill_between(mags_w_dc, phi[0], phi[1], **kwargs)
            ax.set_yscale('log')
        else:
            if take_log:
                phi = 10**phi
            
            #if samples > 1:
            #    ax.semilogy(mags_w_dc, np.array(phi).T, **kwargs)
            #else:    
            ax.semilogy(mags_w_dc, phi, **kwargs)

        ax.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax.set_ylabel(r'$\phi(M)$')
        ax.set_ylim(1e-8, 10)
        ax.set_xlim(-25, -10)
        pl.draw()

        return ax

    def FaintEndSlope(self, z, mag=None, ax=None, fig=1, N=100, 
        name='alpha_lf', best_only=False, **kwargs):
        """
        
        """
        
        if ax is None:
            gotax = False
            fig = pl.figure(fig)
            ax = fig.add_subplot(111)
        else:
            gotax = True
            
        info = self.blob_info(name)
        ivars = self.blob_ivars[info[0]]
        
        i = list(ivars[0]).index(z)
        M = ivars[1]
        
        loc = np.argmax(self.logL)
        
        alpha = []
        for i, mag in enumerate(M):
            data = self.ExtractData(name, ivar=[z, M[i]])
            
            if best_only:
                alpha.append(data[name][loc])
            else:
                alpha.append(data[name])
         
        alpha = np.array(alpha) 
                
        if best_only:    
            ax.plot(M, alpha, **kwargs)
        else:
            for i in range(N):
                ax.plot(M, alpha[:,i], **kwargs)   
                
        ax.set_xlabel(r'$M_{\mathrm{UV}}$')
        ax.set_ylabel(r'$\alpha$')
        #ax.set_ylim(1e-7, 1e-1)
        #ax.set_xlim(-25, Mlim[1])
        
        return ax         
            
    def AddConstraints(self, z, compare_to, ax):
        """
        
        """
        lit = read_lit(compare_to)

        if (compare_to is not None) and (z in lit.redshifts):
            ax.errorbar(lit.data['lf'][z]['M'], lit.data['lf'][z]['phi'],
                yerr=lit.data['lf'][z]['err'], fmt='o', zorder=10,
                **scatter_kwargs)

    def PlotData(self, z, ax=None, fig=1, **kwargs):
        M, phi, err = self.get_data(z)
        
        ax, gotax = self.get_ax(ax, fig)
            
        ax.errorbar(M, phi, yerr=err, fmt='none', **kwargs)
        ax.set_xlabel(r'$M$')
        ax.set_ylabel(r'$\phi \ (\mathrm{cMpc}^{-3} \ \mathrm{mag}^{-1})$')
        ax.set_yscale('log')
        pl.draw()
        
        return ax
            
    def StellarSED(self, ax=None, fig=1, N=500, src='leitherer1999', **kwargs):
        """
        Make a triangle plot of the stellar SED parameters.
        """
        
        pars = []
        for par in self.parameters:
            if re.search('pop_Z', par):
                pars.append(par)
                break
                
        data = self.ExtractData(pars)
        
        if N > self.chain.shape[0]:
            N = self.chain.shape[0]
            
        src = read_lit(src)
        i_Z = self.parameters.index(pars[0])
        
        Z = []; kappa_UV = []; Nion = []; Nlw = []
        for i in range(500):
            
            j = np.random.randint(0, N)
            
            kw = {par: data[par][j] for par in data}
            
            pop = src.StellarPopulation(**kw)
            
            Z.append(self.chain[j,i_Z])
            kappa_UV.append(pop.kappa_UV)
            Nion.append(pop.Nion)
            Nlw.append(pop.Nlw)
            
        
        parnames = ['pop_Z', 'pop_kappa_UV', 'pop_Nion', 'pop_Nlw']
            
    def alpha_hmf(self, z, M=1e9):
        """
        Compare constraints on the slope of the LF vs. the slope of the HMF.
        """
        
        if not hasattr(self, '_pop'):
            self._pop = ares.populations.GalaxyPopulation(verbose=False)
        
        i_z = np.argmin(np.abs(z - self._pop.halos.z))
        
        alpha_of_M = np.diff(np.log(self._pop.halos.dndm[i_z,:])) \
            / np.diff(self._pop.halos.lnM)
        
        return np.interp(M, self._pop.halos.M[0:-1], alpha_of_M)
            
    def SlopeEvolution(self):
        pass
        
    def MstarEvolution(self):
        pass
    
    def LstarEvolution(self):
        pass
        
    def PredictLF(self, z, name='galaxy_lf', **kwargs):
        """
        Plot luminosity function at redshift z, as predicted by fits.
        """
        
        
        
        lf = self.ExtractData(name)
            
        
            
        
        