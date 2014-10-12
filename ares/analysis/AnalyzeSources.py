"""

AnalyzeSources.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sat Mar 23 19:21:46 2013

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
from ..physics.Constants import *
from scipy.integrate import quad as integrate
from ..physics.CrossSections import PhotoIonizationCrossSection as sigma_E

allls = ['-', '--', '-.', ':']
small_number = 1e-5

class Source:
    def __init__(self, rs):
        self.rs = rs
        
    def SpectrumCDF(self, E):
        """
        Returns cumulative energy output contributed by photons at or less 
        than energy E.
        """    
        
        return integrate(self.rs.Spectrum, small_number, E)[0] 
    
    def SpectrumMedian(self, energies = None):
        """
        Compute median emission energy from spectrum CDF.
        """
        
        if energies is None:
            energies = np.linspace(self.rs.EminNorm, self.rs.EmaxNorm, 200)
        
        if not hasattr('self', 'cdf'):
            cdf = []
            for energy in energies:
                cdf.append(self.SpectrumCDF(energy))
                
            self.cdf = np.array(cdf)
            
        return np.interp(0.5, self.cdf, energies)
    
    def SpectrumMean(self):
        """
        Mean emission energy.
        """        
        
        integrand = lambda E: self.rs.Spectrum(E) * E
        
        return integrate(integrand, self.rs.EminNorm, self.rs.EmaxNorm)[0]
        
    def EscapeFraction(self, i=0, logN=0.0, weighted=True):
        if logN <= 0.0:
            return 1.0
                    
        if weighted:
            integrand1 = lambda E: self.rs.Spectrum(E) * sigma_E(E) \
                * np.exp(-10.**logN \
                * (sigma_E(E, 0) + y * sigma_E(E, 1))) / E
            integrand2 = lambda E: self.rs.Spectrum(E) * sigma_E(E) / E 
        else:
            integrand1 = lambda E: self.rs.Spectrum(E) * sigma_E(E) \
                * np.exp(-10.**logN \
                * (sigma_E(E, 0) + y * sigma_E(E, 1))) / E
            integrand2 = lambda E: self.rs.Spectrum(E) * sigma_E(E) / E
        
        fesc_theory = integrate(integrand1, self.rs.spec_pars['Emin'][i], 
            self.rs.spec_pars['Emax'][i])[0] / integrate(integrand2, 
                self.rs.spec_pars['Emin'][i], 
                self.rs.spec_pars['Emax'][i])[0]
        
        return fesc_theory, self.SpecificEscapeFraction(E=E_LL, i=i, logN=logN)
    
    def SpecificEscapeFraction(self, E, i=0, logN=0.0):
        if logN <= 0.0:
            return 1.0
            
        if E < E_LL:
            return 1.0    
    
        return np.exp(-10.**logN \
            * (sigma_E(E, 0) + y * sigma_E(E, 1)))
                    
    def PlotSpectrum(self, color='k', components=True, t=0, normalized=True,
        bins=100, ax=None, label=None, ls='-', xunit='eV', marker=None,
        normalize_to=None):
        """
        
        Parameters
        ----------
        normalize_to : list, tuple
            Normalize such that at energy normalize_to[0], the intensity is
            normalize_to[1]
        """
        
        if not normalized:
            Lbol = self.rs.BolometricLuminosity(t)
        else: 
            Lbol = 1.
        
        E = np.logspace(np.log10(self.rs.Emin), np.log10(self.rs.Emax), bins)
        F = []
        
        for energy in E:
            F.append(self.rs.Spectrum(energy, t = t))
        
        if components and self.rs.N > 1:
            EE = []
            FF = []
            for i, component in enumerate(self.rs.SpectrumPars['type']):
                tmpE = np.logspace(np.log10(self.rs.SpectrumPars['Emin'][i]), 
                    np.log10(self.rs.SpectrumPars['Emax'][i]), bins)
                tmpF = []
                for energy in tmpE:
                    tmpF.append(self.rs.Spectrum(energy, t=t, i=i))
                
                EE.append(tmpE)
                FF.append(tmpF)
        
        if ax is None:
            ax = pl.subplot(111)
                    
        if xunit == 'keV':
            E = np.array(E) / 1e3
            F = np.array(F) * 1e3
        else:
            E = np.array(E)
            F = np.array(F)
            
        if normalize_to is not None:
            norm = normalize_to[1] / F[np.argmin(np.abs(E - normalize_to[0]))]
        else:
            norm = 1
                    
        self.E, self.F = E, F    
        if marker is None:        
            ax.loglog(E, F * Lbol * norm, color=color, ls=ls, 
                label=label)
        else:
            ax.scatter(E, F * Lbol * norm, color=color, marker=marker, 
                label=label)
        
        if components and self.rs.N > 1:
            for i in xrange(self.rs.N):
                ax.loglog(EE[i], np.array(FF[i]) * Lbol, color=color, 
                    ls=allls[i+1])
        
        if xunit == 'keV':
            ax.set_xlabel(r'$h\nu \ (\mathrm{keV})$')
        else:
            ax.set_xlabel(r'$h\nu \ (\mathrm{eV})$')
        
        if normalized:
            ax.set_ylabel(r'$L_{\nu} / L_{\mathrm{bol}}$')
        else:
            ax.set_ylabel(r'$L_{\nu} \ (\mathrm{erg \ s^{-1} \ \mathrm{keV}^{-1}})$')
                
        pl.draw()
              
        return ax 
            