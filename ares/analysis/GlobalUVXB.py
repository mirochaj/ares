"""

AnalyzeXray.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Tue Sep 18 13:36:56 2012

Description: 

"""

import numpy as np
import matplotlib.pyplot as pl
from scipy.integrate import trapz
from ..physics.Constants import *

class SyntheticCXRB:
    def __init__(self, fit='moretti2012'):
        """
        Operations on the best fit to the CXRB from Moretti et al. 2009 from 2keV-2MeV.
        
        Energy units are in keV.
        Flux units are keV^2/cm^2/s/keV/sr
        
        Convert energies to eV throughout, change default flux units
        Include results from Markevich & Hickox
        
        """
        
        self.fit = fit
                
        # Fit parameters from Moretti et al. 2009 (Model = 2SJPL, Table 2)
        if fit in ['moretti2009', 'moretti2009+2SJPL']:
            self.C = 0.109 / sqdeg_per_std / 1e3    # now photons / s / cm^2 / deg^2 / eV
            self.C_err = 0.003 / sqdeg_per_std / 1e3
            self.Gamma1 = 1.4
            self.Gamma1_err = 0.02
            self.Gamma2 = 2.88
            self.Gamma2_err = 0.05
            self.EB = 29.0 * 1e3
            self.EB_err = 0.5 * 1e3
            self.integrated_2_10_kev_flux = 2.21e-11
            self.sigma_integrated_2_10_kev_flux = 0.07e-11
        elif fit == 'moretti2009+PL':
            self.C = 3.68e-3
        elif fit == 'moretti2012':
            self.unresolved_2_10_kev_flux = 5e-12
            self.sigma_unresolved_2_10_kev_flux = 1.77e-12
            
        # Some defaults
        self.E = np.logspace(2, 5, 100)    # eV
    
    def ResolvedFlux(self, E=None, perturb=False):
        """
        Return total CXRB flux (resolved + unresolved) in erg / s / cm^2 / deg^2.
        Plotting this should reproduce Figure 11 in Moretti et al. 2009 (if 
        fit == 1) modulo the units.
        """
        
        if E is None:
            E = self.E
        
        # Randomly vary model parameters assuming their 1-sigma (Gaussian) errors
        if perturb:
            Cerr = np.random.normal(scale = self.C_err)
            EBerr = np.random.normal(scale = self.EB_err)
            Gamma1err = np.random.normal(scale = self.Gamma1_err)
            Gamma2err = np.random.normal(scale = self.Gamma2_err)
        else:
            Cerr = EBerr = Gamma1err = Gamma2err = 0.0
        
        # Return SWIFT-BAT measured CXRB flux at energy E (keV).
        if self.fit == 'moretti2009':
            self.integrated_2_10_kev_flux = 2.21e-11
            self.sigma_integrated_2_10_kev_flux = 0.07e-11
            
            flux = E**2 * (self.C + Cerr) \
                / ((E / (self.EB + EBerr))**(self.Gamma1 + Gamma1err) \
                + (E / (self.EB + EBerr))**(self.Gamma2 + Gamma2err))
                 
            return flux * erg_per_ev # erg / s / cm^2 / deg^2
            
    def IntegratedFlux(self, Emin=2e3, Emax=1e4, Nbins=1e3, perturb=False):
        """
        Integrated flux in [Emin, Emax] (eV) band.
        """        
        
        E = np.logspace(np.log10(Emin), np.log10(Emax), Nbins) 
        F = self.ResolvedFlux(E, perturb=perturb) / E
        
        return trapz(F, E)  # erg / s / cm^2 / deg^2
   
    def Plot(self, color = 'k'):
        """
        Plot measured CXRB nicely. 
        """
        
        self.ax = pl.subplot(111)
        self.ax.loglog(self.E, self.MeasuredFlux(), color = color)
        self.ax.set_xlabel(r'Energy (eV)')
        self.ax.set_ylabel(r'Flux Density $(\mathrm{erg / s / cm^2 / deg^2})$')
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')        
        pl.draw()
   
