"""

ChemicalNetwork.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 13:15:30 2012

Description: ChemicalNetwork object just needs to have methods called
'RateEquations' and 'Jacobian'

"""

import copy, sys
import numpy as np
from scipy.misc import derivative
from scipy.special import erf
from ..util.Warnings import solver_error
from ..physics.RateCoefficients import RateCoefficients
from ..physics.Constants import k_B, sigma_T, m_e, c, s_per_myr, erg_per_ev, h, m_H
        
rad_const = (8. * sigma_T / 3. / m_e / c)
        
class ChemicalNetwork(object):
    def __init__(self, grid, rate_src='fk94', recombination='B', 
        interp_rc='linear'):
        """
        Initialize chemical network.
        
        grid: ares.static.Grid.Grid instance
        rate_src : str
            
        """
        self.grid = grid
        self.cosm = self.grid.cosm
        
        self.coeff = RateCoefficients(grid, rate_src=rate_src,
            recombination=recombination, interp_rc=interp_rc)

        self.isothermal = self.grid.isothermal
        self.include_dm = self.grid.include_dm
        self.secondary_ionization = self.grid.secondary_ionization
        
        # For convenience 
        self.zeros_q = np.zeros(len(self.grid.evolving_fields))
        self.zeros_jac = np.zeros([len(self.grid.evolving_fields)] * 2)
        
        # Faster because we bypass the hasattr in grid property
        self.absorbers = self.grid.absorbers
        self.ions = self.grid.ions
        self.neutrals = self.grid.neutrals
        self.expansion = self.grid.expansion
        self.exotic_heating = self.grid.exotic_heating
        self.isothermal = self.grid.isothermal
        self.is_cgm_patch = self.grid.is_cgm_patch        
        self.is_igm_patch = not self.grid.is_cgm_patch
        self.collisional_ionization = self.grid.collisional_ionization
                
        self.Nev = len(self.grid.evolving_fields)
        
        self.include_He = 2 in self.grid.Z
        self.y = self.cosm.y
        
        if not self.expansion:
            self.C = self.grid.clumping_factor(0.0)

    @property
    def monotonic_EoR(self):
        if not hasattr(self, '_monotonic_EoR'):
            self._monotonic_EoR = False
        
        return self._monotonic_EoR
    
    @monotonic_EoR.setter
    def monotonic_EoR(self, value):
        self._monotonic_EoR = value
    
    def RateEquations(self, t, q, args):
        """
        Compute right-hand side of rate equation ODEs.
    
        Equations 1, 2, 3 and 9 in Mirocha et al. (2012), except
        we're solving for ion fractions instead of number densities.
    
        Parameters
        ----------
        t : float
            Current time.
        q : np.ndarray
            Array of dependent variables, one per rate equation.
        args : list
            Extra information needed to compute rates. They are, in order:
            [cell #, ionization rate coefficient (IRC), secondary IRC,
             photo-heating rate coefficient, particle density, time]
        """       
    
        self.q = q
    
        cell, k_ion, k_ion2, k_heat, ntot, time = args
            
        to_temp = 1. / (1.5 * ntot * k_B)
                    
        if self.expansion:
            z = self.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.cosm.nH(z)
            CF = self.grid.clumping_factor(z)
        else:
            n_H = self.grid.n_H[cell]
            CF = self.C

        if self.include_He:
            y = self.grid.element_abundances[1]
            n_He = self.grid.element_abundances[1] * n_H
        else:
            y = 0.0
            n_He = 0.0

        # Read q vector quantities into dictionaries
        x = {k: q[self.grid.fields_key[k]] for k in self.grid.fields_key}
        n = {'h': n_H, 'he': n_He}
        n_e = x['e'] * n_H

        xe = n_e / n_H
        
        # In two-zone model, this phase is assumed to be fully ionized
        # CF = clumping factor
        if self.is_cgm_patch:            
            CF *= (n_H * (1. + y) / n_e)
                    
        # Where do the electrons live?
        if self.Nev == 6:
            e = -1
        elif self.Nev == 7:
            e = -2
        else:
            e = 2

        if self.include_He:
            xi = self.xi
            omega = self.omega
 
        # Store results here
        dqdt = {field:0.0 for field in self.grid.evolving_fields}
        
        # Correct for H/He abundances
        acorr = {'h_1': 1., 'he_1': y, 'he_2': y}

        ##
        # Secondary ionization (of hydrogen)
        ##
        gamma_HI = 0.0
        if self.secondary_ionization > 0:

            for j, donor in enumerate(self.absorbers):
                elem = self.grid.parents_by_ion[donor]

                term = k_ion2[0,j] * (x[donor] / x['h_1']) \
                     * (acorr[donor] / acorr['h_1'])
                gamma_HI += term
              
        ##
        # Hydrogen rate equations
        ##
        dqdt['h_1'] = -(k_ion[0] + gamma_HI + self.Beta[cell,0] * n_e) \
                      * x['h_1'] \
                      + self.alpha[cell,0] * n_e * x['h_2'] * CF
        dqdt['h_2'] = -dqdt['h_1']
        
        ##
        # Heating & cooling
        ##

        # NOTE: cooling term multiplied by electron density at the very end!

        heat = 0.0
        cool = 0.0
        if not self.isothermal:
            
            for i, sp in enumerate(self.neutrals):
                elem = self.grid.parents_by_ion[sp]
                
                heat += k_heat[i] * x[sp] * n[elem]          # photo-heating
            
                cool += self.zeta[cell,i] * x[sp] * n[elem]  # ionization
                cool += self.psi[cell,i] * x[sp] * n[elem]   # excitation

            for i, sp in enumerate(self.ions):
                elem = self.grid.parents_by_ion[sp]
                
                cool += self.eta[cell,i] * x[sp] * n[elem]   # recombination
                                                            
        ##
        # Helium processes
        ##
        if self.include_He:
            
            # Secondary ionization  
            gamma_HeI = 0.0
            gamma_HeII = 0.0 
            if self.secondary_ionization > 0: 
                                
                for j, donor in enumerate(self.absorbers):
                    elem = self.grid.parents_by_ion[donor]
                
                    term1 = k_ion2[1,j] * (x[donor] / x['he_1']) \
                          * (acorr[donor] / acorr['he_1'])
                    gamma_HeI += term1
                    
                    term2 = k_ion2[2,j] * (x[donor] / x['he_2']) \
                          * (acorr[donor] / acorr['he_2'])
                    gamma_HeII += term2
            
            ##
            # Helium rate equations
            ##
            dqdt['he_1'] = \
                - x['he_1'] * (k_ion[1] + gamma_HeI + self.Beta[cell,1] * n_e) \
                + x['he_2'] * (self.alpha[cell,1] + xi[cell]) * n_e
            
            dqdt['he_2'] = \
                  x['he_1'] * (k_ion[1] + gamma_HeI + self.Beta[cell,1] * n_e) \
                - x['he_2'] * (k_ion[2] + gamma_HeII \
                + (self.Beta[cell,2] + self.alpha[cell,1] + xi[cell]) * n_e) \
                + x['he_3'] * self.alpha[cell,2] * n_e
        
            dqdt['he_3'] = \
                  x['he_2'] * (k_ion[2] + gamma_HeII + self.Beta[cell,2] * n_e) \
                - x['he_3'] * self.alpha[cell,2] * n_e
            
            # Dielectronic recombination cooling    
            if not self.isothermal:
                cool += omega[cell] * x['he_2'] * n_He
                    
        ##            
        # Electrons
        ##
        
        # Gains from ionizations of HI
        dqdt['e'] = 1. * dqdt['h_2']
        
        # Electrons from helium ionizations
        if self.include_He:
            # Gains from ionization of HeI
            dqdt['e'] += y * x['he_1'] \
                * (k_ion[1] + gamma_HeI + self.Beta[cell,1] * n_e)
            
            # Gains from ionization of HeII
            dqdt['e'] += y * x['he_2'] \
                * (k_ion[2] + gamma_HeII + self.Beta[cell,2] * n_e)
                
            # Losses from HeII recombinations
            dqdt['e'] -= y * x['he_2'] \
                * (self.alpha[cell,1] + xi[cell]) * n_e
                
            # Losses from HeIII recombinations
            dqdt['e'] -= y * x['he_3'] * self.alpha[cell,2] * n_e
            
        # Finish heating and cooling
        if not self.isothermal:
            hubcool = 0.0
            compton = 0.0
            
            # Hubble Cooling
            if self.expansion:
                hubcool = 2. * self.cosm.HubbleParameter(z) * x['Tk']

                # Compton cooling
                if self.grid.compton_scattering:
                    Tcmb = self.cosm.TCMB(z)
                    ucmb = self.cosm.UCMB(z)

                    # Seager, Sasselov, & Scott (2000) Equation 54
                    compton = rad_const * ucmb * n_e * (Tcmb - x['Tk']) / ntot
            
            if self.grid.cosm.pf['approx_thermal_history']:
                dqdt['Tk'] = heat * to_temp \
                    - self.cosm.cooling_rate(z, x['Tk']) / self.cosm.dtdz(z)
            else:    
                dqdt['Tk'] = (heat - n_e * cool) * to_temp + compton \
                    - hubcool - q[-1] * n_H * dqdt['e'] / ntot
    
        else:
            dqdt['Tk'] = 0.0

        if self.include_dm and not self.isothermal:
            Tk, Tchi, Vchib = x['Tk'], x['Tchi'], x['Vchib']
            dTk_dt, dTchi_dt, dVchib_dt = self.dm_heating(z, Tk, Tchi, Vchib)

            H = self.cosm.HubbleParameter(z)
            dqdt['Tchi'] = -2*H*x['Tchi'] + dTchi_dt
            dqdt['Tk'] += dTk_dt
            dqdt['Vchib'] = -H*Vchib + dVchib_dt

            if self.isothermal:
                dqdt['Tchi'] = 0
                dqdt['Tk'] = 0
                dqdt['Vchib'] = 0
        ##
        # Add in exotic heating
        ##    
        if self.exotic_heating:
            dqdt['Tk'] += self.grid._exotic_func(z=z) * to_temp

        # Can effectively turn off ionization equations once EoR is over.
        if self.monotonic_EoR:
            if x['h_1'] <= self.monotonic_EoR:
                dqdt['h_1'] = dqdt['h_2'] = 0.0
            if self.include_He:
                if x['he_1'] <= self.monotonic_EoR:
                    dqdt['he_1'] = 0.0
                if x['he_2'] <= self.monotonic_EoR:
                    dqdt['he_2'] = 0.0
                        
        self.dqdt = self.zeros_q.copy()
        for i, sp in enumerate(self.grid.qmap):
            self.dqdt[i] = dqdt[sp]

        if np.isnan(self.dqdt).sum():
            raise ValueError('NaN encountered in RateEquations!')
        if (self.q < 0).sum():
            solver_error(self.grid, -1000, [self.q], [self.dqdt], -1000, cell, -1000)
            raise ValueError('Something < 0.')

        return self.dqdt
                          
    def Jacobian(self, t, q, args):
        """
        Compute the Jacobian for the system of equations.
        """
        self.q = q
        self.dqdt = self.zeros_q.copy()
    
        cell, k_ion, k_ion2, k_heat, ntot, time = args
                
        to_temp = 1. / (1.5 * ntot * k_B)
    
        if self.expansion:
            z = self.cosm.TimeToRedshiftConverter(0., time, self.grid.zi)
            n_H = self.cosm.nH(z)
            CF = self.grid.clumping_factor(z)
        else:
            n_H = self.grid.n_H[cell]
            CF = self.C

        x = {k: q[self.grid.fields_key[k]] for k in self.grid.fields_key}
        n = {'h': n_H, 'he': self.Y * n_H}
        ne = x['e'] * n_H

        if self.include_He:
            y = self.grid.element_abundances[1]
            n_He = self.grid.element_abundances[1] * n_H
        else:
            y = 0.0
            n_He = 0.0
            
        # Correct for H/He abundances
        acorr = {'h_1': 1., 'he_1': y, 'he_2': y}    
        
        xe = n_e / n_H

        if self.is_cgm_patch:
            CF *= (n_H * (1. + y) / n_e)

        xi = self.xi
        omega = self.omega
    
        # For Jacobian
        if not self.isothermal and self.include_He:
            dxi = self.dxi
            domega = self.domega
    
        J = self.zeros_jac.copy()
        
        # Where do the electrons live?
        if self.Nev == 6:
            e = -1
        elif self.Nev == 7:
            e = -2
        else:
            e = 2
            
        ##
        # Secondary ionization (of hydrogen)
        ##
        gamma_HI = 0.0
        if self.secondary_ionization > 0:
        
            for j, donor in enumerate(self.absorbers):
                elem = self.grid.parents_by_ion[donor]
                term = k_ion2[0,j] * (x[donor] / x['h_1']) \
                     * (acorr[donor] / acorr['h_1'])
                gamma_HI += term    
        
        ##
        # FIRST: HI and HII terms. Always in slots 0 and 1.
        ##
            
        # HI by HI
        J[0,0] = -(k_ion[0] + gamma_HI + self.Beta[cell,0] * n_e)
        
        # HI by HII
        J[0,1] = self.alpha[cell,0] * n_e * CF
         
        # HII by HI
        J[1,0] = -J[0,0]
        
        # HII by HII
        J[1,1] = -J[0,1]
                    
        ##    
        # Hydrogen-Electron terms
        ##
        
        J[0,e] = -self.Beta[cell,0] * x['h_1'] \
               + self.alpha[cell,0] * x['h_2'] * CF
        J[1,e] = -J[0,e]
        J[e,e] = n_H * J[1,e]
           
        J[e,0] = n_H * (k_ion[0]+ gamma_HI + self.Beta[cell,0] * n_e)
        J[e,1] = -n_H * self.alpha[cell,0] * n_e * CF
            
        ###
        ## HELIUM INCLUDED CASES: N=6 (isothermal), N=7 (thermal evolution)   
        ###
        if self.Nev in [6, 7]:
            
            # Secondary ionization  
            gamma_HeI = 0.0
            gamma_HeII = 0.0 
            if self.secondary_ionization > 0: 
                                
                for j, donor in enumerate(self.absorbers):
                    elem = self.grid.parents_by_ion[donor]
                
                    term1 = k_ion2[1,j] * (x[donor] / x['he_1']) \
                          * (acorr[donor] / acorr['he_1'])
                    gamma_HeI += term1
                    
                    term2 = k_ion2[2,j] * (x[donor] / x['he_2']) \
                          * (acorr[donor] / acorr['he_2'])
                    gamma_HeII += term2
            
            # HeI by HeI
            J[2,2] = -(k_ion[1] + gamma_HeI + self.Beta[cell,1] * n_e)
            
            # HeI by HeII
            J[2,3] = (self.alpha[cell,1] + xi[cell]) * n_e
            
            # HeI by HeIII
            J[2,4] = 0.0

            # HeII by HeI
            J[3,2] = -J[2,2]

            # HeII by HeII
            J[3,3] = -(k_ion[2] + gamma_HeII) \
                   - (self.Beta[cell,2] + self.alpha[cell,1] + xi[cell]) * n_e

            # HeII by HeIII
            J[3,4] = self.alpha[cell,2] * n_e
            
            # HeIII by HeI
            J[4,2] = 0.0

            # HeIII by HeII
            J[4,3] = k_ion[2] + self.Beta[cell,2] * n_e

            # HeIII by HeIII
            J[4,4] = -self.alpha[cell,2] * n_e

            ##
            # Helium-Electron terms
            ##

            J[2,e] = -self.Beta[cell,1] * x['he_1'] \
                   + (self.alpha[cell,1] + xi[cell]) * x['he_2']
            J[3,e] = self.Beta[cell,1] * x['he_1'] \
                   - (self.Beta[cell,2] + self.alpha[cell,1] + xi[cell]) * x['he_2'] \
                   + self.alpha[cell,2] * x['he_3']
            J[4,e] = self.Beta[cell,2] * x['he_2'] - self.alpha[cell,2] * x['he_3']
            
            J[e,2] = n_He \
                * (k_ion[1] + gamma_HeI + self.Beta[cell,1] * n_e)
            
            J[e,3] = n_He \
                * ((k_ion[2] + gamma_HeII + self.Beta[cell,2] * n_e) \
                - (self.alpha[cell,1] + xi[cell]) * n_e)
            
            J[e,4] = -n_He * self.alpha[cell,2] * n_e
            
            # Electron-electron terms (increment from H-only case)
            J[e,e] += n_He * x['he_1'] * self.Beta[cell,1]
            
            J[e,e] += n_He \
                * (x['he_2'] \
                * ((self.Beta[cell,2] - (self.alpha[cell,1] + xi[cell]))) \
                - x['he_3'] * self.alpha[cell,2])
                
        ##
        # Heating/Cooling from here onwards
        ##
        if self.isothermal:
            return J              

        ##
        # Hydrogen derivatives wrt Tk
        ##
        
        # HI by Tk
        J[0,-1] = -n_e * x['h_1'] * self.dBeta[cell,0] \
                +  n_e * x['h_2'] * self.dalpha[cell,0] * CF
        # HII by Tk
        J[1,-1] = -J[0,-1]
        
        ##
        # Helium derivatives wrt Tk
        ##
        if self.include_He:  
            # HeI by Tk
            J[2,-1] = -n_e * (x['he_1'] * self.dBeta[cell,1] \
                    - x['he_2'] * (self.dalpha[cell,1] + dxi[cell]))

            # HeII by Tk
            J[3,-1] = -n_e * (x['he_2'] * (self.dBeta[cell,2] \
                    + self.dalpha[cell,1] + dxi[cell]) \
                    - x['he_3'] * self.dalpha[cell,2])

            # HeIII by Tk
            J[4,-1] = n_e * (x['he_2'] * self.dBeta[cell,2] \
                - x['he_3'] * self.dalpha[cell,2])

        ##
        # Electron by Tk terms
        ##
        J[e,-1] = n_H * n_e \
            * (x['h_1'] * self.dBeta[cell,0] - x['h_2'] * self.dalpha[cell,0] * CF)
        
        ##
        # A few last Tk by Tk and Tk by electron terms (dielectronic recombination)
        ##
        if self.include_He:            
            J[e,-1] += n_He * n_e \
                * (self.dBeta[cell,1] * x['he_1'] + self.dBeta[cell,2] * x['he_2'] \
                - (self.dalpha[cell,1] + dxi[cell]) * x['he_2'] \
                - self.dalpha[cell,2] * x['he_3'])
                
        ##
        # Tk derivatives wrt neutrals
        ##
        for i, sp in enumerate(self.neutrals):
            j = self.grid.qmap.index(sp)
            elem = self.grid.parents_by_ion[sp]

            # Photo-heating
            J[-1,j] += n[elem] * k_heat[i]
            
            # Collisional ionization cooling
            J[-1,j] -= n[elem] * self.zeta[cell,i] * n_e
            
            # Collisional excitation cooling
            J[-1,j] -= n[elem] * self.psi[cell,i] * n_e
            
        ##
        # Tk derivatives wrt ions (only cooling terms)
        ##
        for i, sp in enumerate(self.ions):
            j = self.grid.qmap.index(sp)
            elem = self.grid.parents_by_ion[sp]

            # Recombination cooling
            J[-1,j] -= n[elem] * self.eta[cell,i] * n_e
            
        # Dielectronic recombination term
        if self.include_He:
            J[-1,3] -= n_He * omega[cell] * n_e
        
        ##
        # Tk by Tk terms and Tk by electron terms
        ##
        for i, sp in enumerate(self.absorbers):
            elem = self.grid.parents_by_ion[sp]
            J[-1,-1] -= n_e * n[elem] * x[sp] * self.dzeta[cell,i]
            J[-1,-1] -= n_e * n[elem] * x[sp] * self.dpsi[cell,i]

            J[-1,e] -= n[elem] * x[sp] * self.zeta[cell,i]
            J[-1,e] -= n[elem] * x[sp] * self.psi[cell,i]

        for i, sp in enumerate(self.ions):
            elem = self.grid.parents_by_ion[sp]
            J[-1,-1] -= n_e * n[elem] * x[sp] * self.deta[cell,i]
                
            J[-1,e] -= n[elem] * x[sp] * self.eta[cell,i]
         
        # Dielectronic recombination term
        if self.include_He: 
            J[-1,e] -= omega[cell] * n_He * x['he_2']      
            J[-1,-1] -= n_e * x['he_2'] * n_He * self.domega[cell]
            
        # So far, everything in units of energy, must convert to temperature    
        J[-1,:] *= to_temp
        
        # Energy distributed among particles
        J[-1,-1] -= n_H * self.dqdt[e] / ntot

        # Cosmological effects
        if self.expansion:

            # These terms have the correct units already
            J[-1,-1] -= 2. * self.cosm.HubbleParameter(z)

            if self.grid.compton_scattering:
                Tcmb = self.cosm.TCMB(z)
                ucmb = self.cosm.UCMB(z)
                tcomp = 3. * m_e * c / (8. * sigma_T * ucmb)

                J[-1,-1] -= q[-1] * xe \
                    / tcomp / (1. + self.y + xe)
                J[-1,e] -= (Tcmb - q[-1]) * (1. + self.y) \
                    / (1. + self.y + xe)**2 / tcomp


        # Add in any parametric modifications?
        if self.exotic_heating:
            J[-1,-1] += derivative(self.grid._exotic_func(z=z) * to_temp, z,
                dx=0.05)
        
        return J

    def SourceIndependentCoefficients(self, T, z=None):
        """
        Compute values of rate coefficients which depend only on 
        temperature and/or number densities of electrons/ions.
        """    
                
        self.T = T
        self.Beta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.alpha = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.zeta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.eta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        self.psi = np.zeros_like(self.grid.zeros_grid_x_absorbers)
        
        self.xi = np.zeros(self.grid.dims)
        self.omega = np.zeros(self.grid.dims)
        
        if not self.isothermal:
            self.dBeta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.dalpha = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.dzeta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.deta = np.zeros_like(self.grid.zeros_grid_x_absorbers)
            self.dpsi = np.zeros_like(self.grid.zeros_grid_x_absorbers)

        for i, absorber in enumerate(self.absorbers):

            if self.collisional_ionization:
                self.Beta[...,i] = self.coeff.CollisionalIonizationRate(i, T)

            self.alpha[...,i] = self.coeff.RadiativeRecombinationRate(i, T)

            if self.isothermal:
                continue
            
            self.dalpha[...,i] = self.coeff.dRadiativeRecombinationRate(i, T)
                
            if self.collisional_ionization:
                self.zeta[...,i] = self.coeff.CollisionalIonizationCoolingRate(i, T)
                self.dzeta[...,i] = self.coeff.dCollisionalIonizationCoolingRate(i, T)
                self.dBeta[...,i] = self.coeff.dCollisionalIonizationRate(i, T)
                
            self.eta[...,i] = self.coeff.RecombinationCoolingRate(i, T)
            self.psi[...,i] = self.coeff.CollisionalExcitationCoolingRate(i, T)

            self.deta[...,i] = self.coeff.dRecombinationCoolingRate(i, T)
            self.dpsi[...,i] = self.coeff.dCollisionalExcitationCoolingRate(i, T)

        # Di-electric recombination
        if self.include_He:
            self.xi = self.coeff.DielectricRecombinationRate(T)
            self.dxi = self.coeff.dDielectricRecombinationRate(T)
            
            if not self.isothermal:
                self.omega = self.coeff.DielectricRecombinationCoolingRate(T)
                self.domega = self.coeff.dDielectricRecombinationCoolingRate(T)
                        
        return {'Beta': self.Beta, 'alpha': self.alpha,
                'zeta': self.zeta, 'eta': self.eta, 'psi': self.psi,
                'xi': self.xi, 'omega': self.omega}

    def dm_heating(self, z, Tb, Tchi, Vchib):
        """
        Dark Matter heating differential equations.
        Equations 16 - 20 of Munoz et al. 2015

        Returns dTb_dt, dTchi_dt, and dVchib_dt components from heating and
        drag, not including the change due to hubble expansion.

        Parameters
        ------------
        z : float
            redshift.
        Tb : float
            baryon temperature
        Tchi : float
            Dark matter temp
        Vchib : float
            DM-b relative velocity
        """
        #### MAKE SURE THESE ALL HAVE THE RIGHT UNITS!!!
        #### Putting everything in [K]/[sec] because of reasons...
        mb = m_H * c ** 2 / k_B  # Baryon mass [K]
        mchi = self.cosm.m_dmeff * 1e9 / k_B  # DM mass [K]

        sig = self.cosm.sigma_dmeff * c ** 2

        rho_chi = self.cosm.MeanDarkMatterDensity(z) / k_B / c
        rho_m = self.cosm.MeanMatterDensity(z) / k_B / c
        rho_b = self.cosm.MeanBaryonDensity(z) / k_B / c

        uth = np.sqrt(Tb / mb + Tchi / mchi)
        r = Vchib / uth
        F = erf(r / np.sqrt(2)) - np.sqrt(2 / np.pi) * np.exp(-r ** 2 / 2) * r

        if Vchib == 0:
            drag = 0
        else:
            drag = rho_m * sig * F / (mb + mchi) / Vchib ** 2

        # Equation (16) from Munoz et al. 2015
        dQb_dt1 = 2 * mb * rho_chi * sig * (Tchi - Tb) * np.exp(-r ** 2 / 2)
        dQb_dt1 /= (mchi + mb) ** 2 * np.sqrt(2 * np.pi) * uth ** 3
        dQb_dt2 = rho_chi / rho_m * (mchi * mb) / (mchi + mb) * Vchib * drag
        dQb_dt = dQb_dt1 + dQb_dt2

        dQchi_dt1 = 2 * mchi * rho_b * sig * (Tb - Tchi) * np.exp(-r ** 2 / 2)
        dQchi_dt1 /= (mchi + mb) ** 2 * np.sqrt(2 * np.pi) * uth ** 3
        dQchi_dt2 = rho_b / rho_m * (mchi * mb) / (mchi + mb) * Vchib * drag
        dQchi_dt = dQchi_dt1 + dQchi_dt2

        # Equations 18 - 20 of munoz et al.
        dTchi_dt = 2./3. * dQchi_dt
        dTb_dt = 2/3 * dQb_dt
        dVchib_dt = -drag

        return dTb_dt, dTchi_dt, dVchib_dt

