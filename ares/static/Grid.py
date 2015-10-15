"""

Grid.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Thu Sep 20 14:18:27 2012

Description: 

"""

import copy, types
import numpy as np
from ..util.Stats import rebin
from collections import Iterable
from ..physics.Hydrogen import Hydrogen
from ..physics.Cosmology import Cosmology
from ..util.ParameterFile import ParameterFile
from ..physics.CrossSections import PhotoIonizationCrossSection
from ..physics.Constants import k_B, cm_per_kpc, s_per_myr, m_H, mH_amu, \
    mHe_amu

class fake_chianti:
    def __init__(self):
        pass

    def z2element(self, i):
        if i == 1:
            return 'h'
        elif i == 2:
            return 'he'

    def element2z(self, name):
        if name == 'h':
            return 1
        elif name == 'he':
            return 2   

    def zion2name(self, Z, i):
        if Z == 1:
            if i == 1:
                return 'h_1'
            elif i == 2:
                return 'h_2'
        elif Z == 2:
            if i == 1:
                return 'he_1'
            elif i == 2:
                return 'he_2'
            elif i == 3:
                return 'he_3'             

    def convertName(self, species):
        element, i = species.split('_')
        
        Z = self.element2z(element)
        
        tmp = {}
        tmp['Element'] = element
        tmp['Ion'] = self.zion2name(Z, int(i))
        tmp['Z'] = self.element2z(element)

        return tmp
        
util = fake_chianti()

tiny_number = 1e-8  # A relatively small species fraction

class Grid(object):
    def __init__(self, **kwargs):
        """
        Initialize grid object.
        
        Parameters
        ----------
        dims : int
            Number of resolution elements in grid.
        length_units : float
            Size of domain in centimeters.
        start_radius : float
            Radius (in code units) within which to ignore.
            
        """
        
        self.pf = ParameterFile(**kwargs)
        
        self.dims = int(self.pf['grid_cells'])
        self.length_units = self.pf['length_units']
        self.start_radius = self.pf['start_radius']
        self.approx_Salpha = self.pf['approx_Salpha']
        self.log_grid = self.pf['logarithmic_grid']

        # Compute cell centers and edges
        if self.pf['logarithmic_grid']:
            self.r_edg = self.r = \
                np.logspace(np.log10(self.R0), np.log10(self.length_units), 
                self.dims + 1)            
        else:
            self.r_edg = self.r = \
                np.linspace(self.R0, self.length_units, self.dims + 1)
        
        # Compute interior cell walls, spacing, and mid-points        
        self.r_int = self.r_edg[0:-1]
        self.dr = np.diff(self.r_edg)
        self.r_mid = rebin(self.r_edg)
        
        self.zi = 0
        
        # Override, to set ICs by cosmology
        self.cosmological_ics = self.pf['cosmological_ics']
                
    @property
    def zeros_absorbers(self):
        return np.zeros(self.N_absorbers)
    
    @property
    def zeros_absorbers2(self):
        return np.zeros([self.N_absorbers] * 2)    
    
    @property
    def zeros_grid_x_absorbers(self):
        return np.zeros([self.dims, self.N_absorbers])
        
    @property
    def zeros_grid_x_absorbers2(self):
        return np.zeros([self.dims, self.N_absorbers, self.N_absorbers])     
                
    @property
    def R0(self):
        """ Start radius in length_units. """
        return self.start_radius * self.length_units
        
    @property
    def Vsh(self):
        """ Shell volume in length_units**3. """
        if not hasattr(self, '_Vsh_all'):
            self._Vsh_all = self.ShellVolume(self.r_edg[0:-1], self.dr)
            
        return self._Vsh_all
                
    @property            
    def neutrals(self):
        """ Return list of all neutral species. """            
        if not hasattr(self, '_neutral_species'):
            self._neutral_species = []
            for element in self.elements:
                self._neutral_species.append('%s_1' % element)

        return self._neutral_species
                    
    @property            
    def ions(self):
        """ Return list of all ionized species. """     
        if not hasattr(self, '_ionized_species'):
            neutrals = self.neutrals
            self._ionized_species = []
            for ion in self.all_ions:
                if ion in neutrals:
                    continue
                
                self._ionized_species.append(ion)
                
        return self._ionized_species
    
    @property
    def absorbers(self):    
        """ Return list of absorbers (don't include electrons). """
        if not hasattr(self, '_absorbing_species'):
            self._absorbing_species = copy.copy(self.neutrals)
            for parent in self.ions_by_parent:
                self._absorbing_species.extend(self.ions_by_parent[parent][1:-1])

        return self._absorbing_species
        
    @property
    def N_absorbers(self):
        """ Return number of absorbing species. """
        if not hasattr(self, 'self._num_of_absorbers'):
            absorbers = self.absorbers
            self._num_of_absorbers = int(len(absorbers))
            
        return self._num_of_absorbers
        
    @property
    def species_abundances(self):
        """
        Return dictionary containing abundances of parent
        elements of all ions.
        """
        if not hasattr(self, '_species_abundances'):
            self._species_abundances = {}
            for ion in self.ions_by_parent:
                for state in self.ions_by_parent[ion]:
                    self._species_abundances[state] = \
                        self.element_abundances[self.elements.index(ion)]
    
        return self._species_abundances
        
    @property
    def species(self):
        if not hasattr(self, '_species'):
            self._species = []
            for parent in self.ions_by_parent:
                for ion in self.ions_by_parent[parent]:
                    self._species.append(ion)
                
        return self._species
        
    @property
    def types(self):
        """
        Return list (matching evolving_fields) with integers describing
        species type:
            0 = neutral
           +1 = ion
           -1 = other
        """
        
        if not hasattr(self, '_species_types'):
            self._species_types = []
            for species in self.evolving_fields:
                if species in self.neutrals:
                    self._species_types.append(0)
                elif species in self.ions:
                    self._species_types.append(1)
                else:
                    self._species_types.append(-1) 
                    
            self._species_types = np.array(self._species_types)        
        
        return self._species_types       
        
    @property
    def ioniz_thresholds(self):
        """
        Return dictionary containing ionization threshold energies (in eV)
        for all absorbers.
        """    
        
        if not hasattr(self, '_ioniz_thresholds'):
            self._ioniz_thresholds = {}
            #for absorber in self.absorbers:
            #if absorber == 'h_1':
            self._ioniz_thresholds['h_1'] = 13.6
            #elif absorber == 'he_1':
            self._ioniz_thresholds['he_1'] = 24.4
            #elif absorber == 'he_2':
            self._ioniz_thresholds['he_2'] = 54.4
           
        return self._ioniz_thresholds
        
    @property
    def bf_cross_sections(self):
        """
        Return dictionary containing functions that compute the bound-free 
        absorption cross-sections for all absorbers.
        """    
        
        if not hasattr(self, 'all_xsections'):
            self._bf_xsections = {}
            #for absorber in self.absorbers:
                #ion = cc.continuum(absorber)
                #ion.vernerCross(energy = np.logspace(1, 5, 1000))
                #if absorber == 'h_1':
            self._bf_xsections['h_1'] = lambda E: \
                PhotoIonizationCrossSection(E, species=0)
                #elif absorber == 'he_1':
            self._bf_xsections['he_1'] = lambda E: \
                PhotoIonizationCrossSection(E, species=1)
                #elif absorber == 'he_2':
            self._bf_xsections['he_2'] = lambda E: \
                PhotoIonizationCrossSection(E, species=2) 
                        
        return self._bf_xsections
        
    @property
    def x_to_n(self):
        """
        Return dictionary containing conversion factor between species
        fraction and number density for all species.
        """
        if not hasattr(self, '_x_to_n_converter'):
            self._x_to_n_converter = {}
            for ion in self.all_ions:
                self._x_to_n_converter[ion] = self.n_ref \
                    * self.species_abundances[ion]  
        
        return self._x_to_n_converter
        
    @property
    def expansion(self):
        if not hasattr(self, '_expansion'):
            self.set_physics()
        return self._expansion
    
    @property
    def isothermal(self):
        if not hasattr(self, '_isothermal'):
            self.set_physics()
        return self._isothermal
    
    @property
    def secondary_ionization(self):
        if not hasattr(self, '_secondary_ionization'):
            self.set_physics()
        return self._secondary_ionization
    
    @property
    def compton_scattering(self):
        if not hasattr(self, '_compton_scattering'):
            self.set_physics()
        return self._compton_scattering
        
    @property
    def recombination(self):
        if not hasattr(self, '_recombination'):
            self.set_physics()
        return self._recombination
        
    @property
    def collisional_ionization(self):
        if not hasattr(self, '_collisional_ionization'):
            self.set_physics()
        return self._collisional_ionization   
        
    @property
    def clumping_factor(self):
        if not hasattr(self, '_clumping_factor'):
            self.set_physics()
        return self._clumping_factor
        
    @property
    def hydr(self):
        if not hasattr(self, '_hydr'):
            self._hydr = Hydrogen(self.cosm, **self.pf)
        return self._hydr    
            
    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology()
        return self._cosm            
                
    def set_properties(self, **kwargs):
        """
        Initialize grid properties all in one go.
        """    
        
        self.set_physics(
            isothermal=kwargs['isothermal'], 
            compton_scattering=kwargs['compton_scattering'],
            secondary_ionization=kwargs['secondary_ionization'], 
            expansion=kwargs['expansion'],
            recombination=kwargs['recombination'],
            clumping_factor=kwargs['clumping_factor'],
            collisional_ionization=kwargs['collisional_ionization']
        )
        
        self.set_cosmology(
            initial_redshift=kwargs['initial_redshift'], 
            omega_m_0=kwargs["omega_m_0"], 
            omega_l_0=kwargs["omega_l_0"], 
            omega_b_0=kwargs["omega_b_0"], 
            hubble_0=kwargs["hubble_0"], 
            helium_by_number=kwargs['helium_by_number'], 
            cmb_temp_0=kwargs["cmb_temp_0"],
            approx_highz=kwargs["approx_highz"])
        
        self.set_chemistry(kwargs['include_He'])
        self.set_density(kwargs['density_units'])
        self.set_ionization(kwargs['initial_ionization'])
        self.set_temperature(kwargs['initial_temperature'])
                
    def set_physics(self, isothermal=False, compton_scattering=False,
        secondary_ionization=0, expansion=False, recombination='B',
        clumping_factor=1.0, collisional_ionization=True):
        self._isothermal = isothermal
        self._compton_scattering = compton_scattering
        self._secondary_ionization = secondary_ionization
        self._expansion = expansion
        self._recombination = recombination
        self._collisional_ionization = collisional_ionization
        
        if type(clumping_factor) is not types.FunctionType:
            self._clumping_factor = lambda z: clumping_factor
        else:
            self._clumping_factor = clumping_factor
        
        if self._expansion:
            self.set_cosmology()
        
    @property
    def is_cgm_patch(self):    
        if not hasattr(self, '_is_cgm_patch'):
            self.set_recombination_rate()
            
        return self._is_cgm_patch
        
    def set_recombination_rate(self, is_cgm_patch=False):
        self._is_cgm_patch = is_cgm_patch    
        
    def set_cosmology(self, initial_redshift=1e3, omega_m_0=0.272, 
        omega_l_0=0.728, omega_b_0=0.044, hubble_0=0.702, 
        helium_by_number=None, helium_by_mass=0.2454, cmb_temp_0=2.725, 
        approx_highz=False):
        
        self.zi = initial_redshift
        self._cosm = Cosmology(omega_m_0=omega_m_0, 
            omega_l_0=omega_l_0, omega_b_0=omega_b_0,
            hubble_0=hubble_0, 
            helium_by_mass=helium_by_mass,
            cmb_temp_0=cmb_temp_0, 
            approx_highz=approx_highz)        
        
    def set_chemistry(self, include_He=False):
        """
        Initialize chemistry.
        
        This routine sets the chemical composition of the medium being 
        simulated.
        
        Parameters
        ----------
        include_He : bool
            Solve for helium?

        Example
        -------
        grid = Grid(dims=32)
        grid.set_chemistry()   # H-only
        
        """                
        
        self.Z = [1]    
        self.abundances = [1.]
        if include_He:
            self.Z.append(2)
            self.abundances.append(self.cosm.helium_by_number)
                
        self.Z = np.array(self.Z)
        self.ions_by_parent = {} # Ions sorted by parent element in dictionary
        self.parents_by_ion = {} # From ion name, determine parent element
        self.elements = []       # Just a list of element names
        self.all_ions = []       # All ion species          
        self.evolving_fields = []# Anything with an ODE we'll later solve
          
        for i, element in enumerate(self.Z):
            element_name = util.z2element(element)
                
            self.ions_by_parent[element_name] = []
            self.elements.append(element_name)
            for ion in xrange(element + 1):
                name = util.zion2name(element, ion + 1)
                self.all_ions.append(name)
                self.ions_by_parent[element_name].append(name)
                self.parents_by_ion[name] = element_name
                self.evolving_fields.append(name)

        self.solve_ge = False      
        self.evolving_fields.append('e')
        if not self.isothermal:
            self.evolving_fields.append('Tk')

        # Create blank data fields    
        if not hasattr(self, 'data'):            
            self.data = {}
            for field in self.evolving_fields:
                self.data[field] = np.zeros(self.dims)
                                        
        self.abundances_by_number = self.abundances
        self.element_abundances = [1.0]
        if include_He:
            self.element_abundances.append(self.cosm.helium_by_number)
                               
        # Initialize mapping between q-vector and physical quantities (dengo)                
        self._set_qmap()

    def set_density(self, nH=None):
        """
        Initialize hydrogen number density.
        
        Setting the gas density is necessary for computing the hydrogen 
        number density, which normalizes fractional abundances of elements
        to proper number densities of all species.

        Parameters
        ----------
        rho0 : float, array
            Density of medium in g / cm**3. Can be a float (uniform medium),
            or an array of values the same size as the grid itself.
            
        """
        
        if self.cosmological_ics:
            self.n_H = self.cosm.nH(self.zi) 
        elif isinstance(nH, Iterable):    
            self.n_H = nH
        else:
            self.n_H = nH * np.ones(self.dims)    
            
        if 2 in self.Z:
            self.n_He = self.n_H * self.abundances[1]
        else:
            self.n_He = 0.0     
                        
        self.n_ref = self.n_H    
        
        self.data['rho'] = m_H * (self.n_H * mH_amu + self.n_He * mHe_amu)
                            
    def set_temperature(self, T0):
        """
        Set initial temperature in grid.  
        
        Parameters
        ----------
        T0 : float, array
            Initial temperature in grid. Can be constant value (corresponding
            to uniform medium), or an array of values like the grid.
        """
        
        if self.cosmological_ics:
            Tgas = self.cosm.Tgas(self.zi)
            if isinstance(T0, Iterable):
                self.data['Tk'] = np.array(Tgas)
            else:
                self.data['Tk'] = Tgas * np.ones(self.dims)
        elif isinstance(T0, Iterable):
            self.data['Tk'] = np.array(T0)
        else:
            self.data['Tk'] = T0 * np.ones(self.dims)
            
    def set_ionization(self, x=None):
        """
        Set initial ionization state.  
                
        Parameters
        ----------
        x : float, list
            Initial ionization state for all species. Must be a 1:1 mapping
            between values in this list and values in self.species.
          
        """    
                
        if x is not None:

            assert(len(x) == len(self.species))
               
            for j, species in enumerate(self.species):
                element, state = species.split('_')
                Z = util.element2z(element)
                i = int(state)
                         
                name = util.zion2name(Z, i)
                self.data[name].fill(x[j])
                  
        # Otherwise assume neutral
        else:
            for sp in self.ions:
                self.data[sp].fill(1e-8)
            for sp in self.neutrals:
                self.data[sp].fill(1.0 - 1e-8)
        
        # Set electron density
        self._set_electron_fraction()
        
        if self.solve_ge:
            self.set_gas_energy()
        
    def set_ics(self, data):
        """
        Simple way of setting all initial conditions at once with a data 
        dictionary.
        """
        
        self.data = {}
        for key in data.keys():
            if type(data[key]) is float:
                self.data[key] = data[key]
                continue
                
            self.data[key] = data[key].copy()
    
    def create_slab(self, **kwargs):
        """ Create a slab. """
                
        if not kwargs['slab']:
            return        
                
        # Figure out where the clump is
        gridarr = np.linspace(0, 1, self.dims)
        isslab = (gridarr >= (kwargs['slab_position'] - kwargs['slab_radius'])) \
                & (gridarr <= (kwargs['slab_position'] + kwargs['slab_radius']))
                
        # First, modify density and temperature
        if kwargs['slab_profile'] == 0:
            self.data['rho'][isslab] *= kwargs['slab_overdensity']
            self.n_H[isslab] *= kwargs['slab_overdensity']
            self.data['Tk'][isslab] = kwargs['slab_temperature']
        else:
            raise NotImplemented('only know uniform slabs')
                
        # Ionization state - could generalize this more
        for j, species in enumerate(self.species):
            element, state = species.split('_')
            Z = util.element2z(element)
            i = int(state)
                     
            name = util.zion2name(Z, i)
            self.data[name][isslab] = np.ones(isslab.sum()) \
                * kwargs['slab_ionization'][j]
                
        # Reset electron density, particle density, and gas energy
        self._set_electron_fraction()
                
        if hasattr(self, '_x_to_n_converter'):        
            del self._x_to_n_converter
        
    def _set_electron_fraction(self):
        """
        Set electron density - must have run set_density beforehand.
        """
        
        self.data['e'] = np.zeros(self.dims)
        for i, Z in enumerate(self.Z):
            for j in np.arange(1, 1 + Z):   # j = number of electrons donated by ion j + 1
                x_i_jp1 = self.data[util.zion2name(Z, j + 1)]
                self.data['e'] += j * x_i_jp1 * self.n_ref \
                    * self.element_abundances[i]  
                    
        self.data['e'] /= self.n_H              
                
    def particle_density(self, data, z=0):
        """
        Compute total particle number density.
        """    
        
        n = data['e'].copy()
        #for ion in self.all_ions:
        #    n += data[ion] * self.x_to_n[ion] * (1. + z)**3 \
        #        / (1. + self.zi)**3
        
        if self.expansion:
            n *= self.cosm.nH(z)
            n += self.cosm.nH(z)
            
            if 2 in self.Z:
                n += self.cosm.nHe(z)
                
        else:
            n *= self.n_H
            
            n += self.n_H
            
            if 2 in self.Z:
                n += self.n_H * self.cosm.helium_by_number
             
        return n 
            
    def electron_fraction(self, data, z):
        de = np.zeros(self.dims)
        for i, Z in enumerate(self.Z):
            for j in np.arange(1, 1 + Z):   # j = number of electrons donated by ion j + 1
                x_i_jp1 = data[util.zion2name(Z, j + 1)]
                de += j * x_i_jp1 * self.n_ref * (1. + z)**3 / (1. + self.zi)**3 \
                    * self.element_abundances[i]

        return de / self.n_H

    def ColumnDensity(self, data):
        """ Compute column densities for all absorbing species. """    
        
        N = {}
        Nc = {}
        logN = {}
        for absorber in self.absorbers:
            Nc[absorber] = self.dr * data[absorber] * self.x_to_n[absorber]            
            N[absorber] = np.cumsum(Nc[absorber])
            logN[absorber] = np.log10(N[absorber])
            
        return N, logN, Nc

    def _set_qmap(self):
        """
        The vector 'q' is an array containing the values of all ion fractions and the
        gas energy.  This routine sets up the mapping between elements in q and the
        corrresponding physical quantities.
        
        Will be in order of increasing Z, then de, then ge.
        """
        
        self.qmap = []
        for species in self.evolving_fields:
            self.qmap.append(species)
            
    def ShellVolume(self, r, dr):
        """
        Return volume of shell at distance r, thickness dr.
        """
        
        return 4. * np.pi * ((r + dr)**3 - r**3) / 3.            

        

        