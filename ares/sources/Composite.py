"""

RadiationSources.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Sun Jul 22 16:22:12 2012

Description: Container for several RadiationSource____ instances.  Will loop
over said instances in ionization and heating rate calculations.

"""

import re
from .Toy import Toy
from .Star import Star
from .BlackHole import BlackHole
from .SynthesisModel import SynthesisModel

class Composite(object):
    """ Class for stitching together several radiation sources. """
    def __init__(self, grid=None, **kwargs):
        """
        Initialize composite radiation source object.        
        
        Parameters
        ----------
        grid : ares.static.Grid.Grid instance
        
        """
        
        self.pf = kwargs.copy()
        self.grid = grid
        
        if type(self.pf['source_type']) is not list:
            self.pf['source_type'] = [self.pf['source_type']]    

        self.Ns = len(self.pf['source_type'])
        
        self.all_sources = self.src = self.initialize_sources()
        
    def initialize_sources(self):
        """ Construct list of RadiationSource class instances. """    
        
        sources = []
        for i in range(self.Ns):

            sf = self.pf.copy()
                                                
            # Look for {0}, {1}, etc. here
                                                                        
            # Create RadiationSource class instance
            if sf['source_type'][i] == 'star':
                rs = Star(**sf)
            elif sf['source_type'][i] == 'bh':
                rs = BlackHole(**sf)
            elif sf['source_type'][i] == 'toy':
                rs = Toy(**sf)
            elif sf['source_type'][i] in ['cluster', 'galaxy']:
                # Only difference is galaxies can have SFHs
                rs = SynthesisModel(**sf)
            else:
                msg = 'Unrecognized source_type: {!s}'.format(\
                    sf['source_type'][i])
                raise ValueError(msg)
            
            rs.grid = self.grid
            
            sources.append(rs)
                
        return sources

            
    
        
