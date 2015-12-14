"""

BlobFactory.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Fri Dec 11 14:24:53 PST 2015

Description: 

"""

import numpy as np
from ..util.ParameterFile import ParameterFile

class BlobFactory(object):
    """
    This class must be inherited by another class, which need only have the
    ``pf`` attribute.
    
    The three most (only?) important parameters are:
        blob_names
        blob_ivars
        blob_funcs
        
    """
    
    def _parse_blobs(self):
        names = self.pf['blob_names']
        if names is None:
            self._blob_names = self._blob_ivars = None
            self._blob_dims = self._blob_nd = None
            self._blob_funcs = None
            return None
        else:
            # Otherwise, figure out how many different kinds (shapes) of
            # blobs we have
            assert type(names) in [list, tuple], \
                "Must supply blob_names as list or tuple!"

            self._blob_names = names
            self._blob_ivars = self.pf['blob_ivars']
            
            self._blob_nd = []
            self._blob_dims = []
            self._blob_funcs = []
            for i, element in enumerate(self._blob_names):
                try:
                    self._blob_nd.append(len(np.shape(self._blob_ivars[i])))
                    self._blob_dims.append(np.shape(self._blob_ivars[i]))
                except:
                    # Scalars!
                    self._blob_nd.append(0)
                    self._blob_dims.append(0)
                    
                if self.pf['blob_funcs'] is None:
                    self._blob_funcs.append([None] * len(element))
                elif self._blob_dims[i] == 1 and self.pf['blob_funcs'] is None:
                    self._blob_funcs.append([None] * len(element))
                else:
                    self._blob_funcs.append(self.pf['blob_funcs'][i])
    
    @property
    def blob_groups(self):
        if not hasattr(self, '_blob_groups'):
            self._blob_groups = len(self.blob_nd)
        return self._blob_nd
                
    @property
    def blob_nd(self):    
        if not hasattr(self, '_blob_nd'):
            self._parse_blobs()
        return self._blob_nd
    
    @property
    def blob_dims(self):    
        if not hasattr(self, '_blob_dims'):
            self._parse_blobs()
        return self._blob_dims    
        
    @property
    def blob_names(self):
        if not hasattr(self, '_blob_names'):
            self._parse_blobs()
        return self._blob_names    
            
    @property
    def blob_ivars(self):
        if not hasattr(self, '_blob_ivars'):
            self._parse_blobs()
        return self._blob_ivars
        
    @property
    def blob_funcs(self):
        if not hasattr(self, '_blob_funcs'):
            self._parse_blobs()
        return self._blob_funcs

    @property
    def blobs(self):
        if not hasattr(self, '_blobs'):
            self._generate_blobs()    

        return self._blobs

    def _generate_blobs(self):
        """
        Create a list of blobs, one per blob group.
        
        Returns
        -------
        List, where each element has shape (ivar x blobs). Each element of 
        this corresponds to the blobs for one blob group, which is defined by
        either its dimensionality, its independent variables, or both.
        
        For example, for 1-D blobs, self.blobs[i][j][k] would mean
            i = blob group
            j = index corresponding to elements of self.blob_names
            k = index corresponding to elements of self.blob_ivars[i]
        """
        
        self._blobs = []
        for i, element in enumerate(self.blob_names):
                        
            this_group = []
            for j, key in enumerate(element):
                
                # 0-D blobs. Need to know name of attribute where stored!
                if self.blob_nd[i] == 0:
                    if self.blob_funcs[i][j] is not None:
                        raise NotImplemented('help!')
                    else:
                        # Assume blob name is the attribute
                        blob = self.__getattribute__(key)
                
                # 1-D blobs. Assume the independent variable is redshift.
                elif self.blob_nd[i] == 1:
                    z = self.blob_ivars[i]
                    blob = np.interp(z, self.history['z'][-1::-1], 
                        self.history[key][-1::-1])
                else:
                    pass
                                
                this_group.append(blob)
                
            self._blobs.append(np.array(this_group))
            
    def old(self):    
        if self.pf['track_extrema']:
            if hasattr(self, 'track'):
                self.turning_points = self.track.turning_points
            else:
                from ..analysis.InlineAnalysis import InlineAnalysis
                anl = InlineAnalysis(self)
                self.turning_points = anl.turning_points
    
                self.blobs = anl.blobs
                self.blob_names, self.blob_redshifts = \
                    anl.blob_names, anl.blob_redshifts
    
                return
    
        if (self.pf['inline_analysis'] is None) and \
           (self.pf['auto_generate_blobs'] == False):
            return
    
        elif self.pf['inline_analysis'] is not None:
            self.blob_names, self.blob_redshifts = self.pf['inline_analysis']
    
        # Get da blobs
        from ..analysis.InlineAnalysis import InlineAnalysis
        anl = InlineAnalysis(self)  
        anl.run_inline_analysis()      
        self.blobs = anl.blobs
    
        self.anl = anl
    
        # Just arrayify history elements if they aren't already arrays
        tmp = {}
        for key in self.history:
            if type(self.history[key]) is list:
                tmp[key] = np.array(self.history[key])
            else:
                tmp[key] = self.history[key]
    
        self.history = tmp
        
