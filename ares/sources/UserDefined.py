from .Source import Source
from inspect import ismethod
from types import FunctionType
from ..util.Math import interp1d
from ..util.SetDefaultParameterValues import SourceParameters
try:
    # this runs with no issues in python 2 but raises error in python 3
    basestring
except:
    # this try/except allows for python 2/3 compatible string type checking
    basestring = str

class UserDefined(Source):
    def __init__(self, **kwargs):
        """ 
    
        Parameters
        ----------
        pf: dict
            Full parameter file.
    
        """  
        
        self.pf = SourceParameters()
        self.pf.update(kwargs)    
        Source.__init__(self)
                
        self._name = 'user_defined'
        
        self._load()
        
    def _load(self):
        sed = self.pf['source_sed']
        E = self.pf['source_E']
        L = self.pf['source_L']
        
        if sed is not None:
            
            if sed == 'user':
                pass
            elif type(sed) is FunctionType or ismethod(sed) or \
                 isinstance(sed, interp1d):
                self._func = sed
                return
            elif type(sed) is tuple:
                E, L = sed
        
        elif isinstance(sed, basestring):
            E, L = np.loadtxt(sed, unpack=True)
        elif (E is not None) and (L is not None):
            assert len(E) == len(L)
        else:
            raise NotImplemented('sorry, dont understand!')
        
        self._func = interp1d(E, L, kind='cubic', bounds_error=False)
    
    def _Intensity(self, E, t=0):
        return self._func(E)
        
        
        
