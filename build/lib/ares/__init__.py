import os as _os
import imp as _imp

_HOME = _os.environ.get('HOME')

# Load custom defaults    
if _os.path.exists('{!s}/.ares/defaults.py'.format(_HOME)):
    (_f, _filename, _data) =\
        _imp.find_module('defaults', ['{!s}/.ares/'.format(_HOME)])
    rcParams = _imp.load_module('defaults.py', _f, _filename, _data).pf
else:
    rcParams = {}

import ares.physics
import ares.util
import ares.analysis
import ares.sources
import ares.populations
import ares.static
import ares.solvers
import ares.simulations
import ares.inference
