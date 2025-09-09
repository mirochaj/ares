import os as _os
import importlib as _imp
# updates to imp: https://docs.python.org/3/whatsnew/3.12.html#whatsnew312-removed-imp

_HOME = _os.environ.get('HOME')

# Load custom defaults    
if _os.path.exists('~/.ares/defaults.py'.format(_HOME)):
    (_f, _filename, _data) =\
        _imp.util.find_spec('defaults', ['~/.ares/'.format(_HOME)])
    rcParams = _imp.import_module('defaults.py', _f, _filename, _data).pf
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
