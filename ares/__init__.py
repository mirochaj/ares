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

from . import ares.physics
from . import ares.util
from . import ares.analysis
from . import ares.sources
from . import ares.populations
from . import ares.core
from . import ares.solvers
from . import ares.simulations
from . import ares.inference
from . import ares.realizations
