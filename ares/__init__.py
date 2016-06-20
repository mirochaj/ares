import os as _os
import imp as _imp

_HOME = _os.environ.get('HOME')

# Load custom defaults    
if _os.path.exists('%s/.ares/defaults.py' % _HOME):
    _f, _filename, _data = _imp.find_module('defaults', ['%s/.ares/' % _HOME])
    rcParams = _imp.load_module('defaults.py', _f, _filename, _data).pf
else:
    rcParams = {}
    
import physics, util, phenom, analysis
import sources, populations
import static, solvers
import simulations
import inference
