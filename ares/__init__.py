"""Init file for ARES package."""

import os
import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from setuptools_scm import get_version

# get version information
try:
    # get accurate version for developer installs
    version_str = get_version(Path(__file__).parent.parent)

    __version__ = version_str
except(LookupError, ImportError):
    try:
        # Set the version automatically from the package details
        __version__ = version("ares")
    except PackageNotFoundError:
        # package is not installed
        pass

_HOME = os.environ.get('HOME')

# Load custom defaults
if os.path.exists(f'{_HOME}/.ares/defaults'):
    mod = importlib.import_module(f'{_HOME}/.ares/defaults')
    rcParams = mod.pf
else:
    rcParams = {}

from . import physics
from . import util
from . import analysis
from . import sources
from . import populations
from . import core
from . import solvers
from . import simulations
from . import inference
from . import realizations
