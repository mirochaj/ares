Installation
++++++++++++

Dependencies
------------
If installed via pip, ARES' dependencies will be built automatically.

But, in case you're curious, the core dependencies are:

- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [matplotlib](http://matplotlib.org/)
- [h5py](http://www.h5py.org/)

and the optional dependencies are:

- [camb](https://camb.readthedocs.io/en/latest/)
- [hmf](https://github.com/steven-murray/hmf)
- [astropy](https://www.astropy.org/)
- [dust_extinction](https://dust-extinction.readthedocs.io/en/stable/index.html)
- [dust_attenuation](https://dust-extinction.readthedocs.io/en/stable/index.html)
- [mpi4py](http://mpi4py.scipy.org)
- [pymp](https://github.com/classner/pymp)
- [progressbar2](http://progressbar-2.readthedocs.io/en/latest/)
- [setuptools](https://pypi.python.org/pypi/setuptools)
- [mpmath](http://mpmath.googlecode.com/svn-history/r1229/trunk/doc/build/setup.html)
- [shapely](https://pypi.python.org/pypi/Shapely)
- [descartes](https://pypi.python.org/pypi/descartes)

If you'd like to build the documentation locally, you'll need:

- [numpydoc](https://numpydoc.readthedocs.io/en/latest/)
- [nbsphinx](https://nbsphinx.readthedocs.io/en/0.8.8/)

and if you'd like to run the test suite locally, you'll want:

- [pytest](https://docs.pytest.org/en/7.1.x/)
- [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/)

which are all pip-installable.

Note: ARES has been tested only with Python 2.7.x and Python 3.7.x.

External datasets: stellar population synthesis (SPS) models
------------------------------------------------------------
As discussed in the `README </README.rst>`_, ARES relies on many external datasets. The `ares init` command builds a minimal install, including some cosmological initial conditions, a single metallicity, :math:`Z=0.004`, constant star formation rate, single-star stellar population synthesis model from BPASS version 1.0, and a high redshift lookup table for the Tinker et al. 2010 halo mass function generated with `hmf <https://github.com/steven-murray/hmf>`.

There are many more external datasets that can be downloaded easily using the ARES CLI. For example, to fetch the complete set of BPASS v1 models (all metallicities, constant star formation and simple stellar populations, single star and binaries), you can do

```
ares download bpass_v1
```

There are now newer versions of BPASS, which must be downloaded by hand. To download BPASS v2 models, navigate to `this page <https://bpass.auckland.ac.nz/9.html>`_ and download the desired models in the ``$HOME/.ares/bpass_v2`` directory. If you initialized ARES with a different path (via the `--path` flag; see `README </README.md`_), make sure you instead move files there.

ARES is also setup to handle the starburst99 or Bruzual \& Charlot models, which can be downloaded via

```
ares download starburst99
ares download bc03
ares download bc03_2013   # for the 2013 update to the BC03 models
```

Pre-processing
~~~~~~~~~~~~~~
The spectral resolution of the BPASS models (and some other SPS models) is 1 Angstrom, which is much better than we need for most ARES modeling. So, many examples use "degraded" BPASS models, which just smooth the standard BPASS SEDs with a tophat of some width, generally 10 Angstroms. To do this SED degradation, we use the ARES CLI:

```python
import os
from ares.util import cli as ares_cli

ares_cli.generate_lowres_sps(f"{os.environ.get('HOME')}/.ares/bpass_v2/BPASSv2_imf135_300/OUTPUT_CONT", degrade_to=10)
ares_cli.generate_lowres_sps(f"{os.environ.get('HOME')}/.ares/bpass_v2/BPASSv2_imf135_300/OUTPUT_POP", degrade_to=10)
```

Whether or not these files are required should be discussed in the various examples in the documentation, so please be sure to check there first for instructions if you get errors indicative of missing files.


External datasets: filter transmission curves
---------------------------------------------
For generating mock photometry, we must convolve modeled SEDs with the transmission curves of a given telescope's photometric filters. We've got many common filters available, which you can download via, e.g., ::


    for telescope in roman euclid rubin wise 2mass spherex wise dirbe irac sdss
      ares download $telescope


Again, be sure to supply the ``--path`` flag if your `$HOME` directory has a small quota.

External datasets: posteriors for cosmological parameters from Planck
---------------------------------------------------------------------
Instead of supplying cosmological parameters by hand, you can also draw from published Planck chains directly via the `cosmology_name`, `cosmology_id`, and `cosmology_number` parameters. But first, you need the chains themselves. As per usual, you can do: ::

    ares download planck
