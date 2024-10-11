
# **ARES**
[![Documentation Status](https://readthedocs.org/projects/ares/badge/?version=latest)](http://ares.readthedocs.io/en/latest/?badge=latest) ![Tests](https://github.com/mirochaj/ares/actions/workflows/test_suite.yaml/badge.svg) [![codecov](https://codecov.io/gh/mirochaj/ares/branch/main/graph/badge.svg?token=Q3CCKIMQJF)](https://codecov.io/gh/mirochaj/ares) [![Last Commit](https://img.shields.io/github/last-commit/mirochaj/ares)](https://img.shields.io/github/last-commit/mirochaj/ares)

The Accelerated Reionization Era Simulations (ARES) code was designed to
rapidly generate models for the global 21-cm signal. It can also be used as a
1-D radiative transfer code, stand-alone non-equilibrium chemistry solver,
global radiation background calculator, or semi-analytic galaxy formation model.

The documentation is [here](https://ares.readthedocs.io/en/latest/index.html).

## Technical Details

The main papers that describe how ARES works include:

- 1-D radiative transfer: [Mirocha et al. (2012)](http://adsabs.harvard.edu/abs/2012ApJ...756...94M)
- Uniform backgrounds \& global 21-cm signal: [Mirocha (2014)](http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M)
- Galaxy luminosity functions: [Mirocha, Furlanetto, & Sun (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.1365M/abstract)
- Population III star formation: [Mirocha et al. (2018)](http://adsabs.harvard.edu/abs/2018MNRAS.478.5591M)
- Rest-ultraviolet colours at high-z: [Mirocha, Mason, & Stark (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract)
- Near-infrared background and nebular emission: [Sun et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1954S/abstract)

Plus some more applications:

- [Mirocha & Furlanetto (2019)](http://adsabs.harvard.edu/abs/2018arXiv180303272M)
- [Schneider (2018)](http://adsabs.harvard.edu/abs/2018PhRvD..98f3021S)
- [Tauscher et al. (2017)](http://adsabs.harvard.edu/abs/2018ApJ...853..187T)
- [Mirocha, Harker, & Burns (2015)](http://adsabs.harvard.edu/abs/2015ApJ...813...11M)

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed.

## Citation

If you use ARES in paper please reference [Mirocha (2014)](http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M) if it's an application of the global 21-cm modeling machinery and [Mirocha et al. (2012)](http://adsabs.harvard.edu/abs/2012ApJ...756...94M) if you use the 1-D radiative transfer and/or SED optimization. For galaxy semi-analytic modeling, please have a look at [Mirocha, Furlanetto, & Sun (2017)](http://adsabs.harvard.edu/abs/2016arXiv160700386M), [Mirocha, Mason, & Stark (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract), and [Mirocha (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.4534M/abstract), and for PopIII star modeling, see [Mirocha et al. (2018)](http://adsabs.harvard.edu/abs/2018MNRAS.478.5591M).

Please also provide a link to [this page](https://github.com/mirochaj/ares) as a footnote.

**Note that for some applications, ARES relies heavily on lookup tables and publicly-available software packages that should be referenced as well.** These include:

- Code for Anisotropies in the Microwave Background ([CAMB](https://camb.readthedocs.io/en/latest/)).
- The Halo Mass Function ([hmf](https://hmf.readthedocs.io/en/latest/)) package (see [Murray et al.(2013)](https://arxiv.org/abs/1306.6721)).
- Lookup tables and fitting formulae for the fraction of photo-electron energy deposited in heat, ionization, excitation from [Shull \& van Steenberg (1985)](https://ui.adsabs.harvard.edu/abs/1985ApJ...298..268S/abstract), [Ricotti, Gnedin, \& Shull (2002)](https://ui.adsabs.harvard.edu/abs/2002ApJ...575...33R/abstract), and [Furlanetto \& Stoever (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1869F/abstract) (see `secondary_ionization` parameter, values of 2, 3, and 4, respectively).
- Collisional coupling coefficients for the 21-cm line from [Zygelman (2005)](https://ui.adsabs.harvard.edu/abs/2005ApJ...622.1356Z/abstract).
- Wouthuysen-Field coupling coefficients for the 21-cm line from [Chuzhoy & Shapiro (2006)](https://ui.adsabs.harvard.edu/abs/2006ApJ...651....1C/abstract), [Furlanetto \& Pritchard (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372.1093F/abstract), [Hirata (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..259H/abstract), and [Mittal & Kulkarni (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract) (see `approx_Salpha` parameter, values of 2, 3, 4, and 5, respectively).
- Lyman-alpha transition probabilities from [Pritchard \& Furlanetto (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367.1057P/abstract).
- Stellar population synthesis model options include starburst99 ([Leitherer et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999ApJS..123....3L/abstract)), BPASS versions 1 ([Eldridge \& Stanway (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.400.1019E/abstract)) and 2 ([Eldridge et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017PASA...34...58E/abstract),[Stanway \& Eldridge (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479...75S/abstract)), and the [Bruzual \& Charlot (2003)](https://www.bruzual.org/bc03/) models. Which model is used is controlled by the `pop_sed` parameter, values `'starburst99'`, `'bpass_v1'`, `'bpass_v2'`, and `'bc03'` respectively. Note that each of these models is essentially a big lookup table; some can be downloaded automatically (see below).

Feel free to get in touch if you are unsure of whether any of these tools are being used under the hood for your application.

## Dependencies

You will need:

- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [matplotlib](http://matplotlib.org/)
- [h5py](http://www.h5py.org/)

and optionally,

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

which are pip-installable.

Note: ARES has been tested only with Python 2.7.x and Python 3.7.x.

## Installation

To clone a copy and install:

```
git clone https://github.org/mirochaj/ares.git
cd ares
pip install . # or pip install -e .
```

ARES will look in ``$HOME/.ares`` for lookup tables of various kinds. To download the core set of lookup tables needed for the most common use-cases, we can use the ARES command-line interface (CLI):

```
ares download all
```

This might take a few minutes. If something goes wrong with the download, e.g., you lose your internet connection, you can run

```
ares download all --fresh
```

to get fresh copies of everything. You can also download or re-download one dataset at a time, e.g.,

```
ares download bc03
```

The examples within the documentation should say whether they require any non-standard lookup tables that, e.g., cannot be downloaded automatically using `ares download`. Please keep an eye out for that -- if you don't see any special instructions, and you're getting `IOError` or `OSError` or the like, do reach out.

Last note on this front. If you are running ARES on a machine with a very small quota in the `$HOME` directory, our trick of hiding lookup tables in `$HOME/.ares` will cause problems. A quick solution to this is to move the contents of `$HOME/.ares` somewhere else with plenty of disk space, and then make the file `$HOME/.ares` a symbolic link that points to this new folder. Probably we should add a flag to the CLI that can re-direct downloads to a user-supplied location to automate this hack in the future.

## Pre-processing

Not only do some ARES calculations rely on external datasets, they often benefit from using slightly-modified versions of those datasets. For example, the spectral resolution of the BPASS models is 1 Angstrom, which is much better than we need for most ARES modeling. So, many examples use "degraded" BPASS models, which just smooth the standard BPASS SEDs with a tophat of some width, generally 10 Angstroms. To do this SED degradation, we also use the ARES CLI:

```python
import os
from ares.util import cli as ares_cli

ares_cli.generate_lowres_sps(f"{os.environ.get('HOME')}/.ares/bpass_v2/BPASSv2_imf135_300/OUTPUT_CONT", degrade_to=10)
ares_cli.generate_lowres_sps(f"{os.environ.get('HOME')}/.ares/bpass_v2/BPASSv2_imf135_300/OUTPUT_POP", degrade_to=10)
```

Once again, this kind of information should be included in our examples, so please check there for instructions if you get errors indicative of missing files.

## Quick Examples

To generate a model for the global 21-cm signal, simply type:

```python
import ares

pars = ares.util.ParameterBundle('global_signal:basic') # Parameters
sim = ares.simulations.Simulation(**pars)               # Initialize a simulation object
gs = sim.get_21cm_gs()   
```                                               

You can examine the contents of ``gs.history``, a dictionary which contains
the redshift evolution of all IGM physical quantities, or use some built-in
analysis routines:

```python
gs.Plot21cmGlobalSignal()
```    

If the plot doesn't appear automatically, set ``interactive: True`` in your matplotlibrc file or type:

```python
import matplotlib.pyplot as plt
plt.show()
```

To generate a quick luminosity function, you could do

```python
pars = ares.util.ParameterBundle('mirocha2017:base').pars_by_pop(0, 1)
pop = ares.populations.GalaxyPopulation(**pars)

bins, phi = pop.get_uvlf(z=6, bins=np.arange(-25, -10, 0.1))

plt.semilogy(bins, phi)
```

If you're a pre-version-1.0 ARES user, most of this will look familiar, except these days we're running all models (21-cm, near-infrared background, etc.) through the `ares.simulations.Simulation` interface rather than specific classes. There's also a lot more consistency in call sequences, e.g., we adopt the convention of naming commonly-used functions and attributes as `get_<something>` and `tab_<something>`. A much longer list of v1 convention changes can be found in [Pull Request 61](https://github.com/mirochaj/ares/pull/61).

## Contributors

Primary author: [Jordan Mirocha](https://sites.google.com/site/jordanmirocha/home)

Additional contributions / corrections / suggestions from:

.. hlist::
   :columns: 3

   * Geraint Harker               
   * Jason Sun
   * Keith Tauscher
   * Jacob Jost
   * Greg Salvesen
   * Adrian Liu
   * Saurabh Singh
   * Rick Mebane
   * Krishma Singal
   * Donald Trinh
   * Omar Ruiz Macias
   * Arnab Chakraborty
   * Madhurima Choudhury
   * Saul Kohn
   * Aurel Schneider
   * Kristy Fu
   * Garett Lopez
   * Ranita Jana
   * Daniel Meinert
   * Henri Lamarre
   * Matteo Leo
   * Emma Klemets
   * Felix Bilodeau-Chagnon
   * Venno Vipp
   * Oscar Hernandez
   * Joshua Hibbard
   * Trey Driskell
   * Judah Luberto
   * Paul La Plante
