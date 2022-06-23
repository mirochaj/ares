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

Note that for some applications, ARES relies heavily on lookup tables and publicly-available software packages that should be referenced as well. These include:

- Code for Anisotropies in the Microwave Background ([CAMB](https://camb.readthedocs.io/en/latest/)).
- The Halo Mass Function ([hmf](https://hmf.readthedocs.io/en/latest/)) package (see [Murray et al.(2013)](https://arxiv.org/abs/1306.6721)).
- Lookup tables and fitting formulae for the fraction of photo-electron energy deposited in heat, ionization, excitation from [Shull \& van Steenberg (1985)](https://ui.adsabs.harvard.edu/abs/1985ApJ...298..268S/abstract), [Ricotti, Gnedin, \& Shull (2002)](https://ui.adsabs.harvard.edu/abs/2002ApJ...575...33R/abstract), and [Furlanetto \& Stoever (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1869F/abstract) (see `secondary_ionization` parameter, values of 2, 3, and 4, respectively).
- Collisional coupling coefficients for the 21-cm line from [Zygelman (2005)](https://ui.adsabs.harvard.edu/abs/2005ApJ...622.1356Z/abstract).
- Wouthuysen-Field coupling coefficients for the 21-cm line from [Chuzhoy, Alvarez, & Shapiro (2006)](https://ui.adsabs.harvard.edu/abs/2006ApJ...651....1C/abstract), [Furlanetto \& Pritchard (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372.1093F/abstract), [Hirata (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..259H/abstract), and [Mittal & Kulkarni (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract) (see `approx_Salpha` parameter, values of 2, 3, 4, and 5, respectively).
- Lyman-alpha transition probabilities from [Pritchard \& Furlanetto (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367.1057P/abstract).
- Stellar population synthesis model options include starburst99 ([Leitherer et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999ApJS..123....3L/abstract)) and BPASS versions 1 ([Eldridge \& Stanway (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.400.1019E/abstract)) and 2 ([Eldridge et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017PASA...34...58E/abstract),[Stanway \& Eldridge (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479...75S/abstract)) (via `pop_sed` parameter, values `'starburst99'`, `'bpass_v1'`, and `'bpass_v2'`, respectively).

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
- [mpi4py](http://mpi4py.scipy.org)
- [pymp](https://github.com/classner/pymp)
- [emcee](http://dan.iel.fm/emcee/current/)
- [distpy](https://bitbucket.org/ktausch/distpy)
- [progressbar2](http://progressbar-2.readthedocs.io/en/latest/)
- [setuptools](https://pypi.python.org/pypi/setuptools)
- [mpmath](http://mpmath.googlecode.com/svn-history/r1229/trunk/doc/build/setup.html)
- [shapely](https://pypi.python.org/pypi/Shapely)
- [descartes](https://pypi.python.org/pypi/descartes)

If you'd like to build the documentation locally, you'll need:

- [numpydoc](https://numpydoc.readthedocs.io/en/latest/)
- [nbsphinx](https://nbsphinx.readthedocs.io/en/0.8.8/)

Note: **ares** has been tested only with Python 2.7.x and Python 3.7.x.

## Getting started

To clone a copy and install:

```
git clone https://github.org/mirochaj/ares.git
cd ares
python setup.py install
```

**ares** will look in ``ares/input`` for lookup tables of various kinds. To download said lookup tables, run:

```
python remote.py
```

This might take a few minutes. If something goes wrong with the download, you can run

```
python remote.py fresh
```

to get fresh copies of everything.

## Quick Example

To generate a model for the global 21-cm signal, simply type:

```python
import ares

sim = ares.simulations.Global21cm()      # Initialize a simulation object
sim.run()   
```                                               

You can examine the contents of ``sim.history``, a dictionary which contains
the redshift evolution of all IGM physical quantities, or use some built-in
analysis routines:

```python
sim.GlobalSignature()
```    

If the plot doesn't appear automatically, set ``interactive: True`` in your matplotlibrc file or type:

```python
import matplotlib.pyplot as pl
pl.show()
```

## Help

If you encounter problems with installation or running simple scripts, first check the Troubleshooting page in the documentation to see if you're dealing with a common problem. If you don't find your problem listed there, please let me know!

## Contributors

Primary author: [Jordan Mirocha](https://sites.google.com/site/jordanmirocha/home) (McGill)

Additional contributions / corrections / suggestions from:

- Geraint Harker
- Jason Sun
- Keith Tauscher
- Jacob Jost
- Greg Salvesen
- Adrian Liu
- Saurabh Singh
- Rick Mebane
- Krishma Singal
- Donald Trinh
- Omar Ruiz Macias
- Arnab Chakraborty
- Madhurima Choudhury
- Saul Kohn
- Aurel Schneider
- Kristy Fu
- Garett Lopez
- Ranita Jana
- Daniel Meinert
- Henri Lamarre
- Matteo Leo
- Emma Klemets
- Felix Bilodeau-Chagnon
- Venno Vipp
- Oscar Hernandez
- Joshua Hibbard
- Trey Driskell
