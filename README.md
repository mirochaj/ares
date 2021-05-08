[![Documentation Status](https://readthedocs.org/projects/ares/badge/?version=latest)](http://ares.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.com/mirochaj/ares.svg?branch=master)](https://travis-ci.com/mirochaj/ares) [![Coverage Status](https://coveralls.io/repos/github/mirochaj/ares/badge.svg?branch=master)](https://coveralls.io/github/mirochaj/ares?branch=master) [![Last Commit](https://img.shields.io/github/last-commit/mirochaj/ares)](https://img.shields.io/github/last-commit/mirochaj/ares)


# **ARES**
The Accelerated Reionization Era Simulations (ARES) code was designed to
rapidly generate models for the global 21-cm signal. It can also be used as a
1-D radiative transfer code, stand-alone non-equilibrium chemistry solver, or
global radiation background calculator.

A few papers on how it works:

- 1-D radiative transfer: [Mirocha et al. (2012)](http://adsabs.harvard.edu/abs/2012ApJ...756...94M)
- Uniform backgrounds \& global 21-cm signal: [Mirocha (2014)](http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M)
- Galaxy luminosity functions: [Mirocha, Furlanetto, & Sun (2017)](http://adsabs.harvard.edu/abs/2016arXiv160700386M)
- Population III star formation: [Mirocha et al. (2018)](http://adsabs.harvard.edu/abs/2018MNRAS.478.5591M)
- Rest-ultraviolet colours at high-z: [Mirocha, Mason, & Stark (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract)

Plus some more applications:

- [Mirocha & Furlanetto (2019)](http://adsabs.harvard.edu/abs/2018arXiv180303272M)
- [Schneider (2018)](http://adsabs.harvard.edu/abs/2018PhRvD..98f3021S)
- [Tauscher et al. (2017)](http://adsabs.harvard.edu/abs/2018ApJ...853..187T)
- [Mirocha, Harker, & Burns (2015)](http://adsabs.harvard.edu/abs/2015ApJ...813...11M)

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed.

If you'd like to live on the bleeding edge, check out the ares-dev branch! Once you clone **ares** you can switch via: ::

    git checkout ares-dev

The [documentation](http://ares.readthedocs.org/en/latest/) is still a work in progress.

## Citation

If you use ARES in paper please reference [Mirocha (2014)](http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M) if it's an application of the global 21-cm modeling machinery and [Mirocha et al. (2012)](http://adsabs.harvard.edu/abs/2012ApJ...756...94M) if you use the 1-D radiative transfer and/or SED optimization. Either way, please provide a link to [this page](https://github.com/mirochaj/ares) as a footnote.

## Getting started

To clone a copy and install:

```
git clone https://github.org/mirochaj/ares.git
cd ares
python setup.py install
```

You'll need to set an environment variable which points to the *ares* install directory, e.g. (in bash):

```
export ARES=/users/<yourusername>/ares
```

**ares** will look in ``$ARES/input`` for lookup tables of various kinds. To download said lookup tables, run:

```
python remote.py
```

This might take a few minutes. If something goes wrong with the download, you can run

```
python remote.py fresh
```

to get fresh copies of everything.

## Dependencies

You will need:

- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/)
- [matplotlib](http://matplotlib.org/)
- [h5py](http://www.h5py.org/)

and optionally,

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


Note: **ares** has been tested only with Python 2.7.x and Python 3.7.x.

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

See the documentation for more examples.

## Documentation

To generate the documentation locally,

```
cd $ARES/doc
make html
open _build/html/index.html
```

This will open the documentation in a browser. For the above to work, you'll
need [sphinx](http://sphinx-doc.org/contents.html), which can be installed
via pip:

```
pip install sphinx
```

This depends on [numpydoc](https://github.com/numpy/numpydoc), which can also
be installed via pip:

```
pip install numpydoc
```

You can also just view the latest build [here](http://ares.readthedocs.org/en/latest/).

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
