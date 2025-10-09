
# **ARES**
[![Documentation Status](https://readthedocs.org/projects/ares/badge/?version=latest)](http://ares.readthedocs.io/en/latest/?badge=latest) ![Tests](https://github.com/mirochaj/ares/actions/workflows/test_suite.yaml/badge.svg) [![codecov](https://codecov.io/gh/mirochaj/ares/branch/main/graph/badge.svg?token=Q3CCKIMQJF)](https://codecov.io/gh/mirochaj/ares) [![Last Commit](https://img.shields.io/github/last-commit/mirochaj/ares)](https://img.shields.io/github/last-commit/mirochaj/ares)

The Accelerated Reionization Era Simulations (ARES) code was originally designed to
rapidly generate models for the global 21-cm signal. However, it can also be used as a
1-D radiative transfer code, stand-alone non-equilibrium chemistry solver,
global radiation background calculator, or semi-empirical galaxy formation model.

The documentation is [here](https://ares.readthedocs.io/en/latest/index.html).

## Technical Details

The main papers that describe how ARES works include:

- Galaxy luminosity functions (LFs): [Mirocha, Furlanetto, & Sun (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.464.1365M/abstract)
- Self-consistent LFs and rest-ultraviolet colours: [Mirocha, Mason, & Stark (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract)
- Near-infrared background and nebular emission: [Sun et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.1954S/abstract)
- High-z UV and X-ray background \& global 21-cm signal: [Mirocha (2014)](http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M)
- Population III star formation: [Mirocha et al. (2018)](http://adsabs.harvard.edu/abs/2018MNRAS.478.5591M)
- 1-D radiative transfer: [Mirocha et al. (2012)](http://adsabs.harvard.edu/abs/2012ApJ...756...94M)

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed.

## Citation

If you use ARES, please be sure to cite the relevant papers above depending on your use case (see [here](CITATION.rst) for bibtex entries). Please also provide a link to [this page](https://github.com/mirochaj/ares) as a footnote.

**Note that for some applications, ARES relies heavily on lookup tables and publicly-available software packages that should be referenced as well.** These include:

- Code for Anisotropies in the Microwave Background ([CAMB](https://camb.readthedocs.io/en/latest/)).
- The Halo Mass Function ([hmf](https://hmf.readthedocs.io/en/latest/)) package (see [Murray et al.(2013)](https://arxiv.org/abs/1306.6721)).
- Lookup tables and fitting formulae for the fraction of photo-electron energy deposited in heat, ionization, excitation from [Shull \& van Steenberg (1985)](https://ui.adsabs.harvard.edu/abs/1985ApJ...298..268S/abstract), [Ricotti, Gnedin, \& Shull (2002)](https://ui.adsabs.harvard.edu/abs/2002ApJ...575...33R/abstract), and [Furlanetto \& Stoever (2010)](https://ui.adsabs.harvard.edu/abs/2010MNRAS.404.1869F/abstract) (see `secondary_ionization` parameter, values of 2, 3, and 4, respectively).
- Collisional coupling coefficients for the 21-cm line from [Zygelman (2005)](https://ui.adsabs.harvard.edu/abs/2005ApJ...622.1356Z/abstract).
- Wouthuysen-Field coupling coefficients for the 21-cm line from [Chuzhoy & Shapiro (2006)](https://ui.adsabs.harvard.edu/abs/2006ApJ...651....1C/abstract), [Furlanetto \& Pritchard (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372.1093F/abstract), [Hirata (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..259H/abstract), and [Mittal & Kulkarni (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.503.4264M/abstract) (see `approx_Salpha` parameter, values of 2, 3, 4, and 5, respectively).
- Lyman-alpha transition probabilities from [Pritchard \& Furlanetto (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.367.1057P/abstract).
- Stellar population synthesis model options include starburst99 ([Leitherer et al. (1999)](https://ui.adsabs.harvard.edu/abs/1999ApJS..123....3L/abstract)), BPASS versions 1 ([Eldridge \& Stanway (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.400.1019E/abstract)) and 2 ([Eldridge et al. (2017)](https://ui.adsabs.harvard.edu/abs/2017PASA...34...58E/abstract),[Stanway \& Eldridge (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479...75S/abstract)), and the [Bruzual \& Charlot (2003)](https://www.bruzual.org/bc03/) models. Which model is used is controlled by the `pop_sed` parameter, values `'starburst99'`, `'bpass_v1'`, `'bpass_v2'`, and `'bc03'` respectively. Note that each of these models is essentially a big lookup table; some can be downloaded automatically (see below).

Feel free to get in touch if you are unsure of whether any of these tools are being used under the hood for your application.


## Installation

To install ARES, we recommend using pip:

```
pip install ares-astro # actual name TBD
```

ARES often relies on external datasets.

To download the core set of lookup tables needed for the most common use-cases, and perform some pre-processing, we can use the ARES command-line interface (CLI):

```
ares init
```

By default, ARES will download files to ``$HOME/.ares``. However, if your ``$HOME`` quota is small, do provide the flag ``--path=<someplace-with-plenty-of-disk-space>`` to ``ares init``, and setup a symbolic link that points from ``$HOME/.ares`` to this new location.

Note that ``ares init`` sets up a minimal ARES installation with only the most oft-used external datasets. For more information about what is needed for broader applications, see [this page](INSTALL.rst).

## Quick Examples

To generate a math:`z=6` luminosity function, you can do

```python
import ares
import numpy as np
import matplotlib.pyplot as plt

pars = ares.util.ParameterBundle('mirocha2020:legacy')
pop = ares.populations.GalaxyPopulation(**pars)

bins, phi = pop.get_uvlf(z=6, bins=np.arange(-25, -10, 0.1))

plt.semilogy(bins, phi)
```

Note: if the plot doesn't appear automatically, set ``interactive: True`` in your matplotlibrc file or type:

```python
plt.show()
```

To generate a model for the global 21-cm signal, simply type:

```python
pars = ares.util.ParameterBundle('global_signal:basic')
sim = ares.simulations.Simulation(**pars)               
gs = sim.get_21cm_gs()   
```                                               

You can examine the contents of ``gs.history``, a dictionary which contains
the redshift evolution of all properties of the intergalactic medium, or use some built-in analysis routines:

```python
gs.Plot21cmGlobalSignal()
```    

If you're a pre-version-1.0 ARES user, most of this will look familiar, except these days we're running all models (21-cm, near-infrared background, etc.) through the `ares.simulations.Simulation` interface rather than specific classes. There's also a lot more consistency in call sequences, e.g., we adopt the convention of naming commonly-used functions and attributes as `get_<something>` and `tab_<something>`. A much longer list of v1 convention changes can be found in [Pull Request 61](https://github.com/mirochaj/ares/pull/61).
