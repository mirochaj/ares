.. ares documentation master file, created by
   sphinx-quickstart on Mon Jul  8 08:48:22 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========
**ARES**
========

.. image:: https://readthedocs.org/projects/ares/badge/?version=latest
	:target: https://ares.readthedocs.io/en/latest/?badge=latest
.. image:: https://github.com/mirochaj/ares/actions/workflows/test_suite.yaml/badge.svg
  :target: https://github.com/mirochaj/ares/actions/workflows/test_suite.yaml/badge.svg
.. image:: https://codecov.io/gh/mirochaj/ares
	:target: https://codecov.io/gh/mirochaj/ares
.. image:: https://img.shields.io/github/last-commit/mirochaj/ares
	:target: https://img.shields.io/github/last-commit/mirochaj/ares

The Accelerated Reionization Era Simulations (*ARES*) code was designed to rapidly generate models for the global 21-cm signal. It can also be used as a 1-D radiative transfer code, stand-alone non-equilibrium chemistry solver, or meta-galactic radiation background calculator. As of late 2016, it also contains a home-grown semi-analytic model of galaxy formation.

A few papers on how it works:

- 1-D radiative transfer: `Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.
- Uniform backgrounds \& global 21-cm signal: `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_.
- Galaxy luminosity functions: `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_.
- Population III star formation: `Mirocha et al. (2018) <http://adsabs.harvard.edu/abs/2018MNRAS.478.5591M>`_
- Rest-ultraviolet colours at high-:math:`z`: `Mirocha, Mason, & Stark (2020) <https://ui.adsabs.harvard.edu/abs/2020arXiv200507208M/abstract>`_

Plus some more applications:

- First stars and early galaxies: `Mirocha & Furlanetto (2019) <http://adsabs.harvard.edu/abs/2018arXiv180303272M>`_, `Mebane, Mirocha, \& Furlanetto (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv191010171M/abstract>`_.
- Warm dark matter: `Schneider (2018) <http://adsabs.harvard.edu/abs/2018PhRvD..98f3021S>`_, `Leo et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190904641L/abstract>`_, `Rudakovskyi et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv190906303R/abstract>`_.
- Parameter inference \& forecasting: `Mirocha, Harker, & Burns (2015) <http://adsabs.harvard.edu/abs/2015ApJ...813...11M>`_, `Tauscher et al. (2017) <http://adsabs.harvard.edu/abs/2018ApJ...853..187T>`_, `Sims \& Pober (2019) <https://ui.adsabs.harvard.edu/abs/2019arXiv191003165S/abstract>`_.

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed. This documentation is as much of a work in progress as the code itself, so if you encounter gaps or errors please do let me know.

Quick-Start
-----------
To make sure everything is working, a quick test is to generate a
realization of the global 21-cm signal using all default parameter values:

::

    import ares

    sim = ares.simulations.Global21cm()
    sim.run()
    sim.GlobalSignature()

See :doc:`example_gs_standard` in :doc:`examples` for a more thorough
introduction to this type of calculation.

Contents
--------
.. toctree::
    :maxdepth: 1

    Home <self>
    install
    examples
    performance
    uth
    troubleshooting
    updates
    contributing
    history
    acknowledgements


.. Indices and tables
.. ==================
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
