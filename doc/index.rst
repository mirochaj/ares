.. ares documentation master file, created by
   sphinx-quickstart on Mon Jul  8 08:48:22 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*ares*
==============================
The Accelerated Reionization Era Simulations (*ares*) code was designed to
rapidly generate models for the global 21-cm signal. It can also be used as a 
1-D radiative transfer code, stand-alone non-equilibrium chemistry solver, or
meta-galactic radiation background calculator.

A few papers on how it works:

- 1-D radiative transfer: `Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.
- Uniform backgrounds \& global 21-cm signal: `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_.
- Parameter inference: `Mirocha, Harker, & Burns (2015) <http://adsabs.harvard.edu/abs/2015ApJ...813...11M>`_.
- Galaxy luminosity functions: `Mirocha, Furlanetto, & Sun (2016) <http://adsabs.harvard.edu/abs/2016arXiv160700386M>`_.

Be warned: this code is still under active development -- use at your own
risk! Correctness of results is not guaranteed. This documentation is as much of a work in progress as the code itself, so if you encounter gaps or errors please do let me know.

Current status:

.. image:: https://drone.io/bitbucket.org/mirochaj/ares-dev/status.png
   :target: https://drone.io/bitbucket.org/mirochaj/ares-dev/latest

.. image:: https://readthedocs.org/projects/ares/badge/?version=latest
   :target: http://ares.readthedocs.io/en/latest/?badge=latest


Quick-Start
-----------
To make sure everything is working, a quick test is to generate a
realization of the global 21-cm signal using all default parameter values: 

::

    import ares

    sim = ares.simulations.Global21cm()
    sim.run()
    sim.GlobalSignature()

See :doc:`example_21cm_simple` in :doc:`examples` for a more thorough 
introduction to this type of calculation.

Contents
--------
.. toctree::
    :maxdepth: 1

    Home <self>
    install
    examples
    params
    fields
    .. inits_tables
    uth
    troubleshooting
    updates
    contributing
    history

.. Indices and tables
.. ==================
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

