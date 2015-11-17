.. ares documentation master file, created by
   sphinx-quickstart on Mon Jul  8 08:48:22 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*ares*
==============================
The Accelerated Reionization Era Simulations (*ares*) code was designed to
rapidly generate models for the global 21-cm signal. It can also be used as a 
1-D radiative transfer code, stand-alone non-equilibrium chemistry solver, or
global radiation background calculator.

A few papers on how it works:

- 1-D radiative transfer: `Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_.
- Uniform backgrounds \& global 21-cm signal: `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_.
- Parameter estimation: `Mirocha et al. (2015) <http://adsabs.harvard.edu/abs/2015ApJ...813...11M>`_.

This documentation is very much a work in progress. Feel free to email me if you find gaps or errors.

Quick-Start
-----------
To make sure everything is working, a quick test is to generate a
realization of the global 21-cm signal using all default parameter values: 

::

    import ares

    sim = ares.simulations.Global21cm()
    sim.run()
    
    anl = ares.analysis.Global21cm(sim)
    ax = anl.GlobalSignature()

See :doc:`example_21cm_simple` in :doc:`examples` for a more thorough 
introduction to this type of calculation.

Contents
--------
.. toctree::
   :maxdepth: 1
   
   Home <self>
   install
   examples
   parameters
   fields
   troubleshooting
   contributing

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`