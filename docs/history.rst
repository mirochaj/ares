Development History
===================
*ARES* used to exist as two separate codes: *rt1d* and *glorb*, which were introduced in `Mirocha et al. (2012) <http://adsabs.harvard.edu/abs/2012ApJ...756...94M>`_ and `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014arXiv1406.4120M>`_, respectively. Since then, the codes have been combined and restructured to provide a more unified framework for doing radiative transfer calculations, modeling of the global 21-cm signal, and exploring all types of parameter spaces using MCMC.

Here's an attempt to keep track of major changes to the code over time, which will be tagged in the bitbucket repository with version numbers. I haven't followed conventions for version numbering so far. Instead, I've simply tagged commits with a version number when a paper is submitted using that version of the code (e.g., v0.1 and v0.2), or when a series of noteworthy improvements or bug fixes have been made (v0.3).

v0.5
----
- This is the version of the code used in `Mirocha & Furlanetto (2019) <http://adsabs.harvard.edu/abs/2018arXiv180303272M>`_. 
- Note that several ``litdata'' modules have been updated so that the year is reflective of the year the paper was *published*, not *submitted*!

v0.4
----
- This is the version of the code used in `Mirocha et al. (2018) <http://adsabs.harvard.edu/abs/2018MNRAS.478.5591M>`_. The main addition is global Lyman-Werner feedback, which raises the minimum mass of star-forming halos self-consistently using an iterative technique. 

v0.3
----
- Updated to work with `hmf <http://hmf.readthedocs.org/en/latest/>`_ version 2.0.1.
- Bug fix in :math:`S_{\alpha}` calculation for Furlanetto & Pritchard (2006): sign error in higher order terms.
- Generalized *HaloProperty* objects from version 0.2 to allow dependence on any number of arbitrary quantities. Now called *ParameterizedQuantity* object.

v0.2
----
This is the version of the code used in `Mirocha, Furlanetto, \& Sun (submitted) <http://arxiv.org/abs/1607.00386>`_. 

Main (new) features:

- Can model the star-formation efficiency as a mass- and redshift-dependent quantity using *HaloProperty* objects.
- This, coupled with the *GalaxyPopulation* class, allows one to generate models of the galaxy luminosity function. Also possible to fit real datasets (using ``ares.inference.FitLuminosityFunction`` module).
- Creation of a *litdata* module to facilitate use of data from the literature. At the moment, this includes recent measurements of the galaxy luminosity function and stellar population synthesis models (*starburst99* and *BPASS*).
- Creation of ``ParameterBundle`` objects to ease the initialization of calculations.


v0.1
----
This is the version of the code used in `Mirocha et al. (2015) <http://arxiv.org/abs/1509.07868>`_. 

Main features:

- Simple physical models for the global 21-cm signal available.
- Can use * `emcee <http://dan.iel.fm/emcee/current/>`_ to fit these models to data.








