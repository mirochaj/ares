Alternative Parameterizations for Sources and Spectra
-----------------------------------------------------
If you want to use *ares*'s numerical solvers, but don't care for the
available slew of input options (e.g., spectra, luminosity functions, etc.),
you can pass in functions of your own.

There are a few magical override parameters to help you with this:

* `spectrum`: a Python function of a single variable (photon energy)
* `emissivity`: a Python function of a two variables (photon energy and redshift)
* `rho_L`: a Python function of a single variable (redshift)



