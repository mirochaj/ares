Initial Conditions & Lookup Tables
==================================

.. Cosmological Initial Conditions
.. -------------------------------
.. 
.. 
.. 
.. 
.. 
.. 
.. The Halo Mass Function
.. ----------------------





The Opacity of the Intergalactic Medium
---------------------------------------
Solutions for the evolution of the cosmic X-ray background are greatly accelerated if one tabulates the IGM opacity, :math:`\tau_{\nu}(z, z^{\prime})`, ahead of time (see in Appendix C of `Haardt & Madau (1996) <http://adsabs.harvard.edu/abs/1996ApJ...461...20H>`_ for some discussion of this technique). *ARES* automatically looks in ``$ARES/input/optical_depth`` for :math:`\tau_{\nu}(z, z^{\prime})` lookup tables. 

The shape of the lookup table is defined by the redshift range being considered (set by the parameters ``first_light_redshift`` and ``final_redshift``), the number of redshift bins used to sample that interval, ``tau_redshift_bins``, the minimum and maximum photon energies (``pop_Emin`` and ``pop_Emax``), and the number of photon energies (determined iteratively from the redshift and energy intervals and the value of ``tau_redshift_bins``).

By default, ares generates tables assuming the IGM is fully neutral, but that is not required. To make optical depth tables of your own, see ``$ARES/input/optical_depth/generate_optical_depth_tables.py``. See Section 3 of `Mirocha (2014) <http://adsabs.harvard.edu/abs/2014MNRAS.443.1211M>`_ for more discussion of this technique. 

Tables for ``mirocha2017``-like calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To generate the table used for the calculations in `Mirocha, Furlanetto, & Sun (2017) <http://adsabs.harvard.edu/abs/2017MNRAS.464.1365M>`_, modify the following lines of ``$ARES/input/optical_depth/generate_optical_depth_tables.py``:

:: 

    zf, zi = (5, 50)
    Emin = 2e2
    Emax = 3e4
    Nz = [1e3]
    helium = 1

.. note :: You can run the ``generate_optical_depth_tables.py`` script in parallel via, e.g., ``mpirun -np 4 generate_optical_depth_tables.py``, so long as you have MPI and mpi4py installed.

The set of parameters used for these calculations are described in the "Simulations" section of :doc:`param_bundles`.


