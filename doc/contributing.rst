Developing *ares*
=================
If *ares* lacks functionality you're interested in, but seems to exhibit some 
features you'd like to make use of, adapting it to suit your purpose should
(in principle) be fairly straightforward. The following section shows
how you might go about doing this. 

If you end up developing something that might be useful for others and
are willing to share, you should absolutely `fork ares on bitbucket <https://bitbucket.org/mirochaj/ares/fork>`_.
Feel free to shoot me an email if you need help getting started!

Basic Hacking
-------------
As mentioned in :doc:`structure`, `ares` uses Python generators heavily. The
main reason for this is to make adapting the code to your needs straightforward.
For example, if you don't want to bother with generating astrophysically-motivated
models for ionization and heating in some medium, you can easily adapt `ares` 
to prescribe some pre-existing models of your own. 

For illustrative purposes, let's consider a single zone comprised of 100% 
hydrogen at a density of 1 atom :math:`\mathrm{cm}^{-3}`. Let's also assume 
it is cold (:math:`T = 100` K). 

::
    
    pars = \
    {
     'grid_cells': 1,
     'density_units': 1.,
     'initial_temperature': 1e2,
    }

To actually initialize a gas parcel with these properties, 

::

    import ares
    
    sim = ares.simulations.GasParcel(**pars)
    
Now, we'll call `sim.step`, a generator for the
evolution of this gas parcel, which will yield the current time, time-step, 
and data on each iteration. All we'll do in this example is to force the 
ionization rate coefficient to be a constant:

::
  
    all_t = []
    all_data = []
    for t, dt, data in sim.step():
        
        # Save current time and data
        all_t.append(t)
        all_data.append(data)
        
        # Set the rate coefficient for photo-ionization
        ionization_RC = np.array([[1e-14]])
        
        # Re-computes rate coefficients using gas properties
        # and [optionally] updates rate coefficients related to radiation field
        sim.update_rate_coefficients(data, k_ion=ionization_RC)
        
.. note:: We supplied the ionization rate coefficient as a 2-D array because
    in general you can run calculations containing more than a single zone, and
    more than a single chemical species. In those cases, the rate coefficients
    you supply should have dimensions equal to the `(number of grid cells, number of absorbing species)`.
    For example, coefficients for a calculation on a 64-element grid including 
    helium (in addition to hydrogen) would have shape `(64, 3)`. We need two
    extra elements in the second dimension because helium can be either singly
    or doubly ionized.
    
The data for each snapshot is saved as a dictionary so that we can
access information by name. For instance, if we wanted to know the ionized
fraction at the final snapshot, we'd look at:

::

    all_data[-1]['h_2']
    
To piece together the entire evolution, we could do:

::
    
    xHII = [snapshot['h_2'] for snapshot in all_data]
    
Then plot it

::

    import matplotlib.pyplot as pl
    
    pl.plot(all_t, xHII)

        
Notice how crudely the earliest stages of the evolution are captured. This is 
because by default, the initial time-step is rather large. To fix this,
    
::  

    pf.update({'initial_timestep': 1e-8})
    
and re-run. The earliest stages of the evolution should be well resolved given 
:math:`\Delta t = 10^{-8}`.

Summary
-------
The procedure of repeatedly calling the generator, updating rate coefficients,
and storing data is what is happening ``under the hood'' each time you call the
`run` method of a class in the :py:mod:`ares.simulations` module. 

.. note :: Had we exected `sim.run()` for this example, nothing interesting would
    have happened because the gas is neutral to begin with (by default), the
    ionization and heating rate coefficients are zero (also by default), and
    the gas is too cold to be collisionally ionized.
