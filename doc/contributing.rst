*ares* Development: Contributing!
=================================
If *ares* lacks functionality you're interested in, but seems to exhibit some 
features you'd like to make use of, adapting it to suit your purpose should
(in principle) be fairly straightforward. The following sections describe
how you might go about doing this. 

If you end up developing something that might be useful for others and
are willing to share, you should absolutely `fork ares on bitbucket <https://bitbucket.org/mirochaj/ares/fork>`_.
Feel free to shoot me an email if you need help getting started!

First things first: avoid breaking everything!
----------------------------------------------
There are a few basic rules to follow in developing *ares* that should prevent major crashes. They are covered below.

Imports
~~~~~~~
First and foremost, when you write a new module you should follow the hierarchy that's already in place. Below, the pre-existing sub-modules within *ares* are listed in an order representative of that hierarchy:

- inference
- simulations
- solvers
- static
- populations, sources
- physics, util, analysis

It will hopefully be clear which sub-module your new code ought to be added to. For example, if you're writing code to fit a particular kind of dataset, you'll want to add your new module to ``ares.inference``. If you're creating new kinds of sources populations, ``ares.populations``, and so on. If you're adding new physical constants, rate coefficients, etc., look at ``ares.physics.Constants`` and ``ares.physics.RateCoefficients``.

Now, you'll (hopefully) be making use of at least some pre-existing capabilities of *ares*, which means your module will need to import classes from other sub-modules. There is only one rule here: 

**When writing a new class, let's say within sub-module X, you cannot import classes from sub-modules Y that lie above X in the hierarchy.** 

This is to prevent circular imports (which result in recursion errors).

Inheritance
~~~~~~~~~~~
You might also want to inherit pre-existing classes rather than simply making new instances of them in your own. For example, if creating a class to represent a new type of source population, it would be wise to inherit the ``ares.populations.Population`` class, which has a slew of convenience routines. More on that later.

Again, there's only one rule, which is related to the hierarchy listed in the above section:

**Parent Classes (i.e., those to be inherited) must be defined either at the same level in the hierarchy as the Child Classes or below.**

This is a little obvious from the rule about imports, since a class must be either defined locally or imported before it can be inherited.

"Under the Hood": Walkthrough
-----------------------------
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
    
    # Containers for the data on each timestep
    all_t = []
    all_data = []
    
    # Evolve in time
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

.. note :: Had we executed `sim.run()` for this example, nothing interesting would
    have happened because the gas is neutral to begin with (by default), the
    ionization and heating rate coefficients are zero (also by default), and
    the gas is too cold to be collisionally ionized.

Medium-Advanced Hacking
-----------------------
Here's another example where we initialize a grid of 64 cells near a point 
source of ultraviolet photons, and add ionization and heating from a 
meta-galactic background.

First, setup a dictionary of important parameters. We'll take a short-cut and
adopt the default parameters for Problem #2 from the Radiative Transfer 
Comparison Project (`Iliev et al. 2006
<http://adsabs.harvard.edu/abs/2006MNRAS.371.1057I>`_):

::

    pars = \
    {
     'problem_type': 2,
    }
    
To actually initialize the calculation, now for
a set of gas parcels rather than just one, we use a new class specifically 
designed for evolving radiation fields near point sources:

::

    import ares
    
    sim = ares.simulations.RaySegment(**pars)

Now, we'll call `sim.step`, a generator for the
evolution of this entire set of gas parcels, which 
(as in :class:`ares.simulations.GasParcel`)
will yield the current time, time-step, and data on each iteration.


::

    all_t = []
    all_data = []
    for t, dt, data in sim.step():

        # Save current time and data
        all_t.append(t)
        all_data.append(data)
        
        # Ionization/heating rate coefficients due to presence of UV source
        RCs = sim.field.update_rate_coefficients(data, t)

        # Add a constant ionizing background (shape ``grid_cells`` by absorbing species)
        RCs['k_ion'] += 1e-16 * np.ones([64, 1])

        # Re-computes rate coefficients using gas properties in 'data', 
        # and [optionally] those pertaining to radiation field
        sim.update_rate_coefficients(data, **RCs)
        
To plot up a radial profile of the neutral fraction at the last time snapshot, 
you could do:

::

    import matplotlib.pyplot as pl
    
    pl.plot(sim.grid.r_mid, all_data[-1]['h_1'])
    
.. note:: The variable `sim.grid` is an instance of the :class:`ares.static.Grid.Grid`
    class, which (among other things) holds information about the physical
    size of grid cells and the domain. The attribute `r_mid` refers to the
    cell midpoints. The edges are accessible also (via `r` or `r_edg`), but
    have one more element, thus causing a ``ValueError`` if used in attempts
    to plot radial profiles.
    
    
The procedure of repeatedly calling the generator, updating rate coefficients,
and storing data is what is happening ''under the hood'' each time you call
the `run` method of a class in the :py:mod:`ares.simulations` module. If you
come up with some new type of calculation and are tired of calling the `step`
function explicitly, perhaps it's time to create a new submodule in
:py:mod:`ares.simulations` module!    

Advanced Hacking
----------------

New Classes
~~~~~~~~~~~

New Parameters
~~~~~~~~~~~~~~

New Source Populations
~~~~~~~~~~~~~~~~~~~~~~

New Simulations
~~~~~~~~~~~~~~~

New Fitters
~~~~~~~~~~~







