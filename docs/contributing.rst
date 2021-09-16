*ARES* Development: Contributing!
=================================
If *ARES* lacks functionality you're interested in, but seems to exhibit some 
features you'd like to make use of, adapting it to suit your purpose should
(in principle) be fairly straightforward. The following sections describe
how you might go about doing this. 

If you end up developing something that might be useful for others and
are willing to share, you should absolutely `fork ares on bitbucket <https://bitbucket.org/mirochaj/ares/fork>`_.
Feel free to shoot me an email if you need help getting started!

.. Minor Changes
.. -------------
.. 
.. New Parameters
.. ~~~~~~~~~~~~~~
.. 
.. 
.. New Fields
.. ~~~~~~~~~~

Adding new modules: general rules
---------------------------------
There are a few basic rules to follow in adding new modules to *ARES* that should prevent major crashes. They are covered below.

Imports
~~~~~~~
First and foremost, when you write a new module you should follow the hierarchy that's already in place. Below, the pre-existing sub-modules within *ARES* are listed in an order representative of that hierarchy:

- inference
- simulations
- solvers
- static
- populations, sources
- physics, util, analysis

It will hopefully be clear which sub-module your new code ought to be added to. For example, if you're writing code to fit a particular kind of dataset, you'll want to add your new module to ``ares.inference``. If you're creating new kinds of source populations, ``ares.populations``, and so on. If you're adding new physical constants, rate coefficients, etc., look at ``ares.physics.Constants`` and ``ares.physics.RateCoefficients``.

Now, you'll (hopefully) be making use of at least some pre-existing capabilities of *ARES*, which means your module will need to import classes from other sub-modules. There is only one rule here: 

**When writing a new class, let's say within sub-module X, you cannot import classes from sub-modules Y that lie above X in the hierarchy.** 

This is to prevent circular imports (which result in recursion errors).

Inheritance
~~~~~~~~~~~
You might also want to inherit pre-existing classes rather than simply making new instances of them in your own. For example, if creating a class to represent a new type of source population, it would be wise to inherit the ``ares.populations.Population`` class, which has a slew of convenience routines. More on that later.

Again, there's only one rule, which is related to the hierarchy listed in the above section:

**Parent Classes (i.e., those to be inherited) must be defined either at the same level in the hierarchy as the Child Classes or below.**

This follows from the rule about imports, since a class must be either defined locally or imported before it can be inherited.

.. Adding new modules: specific examples
.. -------------------------------------
.. 
.. New Source Populations
.. ~~~~~~~~~~~~~~~~~~~~~~
.. 
.. New Simulations
.. ~~~~~~~~~~~~~~~
.. 
.. New Fitters
.. ~~~~~~~~~~~







