:orphan:

Inline Analysis
===============
When running a large number of models, each of which takes a few seconds (or more), it's important to do as much analysis "inline" as possible. For example, say we are interested in obtaining confidence intervals for quantities other than the free parameters of our model. Yes, we could go back later and re-run certain subsets of models and extract whatever information we want, but with a little planning, we can eliminate the need for these "extra" computations. The *emcee* code dubs such quantities `arbitrary meta-data blobs <http://dan.iel.fm/emcee/current/user/advanced/#arbitrary-metadata-blobs>`_, and as a result, any quantities computed during calculations in *ares* will be named "blobs" as well.

The goal of this section is to outline the general procedure used to save meta-data blobs of your choosing, which can be tricky because different quantities of interest are computed in very different ways and often are of diverse shapes and variables types.

A few examples of meta-data blobs:

- Scalar blobs (e.g., the CMB optical depth, :math:`\tau_e`, the midpoint of reionization, :math:`z_{\mathrm{rei}}`)
- 1-D blobs (e.g., the global 21-cm signal, :math:`\delta T_b(z)`, the thermal history, :math:`T_K(z)`)
- 2-D blobs (e.g., the star-formation efficiency, :math:`f_{\ast}(z, M_h)`, the meta-galactic radiation background intensity, :math:`J(z, \nu)`)

Example: Common Scalar and 1-D Blobs
------------------------------------
Let's learn by example. Here is a typical calculation (model for the global 21-cm signal), where we have modified the input parameters so that a few quantities of interest are saved:

- Scalars: extrema in the global 21-cm signal (which we label turning points B, C, and D). 
- 1-D arrays: the full ionization history, thermal history, and the global 21-cm signal.

So, we define a nested list containing the names of our blobs:

::

    blob_names = [['tau_e'], ['cgm_h_2', 'igm_Tk', 'dTb']]

The blobs ares sorted by their dimensionality: the first sublist contains the names of all scalar blobs, while the second contains the 1-D blobs. Important question: how do you know the names of blobs? The scalar blobs, in this case just ``tau_e`` (the CMB optical depth) are all *attributes* of the ``Global21cm`` simulation class (well, really of ares.analysis.Global21cm, but they get inherited). The 1-D blobs are all names of *ares* fields: see :doc:`fields` for more information.

Now, for the 1-D blobs we also need to provide a sequence of redshifts at which to save each quantity:

::

    blob_ivars = [None, [('z', np.arange(6, 21))]]
    
Notice that ``blob_ivars`` is a 2-element list (``ivars`` is short for "independent variables," since in general they need not be redshifts): one element for each blob group (scalar and 1-D). Since the scalars are just numbers, the first element in this list is just ``None``, while the second indicates that we'll save the desired quantities at redshifts (``'z'``) :math:`z=6,7,...,20`.

.. note :: *ares* works with redshift internally, so, if you wanted to sample equally over some frequency range, simply define that array first and convert to redshifts via :math:`z = (\nu_0 / \nu) - 1` where :math:`\nu_0 = 1420.4057` MHz.

We supply these lists via parameters of the same name:

::

    pars = \
    {
     'problem_type': 101,           # Simple global 21-cm problem
     'blob_names': blob_names,
     'blob_ivars': blob_ivars,
     'tanh_model': True,            # Just to speed things up
    }
    
Now, we just run the simulation in the usual way:

::    
    
    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
    sim.GlobalSignature()
    
To verify that the simulation knows about our blobs, some basic information is available via attributes:

::
    
    print sim.blob_names
    print sim.blob_ivars
    print sim.blob_dims
    print sim.blob_nd
    
The blobs themselves can be accessed via:

::

    sim.blobs
    
.. note :: In this case, using blobs isn't necessary since we have all data from the simulation at our fingertips. However, the attribute ``sim.blobs`` is extremely important for model grids or MCMC fits, as its contents are the only thing written to disk other than the MCMC samples themselves.

Special Redshifts
~~~~~~~~~~~~~~~~~
We'll often be interested in saving a series of quantities at the redshifts corresponding to the extrema of the global 21-cm signal, or the midpoint of reionization, etc. However, since those aren't known *a-priori*, we can't specify them like we did above. Instead, we tag a suffix (either ``B``, ``C``, or ``D``) onto pre-existing *ares* fields, i.e., 

::

    extrema = ['z_B', 'dTb_B', 'z_C', 'dTb_C', 'igm_Tk_C']
    blob_names = [extrema, ['cgm_h_2', igm_Tk', 'dTb']]
    
and so on.   

Example: Derived Blobs
----------------------
There are many quantities one might be interested in that are **not** computed by *ares* by default, but can be derived after-the-fact from quantities *ares* does compute. Things are setup such that you can provide your own function to compute such "derived blobs," or you can simply refer to built-in functions that are attributes of *ares* simulation objects.

To build on our previous example:

::

    # Note the addition of 'fwhm' and 'slope'
    blob_names = [['tau_e', 'z_C', 'dTb_C'], ['fwhm'], ['slope']]
    blob_ivars = [None, None, [('freq', np.arange(40, 151, 1))]]
    
The ``'fwhm'`` blob is just a number, while ``'slope'`` here will be saved at integer frequencies between 40 and 150 MHz.

Now, we must specify the functions needed to compute ``'fwhm'`` and ``'slope'``. In this case, we don't need to write them from scratch, as they already exist in ``ares.analysis.Global21cm``, which is inherited by ``ares.simulations.Global21cm``. *ares* will assume blob functions are attributes of the simulation class, which means these quantities are readily available:

::

    # Width in MHz, Slope in mK / MHz
    blob_funcs = [None, ['Width()'], ['Slope']]
    
Notice that the width function gets an empty set of parentheses -- this is because there is no independent variable for this quantity. Alternatively, the slope function is given without parentheses to indicate that it must be applied over a range of values.

Before running it, create a parameters dictionary:    

::

    pars = \
    {
     'problem_type': 101,           # Simple global 21-cm problem
     'blob_names': blob_names,
     'blob_ivars': blob_ivars,
     'blob_funcs': blob_funcs,      # NEW!
     'tanh_model': True,            # Just to speed things up
    }

To test:

::

    sim = ares.simulations.Global21cm(**pars)
    sim.run()
    
Check that we got our blobs:

::

    print sim.get_blob('fwhm')
    print sim.get_blob('slope', 150.) # @ 150 MHz

.. Example: 2-D blobs
.. ------------------
.. Now, let's track slightly more complex blobs. For example, if you're running models of galaxy populations (see :doc:`example_pop_galaxy`), you might want to save the galaxy luminosity function at a series of magnitudes *and* a series of redshifts. 

.. ::
.. 
..     blob_names = [['Mpeak', 'fpeak'], ['gamma']]
..     
.. ::
..     
..     blob_ivars = [redshift, [[4.9, 5.9], np.logspace(8, 11, 4)]]
.. ::
.. 
..     blob_funcs = [['pops[0].ham.Mpeak', 'pops[0].ham.fpeak'], ['pops[0].ham.gamma']],
..     
..     
.. 
.. 
.. ::
.. 
..     redshift = 3.8
..     
..     b15 = ares.util.read_lit('bouwens2015')
..     mags = b15.data['lf'][redshift]['M']
..     
..     base_pars = \
..     {
..      'pop_Tmin{0}': 1e5,
..      'pop_model{0}': 'ham',
..      'pop_Macc{0}': 'mcbride2009',
..     
..      'pop_lf_z{0}': [redshift],
..      
..      'pop_ham_fit{0}': 'fstar',
..      'pop_ham_Mfun{0}': 'poly',
..      'pop_ham_zfun{0}': 'const',
..       
..      'pop_lf_mags{0}': [mags],
..     
..      'pop_sed{0}': 'leitherer1999',
..      'pop_fesc{0}': 0.2,
..      'pop_ion_src_igm{1}': False,
..      
..      'problem_type': 101.2,
..      
..      'cgm_initial_temperature': 2e4,
..      'cgm_recombination': 'B',
..      'clumping_factor': 3.,
..      'load_ics': False,
..      
..      'blob_names': blob_names,
..      'blob_ivars': blob_ivars,
..      'blob_funcs': blob_funcs,
..      
..     }
.. 
.. Run the thing:    
..     
.. ::
..     
..     sim.run()
..     
.. and check the blobs
.. 
..     sim.blobs
    