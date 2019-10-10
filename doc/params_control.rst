:orphan:

Control Parameters
==================

Output to screen
----------------
``verbose``
    Print lots of output to screen regarding status of calculation?
    
    Default: ``True``
    
``progress_bar``
    Use `python progress-bar <https://code.google.com/p/python-progressbar/>`_ (if installed)?
    
    Default: ``True``

Starting and stopping calculations
----------------------------------
``initial_redshift``
    Initial redshift of calculation, i.e., the redshift at which we switch from cosmological initial conditions to the *ARES* solver.
    
    Default: 60
    
``final_redshift``
    Calculation stops at this redshift.

    Default: 5    
    
``track_extrema``   
    Track 21-cm extrema in real-time. These are referred to as turning points
    B (first stars), C (first black holes), and D (beginning of Epoch of Reionization) in works such as `Burns et al. (2012) <http://adsabs.harvard.edu/abs/2012AdSpR..49..433B>`_, `Harker et al. (2012) <http://adsabs.harvard.edu/abs/2012MNRAS.419.1070H>`_, and `Mirocha et al. (2013) <http://adsabs.harvard.edu/abs/2013ApJ...777..118M>`_.
    
    Default: ``False``
    
``stop``
    If ``track_extrema==True``, set ``stop`` to ``'B'``, ``'C'``, or ``'D'`` to terminate the calculation once the given turning point is reached.
    
    Default: ``None``

``stop_xavg``
    You can also stop a calculation once a given mean ionized fraction is reached. For instance, if you'd like to terminate once the IGM is half ionized, set ``stop_xavg=0.5``.
    
    Default: 0.99999
    
Time-stepping and data storage
------------------------------
``time_units``
    Internal units for time.
    
    Default: :math:`3.15576 \times 10^{13} \ \text{s}` (i.e., 1 Myr)

``initial_timestep`` 
    Time-step at ``initial_redshift``.
    
    Default: 0.01 [``time_units``]
    
``max_dt``
    Maximum allowed time-step.
    
    Default: 1 [``time_units``]
        
``max_dz``
    Maximum allowed redshift-step.
    
    Default: None
    
``dtDataDump``
    Save all physical quantities at this time cadence.
    
    Default: 1 [``time_units``]
    
``dzDataDump``
    Save all physical quantities at this redshift cadence.
    
    Default: None

``epsilon_dt``
    Maximum fractional change in quantities of interest in a single time-step.
    Quantities of interest are listed in ``restricted_timestep`` (see below).
    
    Default: 0.01

``restricted_timestep``    
    A list containing quantities use to restrict the time-step via ``epsilon_dt``. Options:
    
    + ``'ions'``: restrict time-step based on rate of change in ion fractions.
    + ``'neutrals'``: restrict time-step based on rate of change in neutral fractions.
    + ``'electrons'``: restrict time-step based on rate of change in electron density.
    + ``'temperature'``: restrict time-step based on rate of change in temperature.
    + ``'hubble'``: restrict time-step based on Hubble expansion.
    
    Default: ``['ions', 'electrons', 'temperature']``


Lookup tables
-------------
``redshift_bins``
    Number of points to use when discretizing the IGM optical depth in redshift.
    
    Default: ``None``
    
``tau_prefix``
    Path to directory on disk where optical depth tables are stored. Set this if you keep optical depth tables stored in a place other than the ``$ARES`` environment variable!
    
    Default: ``None``

``load_sed``
    Same as ``tau_prefix``, but refers to lookup tables for complex spectral energy distributions (such as SIMPL) which are expensive to calculate.
    
    Default: ``False``

``sed_prefix``
    Location of SED tables
    
    Default: ``None``

Not done yet
------------


::
     
    # Initialization
    "load_ics": True,
    
    # Real-time optical depth calculation once EoR begins
    "EoR_xavg": 1.0,        # ionized fraction indicating start of EoR (OFF by default)
    "EoR_dlogx": 0.01,    

    "tau_table": None,
            
    "unsampled_integrator": 'quad',
    "sampled_integrator": 'simps',
    "integrator_rtol": 1e-6,
    "integrator_atol": 1e-4,
    "integrator_divmax": 1e2,
    
    "interpolator": 'spline',
    
    "progress_bar": True,
    "verbose": True,
