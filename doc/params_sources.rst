Source Parameters
=================



``source_type``
    Options:
    
    + ``star``: characterized by its temperature, ``source_temperature``, and ionizing photon luminosity, ``source_qdot``
    + ``bh``: characterized by its mass, ``source_mass`` etc.
    + ``toy``: completely parameterized

``source_temperature``
    If ``source_type`` is ``star``, this is its surface temperature. [Kelvin]

    Default: :math:`10^5 \ \text{K}`
``source_mass``
    If ``source_type`` is ``bh``, this is its mass [:math:`M_{\odot}`]
 
    Default: :math:`10 \ M_{\odot}` 
 
``source_qdot``
    Ionizing photon luminosity. [:math:`\text{s}^{-1}`]
    
    Default: :math:`5 \times 10^{48}\ \text{s}^{-1}` 
        
``source_lifetime``
    Time after which radiation from this source will no longer be considered.

    Default: :math:`10^{10}` [``time_units``]
    
 
::
    
    "source_fduty": 1,
    "source_tbirth": 0,
    "source_eta": 0.1,
    "source_isco": 6,  
    "source_rmax": 1e3,
    "source_cX": 1.0,
    
    "source_ion": 0,
    "source_ion2": 0,
    "source_heat": 0,
    "source_lya": 0,
    
    "source_table": None,
    "source_normalized": False,