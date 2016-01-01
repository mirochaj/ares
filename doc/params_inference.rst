:orphan:

Parameter Estimation Parameters
===============================


``inline_analysis``    
    This parameter controls how much information is automatically computed at
    the end of a calculation. We can choose whichever IGM quantities we like,
    and a series of redshifts at which to compute them.
    
    The value of ``inline_analysis`` must be a list containing  
    two sub-lists, where the first sub-list is a list of quantities to be 
    computed, and the second is a sequence of redshifts.
    
    For example

    Non-standard (non-numerical) redshifts of interest:
    + ``'B'``: Turning point B in sky-averaged 21-cm signal.
    + ``'C'``: Turning point C in sky-averaged 21-cm signal.
    + ``'D'``: Turning point D in sky-averaged 21-cm signal.
    + ``'trans'``: Zero-crossing in sky-averaged 21-cm signal.
    + ``'eor_midpt'``: Redshift when HII region volume filling factor is 50%.
    + ``'eor_overlap'``: Redshift when HII region volume filling factor is 99%.
    
    Default: None


