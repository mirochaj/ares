:orphan:

Star Formation Efficiency Parameters
====================================


``sfe_Mfun``
    Function used for the mass dependence of the SFE. Options currently include ``''lognormal''`` and ``''dpl''``.
    
    Default: ``lognormal``

``sfe_zfun``
    Function used for the redshift dependence of the SFE. Options currently include ``''constant''``.
    
    Default: ``constant``
    
``sfe_Mfun_par[0-4]``
    Parameters required by ``sfe_Mfun``. 
    
    ====================  =========================== ===========================
     ``sfe_Mfun``            ``lognormal``                ``dpl``         
    ====================  =========================== ===========================
     ``sfe_Mfun_par0``     :math:`f_{\mathrm{peak}}`   :math:`f_{\mathrm{peak}}` 
     ``sfe_Mfun_par1``     :math:`M_{\mathrm{peak}}`  :math:`M_{\mathrm{peak}}`
     ``sfe_Mfun_par2``     :math:`\sigma`                :math:`\gamma_1`
     ``sfe_Mfun_par3``     n/a                           :math:`\gamma_2`
    ====================  =========================== ===========================
     
    Default: ``None``
    
    