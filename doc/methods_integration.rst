:orphan:

Integrating the Cosmological Radiative Transfer Equation
========================================================
We implemented multiple methods for integrating the cosmological radiative transfer equation (C-RTE), which we outline here. The most important piece of this is computing the IGM optical depth,

.. math::

  \overline{\tau}_{\nu}(z, z^{\prime}) = \sum_j \int_{z}^{z^{\prime}} n_j(z^{\prime \prime}) \sigma_{j, \nu^{\prime\prime}} \frac{dl}{dz^{\prime\prime}}dz^{\prime\prime}
  
  
==============   =============   =========   ========
                  Parameters                 Behavior
------------------------------------------   --------
frequency_bins   redshift_bins   tau_table   xray-flux
==============   =============   =========   =========
 not None             None          None     2-D table
     None         not None          None     generator
     None             None        not None   generator
     None             None          None     function
==============   =============   =========   =========
  
CXRB Generator
--------------



CXRB Tabulation
---------------
  