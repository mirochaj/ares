Grid Parameters
===============


``grid_cells``
    Number of resolution elements in the grid (i.e., number of concentric
    spherical shells to consider in calculation).

``logarithmic_grid``
    If True, discretize space logarithmically, rather than linearly.
    
    Default: False

``start_radius``
    Ignore radiative transfer within this distance from the origin [length_units]
    Must be done in order to avoid divergence in flux as :math:`r\rightarrow 0`

``density_units``
    Default: :math:`10^{-3} m_H \ \text{cm}{-3}` 
    (i.e., equivalent of :math:`10^{-3}` hydrogen atoms per :math:`\text{cm}^{3}`)
    
``length_units``
    Default: :math:`6.6 \times 3.08568 \times 10^{21} \ \text{cm}` (i.e., 6.6 kilo-parsec) [centimeters]
    
``time_units``
    Default: :math:`3.15576 \times 10^{13} \ \text{s}` (i.e., 1 Myr) [seconds]    

``Z``
    List of atomic numbers of elements to be included in calculation.
    
    Default: [1] (hydrogen only)

``abundances``
    List of elemental abundances relative to hydrogen, corresponding to elements
    of Z.

``initial_ionization``
    

``initial_temperature``