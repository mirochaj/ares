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
    Hydrogen number density in units of :math:`\text{cm}^{-3}` 
    Default: :math:`1 \ \text{cm}^{-3}` 
    
``length_units``
    Default: :math:`6.6 \times 3.08568 \times 10^{21} \ \text{cm}` (i.e., 6.6 kilo-parsec) [centimeters]
    
``time_units``
    Default: :math:`3.15576 \times 10^{13} \ \text{s}` (i.e., 1 Myr) [seconds]    

``initial_ionization``
    

``initial_temperature``