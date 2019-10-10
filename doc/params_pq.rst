:orphan:

Parameters for Parameterized Quantities
----------------------------------------
Parameterized quantities are most often used in the context of the galaxy luminosity function, where model the efficiency of star formation as a function of halo mass and (perhaps) redshift. See the ``mirocha2017`` option in :doc:`param_bundles` for a concrete example of how these parameters can be used. The basic idea is to provide a framework that enables *any* parameter to be parameterized more generally as a function of redshift, halo mass, etc. This potential is not yet fully realized, so beware that not all parameters can utilize this functionality!

A more detailed description of the methodology can be found here: :doc:`uth_pq`.

The relevant parameters are:

``pq_func``
    Function adopted. Options include ``pl``, ``dpl``, and many more. See listing below parameter(s) ``pq_func_par[0-5]``.

    Default: ``dpl``

``pq_func_var``
    Independent variable of ``pq_func``.

    Options:
        + ``mass``
        + ``redshift``

    Default: ``mass``

``pq_func_par[0-5]``
    Parameters required by ``pq_func``. Their meaning depends on the type of function employed. See below for meaning of each parameter by ``pq_func`` and number (:math:`x` is either redshift or halo mass in general).

    Options:
        + ``pl``: :math:`p[0] * (x / p[1])^{p[2]}`
        + ``dpl``: :math:`p[0] / ((x / p[1])^{-p[2]} + (x / p[1])^{-p[3]})`
        + ``dpl_arbnorm``: :math:`p[0](p[4]) / ((x / p[1])^-p[2] + (x / p[1])^-p[3])'`
        + ``pwpl``: :math:`p[0] * (x / p[4])^{p[1]}` if :math:`x \leq p[4]` else :math:`p[2] * (x / p[4])^{p[3]}`
        + ``plexp``: :math:`p[0] * (x / p[1])^{p[2]} * np.exp(-x / p[3])`
        + ``lognormal``: :math:`p[0] * np.exp(-(logx - p[1])^2 / 2 / p[2]^2)`
        + ``astep``: :math:`p[0]` if :math:`x \leq p[1]` else :math:`p[2]`
        + ``rstep``: :math:`p[0] * p[2]` if :math:`x \leq p[1]` else :math:`p[2]`
        + ``plsum``: :math:`p[0] * (x / p[1])^{p[2]} + p[3] * (x / p[4])^{p[5]}`

    Default: ``None``

``pq_func_var``
    Independent variable of ``pq_faux``.

    Options:
        + ``mass``
        + ``redshift``

    Default: ``None``