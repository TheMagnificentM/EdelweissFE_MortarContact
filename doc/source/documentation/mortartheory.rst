Mortar contact: theory
======================

This page documents the segment-to-segment (mortar) contact constraint
:mod:`~edelweissfe.constraints.mortarcontact`: what it constrains, how the two contact surfaces are
turned into an integration domain, the dual basis that keeps the multiplier field local, the
semi-smooth treatment of the unilateral condition and the three formulations that enforce it, and
how to use it. The input options are listed under :doc:`constraints`.

It is the counterpart of :doc:`contacttheory`, which documents the node-to-surface and the
integrated (Gauss-point-to-surface) contacts. The difference is what is constrained: a
node-to-surface method asks whether each slave node has penetrated the opposite surface, the mortar
method integrates the non-penetration condition over the overlap of the two surfaces. That is what
lets it transmit a constant pressure exactly across an interface whose two sides do not share nodes.


The formulation
---------------

The weighted gap
~~~~~~~~~~~~~~~~

The two surfaces are element sets of geometry-only contact elements
(:mod:`~edelweissfe.elements.contactelement.element`), whole faces of the solid elements, as
produced by :mod:`~edelweissfe.generators.surfaceelementgenerator` with ``facets=wholeFace``. One of
them is the *non-mortar* (slave) surface and carries the constraint; the other, the *mortar*
(master) surface, is projected onto it.

Per non-mortar node :math:`I` the gap is required to vanish in a weighted sense,

.. math::

    g_I = \int_\Gamma \Phi_I \, \boldsymbol{n} \cdot \big(\boldsymbol{x}^{(2)} - \boldsymbol{x}^{(1)}\big)
    \, \mathrm{d}\Gamma ,

with :math:`\Phi_I` a dual basis function of node :math:`I` (see :ref:`mortar-dual-basis`), the
normal :math:`\boldsymbol{n}` interpolated from the nodal normals, and the integral running over the
*overlap* of the two surfaces, so that both discretizations enter it. Discretized, the weighted gap
is built from two coupling matrices,

.. math::

    D_{IK} = \int_\Gamma \Phi_I \, N_K \, \mathrm{d}\Gamma , \qquad
    C_{IJ} = \int_\Gamma \Phi_I \, N_J \, \mathrm{d}\Gamma ,

with :math:`K` the non-mortar and :math:`J` the mortar nodes, and their vector-valued counterparts
that carry the normal at the integration points (in two dimensions simply :math:`D_{IK}
\boldsymbol{n}_I` and :math:`C_{IJ} \boldsymbol{n}_I`).

Every nodal quantity acquires a weight, the *nodal mortar weight*

.. math::

    D_{II} = \sum_K D_{IK} = \int_\Gamma \Phi_I \, \mathrm{d}\Gamma :

* :math:`g_I / D_{II}` is the physical opening at the node (positive open);
* :math:`\lambda_I \, D_{II}` is a nodal force, and the multiplier :math:`\lambda_I` the contact
  pressure, positive in compression;
* a node with :math:`D_{II} = 0` has no material opposite it.

For linear elements :math:`D_{II} > 0` at every covered node. For quadratic faces it can become
negative under partial coverage; the pressure is then :math:`\lambda_I \operatorname{sgn} D_{II}`,
and the sign is carried explicitly wherever it decides a branch.


The unilateral condition
~~~~~~~~~~~~~~~~~~~~~~~~

The Signorini conditions -- non-negative pressure, non-negative gap, no pressure where there is a
gap -- are written as one non-smooth complementarity function per node,

.. math::

    C_{n,I} = p_I - \max\left(0,\; p_I - c_n \, g_I / D_{II}\right) = 0 ,

so that node :math:`I` is active where :math:`p_I - c_n \, g_I / D_{II} > 0`. A semi-smooth Newton
method -- the primal-dual active set strategy -- re-decides the active set in every iteration
(Hintermüller, Ito & Kunisch 2002; Hüeber & Wohlmuth 2005; Gitterle et al. 2010). The indicator uses
the physical opening :math:`g_I / D_{II}`, a length, where Gitterle et al. use the weighted gap; with
the opening, :math:`c_n g_I / D_{II}` is a pressure comparable to :math:`p_I`. At convergence either
gap vanishes, so the converged solution is the same.

The parameter ``cn`` is purely algorithmic: it does not change the converged solution, only the path
to it. Its unit is a stress per length, so a value that suits a model in metres does not suit the same
model in millimetres. Too small, and the active set does not settle; far too large, and it
oscillates. Left at 0 it is taken as the smallest Young's modulus of the materials adjacent to the
interface, which is reported.

Two details of the active-set iteration:

* A *closed* node without pressure -- zero multiplier and a gap that is zero up to the noise floor of
  the geometry -- counts as **active**. At the kink of :math:`\max(0, \cdot)` the semi-smooth Newton
  method may use any element of the generalized derivative (Hintermüller, Ito & Kunisch 2002), and
  only the active side keeps the tangent of a body that is carried by the contact alone regular when
  it starts exactly touching.
* The set is frozen for the rest of an increment once it has been unchanged for two iterations, or
  once a state already visited in that increment recurs (a cycle), or after 20 iterations. Only the
  last is reported, since only it leaves the set unconfirmed.


Three formulations
~~~~~~~~~~~~~~~~~~

All three share the whole geometric pipeline -- search, projection, clipping, segmentation, nodal
normals, coupling matrices -- and differ only in how the discrete condition is enforced
(``formulation``):

``lagrange`` (default)
    The multipliers are unknowns of the global system, one scalar variable per non-mortar node, and
    the system becomes a saddle-point problem. An active node contributes the constraint row
    :math:`g_I = 0` and the forces :math:`-\lambda_I D_{IK} \boldsymbol{n}_I` and
    :math:`+\lambda_I C_{IJ} \boldsymbol{n}_I`, an inactive one the row :math:`\lambda_I = 0`. The
    condition holds exactly at convergence. Each multiplier row is scaled by the nodal factor of Popp
    et al. (2013) (see :ref:`mortar-boundary-rows`).

``penalty``
    No unknowns are added; the pressure is :math:`\varepsilon_N` (``penaltyStiffness``) times the
    pointwise penetration,

    .. math::

        p_I = \max\left(0,\; z_I + \frac{\varepsilon_N}{|D_{II}|} \, g^{\text{pen}}_I\right) ,
        \qquad g^{\text{pen}}_I = -g_I \operatorname{sgn} D_{II} ,

    with :math:`z_I = 0` for pure penalty. :math:`\varepsilon_N` is the classical pointwise penalty
    parameter (a pressure per length), the same quantity as the ``penalty`` of the node- and
    segment-based contacts. The division by the nodal weight sits in the stiffness, node by node: a
    single stiffness for all nodes would make the pointwise stiffness proportional to the nodal
    weight, and the contact patch test would fail at every finite stiffness. A node without coverage
    gets no spring.

``penalty`` with ``augmentedLagrange``
    As ``penalty``, with the estimate :math:`z_I` advanced by the penalty pressure after each
    converged equilibrium, :math:`z_I \leftarrow \max(0, z_I + \varepsilon_N \delta_I)` with the
    penetration :math:`\delta_I`, and equilibrium re-established (Uzawa; Puso, Laursen & Solberg 2008).
    The converged result then satisfies the condition to ``augmentationTolerance`` and no longer
    depends on :math:`\varepsilon_N`, which only sets how fast the outer loop converges. The
    tolerance is a relative change of the estimate, independent of the length unit; reaching
    ``maxAugmentations`` is reported.

The multipliers of ``lagrange`` are not condensed out of the system: a condensation would have to
replace rows of the assembled system, which the constraint interface does not provide.


The geometry and the tangent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The segmentation, the coupling matrices and the nodal normals depend on the configuration. Two modes
are available (``geometryUpdate``); in both the tangent is that of the residual with the geometry
held fixed, i.e. without the derivatives of the normals and of :math:`D` and :math:`C`. In three
dimensions it is not symmetric, since the weighted gap carries the normal at the integration points
and the forces the nodal one.

``frozen`` (default, and always under explicit dynamics)
    The geometry is built once per increment at the last converged configuration and held fixed for
    the increment. The tangent is then exact for the assembled residual and the Newton iteration
    converges quadratically, but the result carries an error of first order in the increment size,
    since the geometry belongs to the start of the increment.

``incomplete`` (implicit solvers only)
    The geometry is rebuilt at every Newton iterate and held fixed within the Newton step (Hartmann
    2007, Section 6.2.4; the *incomplete linearization* of Popp et al. 2010). The converged state
    satisfies the contact conditions in its own configuration, at the price of a slower Newton
    iteration and a more expensive assembly.

On ``MortarContactTilted`` the frozen error halves with the increment size (2.0e-2 of the largest
displacement at ``maxInc=0.5``, 1.3e-3 at 0.03125), while the incomplete result does not depend on it.
``frozen`` is the robust default. ``incomplete`` pays off where the geometry moves substantially within
an increment, e.g. sliding under coarse increments; at slivers of overlap (see
:ref:`mortar-limitations`) the geometry rebuilt in every iteration is not smooth, and the iteration
may then fail where ``frozen`` converges.

The geometry is always built at the last *converged* state rather than at the extrapolated state the
solver starts an increment from, so that the converged solution does not depend on the extrapolation,
and a cutback re-attempt starts from the same geometry.


Friction in the mortar setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``mu > 0`` the ``penalty`` formulation adds Coulomb friction, under the implicit and the
explicit solvers (``lagrange`` with ``mu > 0`` is refused). The law -- elastic stick predictor,
radial return onto the cone, consistent tangent, history, anti-chattering, ``slipTolerance`` -- is the
one shared by all contact formulations of EdelweissFE and is described in the friction section of
:doc:`contacttheory`. A non-mortar node enters it as a contact point with the coefficients
:math:`\boldsymbol{c} = [D_{IK}, -C_{IJ}] / |D_{II}|`, the measure :math:`A = |D_{II}|`, its nodal
normal and its pressure, so that its slip increment is the mortar-weighted slip
:math:`(\boldsymbol{D}\,\Delta\boldsymbol{u}_s - \boldsymbol{C}\,\Delta\boldsymbol{u}_m)_I / |D_{II}|`
(Yang, Laursen & Meng 2005; Puso, Laursen & Solberg 2008).

That slip, measured with the coupling of the assembly, reads a rigid rotation of a penetrating pair as
slip. At every rebuild of the geometry at a converged configuration the accepted traction is
therefore corrected by the change of the pointwise weighted gap vector,
:math:`\varepsilon_T (\boldsymbol{g}_I^{new}/|D_{II}^{new}| - \boldsymbol{g}_I^{old}/|D_{II}^{old}|)`
with :math:`\boldsymbol{g}_I = \sum_K D_{IK} \boldsymbol{x}_K - \sum_J C_{IJ} \boldsymbol{x}_J`, which
turns it into the objective slip of Yang, Laursen & Meng (2005) between two rebuilds. Under
``incomplete`` a remainder of second order in the increment is left.

The nodal results -- pressure, frictional traction (as a vector), stick/slip state and opening -- are
available through ``getNormalPressures``, ``getTangentialTractions``, ``getSlipStates`` and
``getGaps``, ordered like the node set ``<name>_nonMortarNodes`` that the constraint registers.


Surfaces and integration
------------------------

Segmentation of the overlap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In three dimensions the overlap is constructed per non-mortar face as in Popp et al. (2010),
Section 4.1:

#. candidate mortar faces are found through a bounding-volume hierarchy (Ericson 2005); the box of
   every face is widened by 0.15 of its size and by the largest displacement of a contact node since
   the previous geometry build, so that a pair that approaches by as much again until the next build
   is already segmented;
#. an auxiliary plane is set up through the centre of the non-mortar face, with the nodal normals
   interpolated there as its normal;
#. both faces are projected into that plane, and the projections are clipped against each other
   (Sutherland & Hodgman 1974); the clip polygons have to be convex, which is reported otherwise;
#. the overlap is triangulated into integration cells (a fan from its first vertex, this
   implementation's choice), each integrated with the seven-point rule of
   Dunavant (1985);
#. each integration point is mapped back onto both faces along the auxiliary normal.

Quadratic faces are split into linear sub-segments for the clipping (Popp et al. 2010, Section 4.2;
the decompositions of Puso, Laursen & Solberg 2008). The integration points are mapped back by the
inverse of the isoparametric map of the whole face, and attributed to the face that contains them --
where an edge between two quadratic faces is curved, the straight sub-segments clip the strip between
chord and edge to the neighbouring face. This attribution is this implementation's own step.

In two dimensions the segments follow from projecting the end nodes of each element onto the other
side along the continuous normal field (Popp, Gee & Wall 2009, Section 4).


.. _mortar-dual-basis:

The dual basis
~~~~~~~~~~~~~~

The multiplier is discretized in a basis :math:`\Phi` biorthogonal to the (transformed) standard
basis :math:`\tilde N`,

.. math::

    \int_\Gamma \Phi_a \, \tilde N_b \, \mathrm{d}\Gamma = \delta_{ab} \int_\Gamma \tilde N_a \,
    \mathrm{d}\Gamma

(Wohlmuth 2000). This makes the slave coupling diagonal for linear elements, so that a nodal
multiplier belongs to one node and the constraint stays local. The coefficients are computed per face
from the integration cells, over the part of the face that is actually integrated (Popp, Seitz, Gee &
Wall 2013, Section 4.1; Cichosz & Bischoff 2011 in two dimensions). Where that system is singular on a
sliver of overlap, the coefficients of the whole element are used instead, which is reported.

For the quadratic faces ``CONQUAD8``, ``CONTRI6`` and ``CONLINE3`` a basis transformation is applied
first, which shifts a fraction :math:`\alpha` of each mid-side function onto its two corners (Popp,
Wohlmuth, Gee & Wall 2012):

.. math::

    \tilde N_c = N_c + \alpha \left( N_{m_1} + N_{m_2} \right) , \qquad
    \tilde N_m = \left( 1 - 2\alpha \right) N_m , \qquad \alpha = \tfrac{1}{3} .

Without it the corner functions of an eight-node face integrate to :math:`-A/12`, those of a six-node
triangle to zero, and the dual basis would not exist. The transformation keeps the partition of unity.
:math:`\alpha = 1/3` is the value Farah (2018) uses for the analogous transformation of quadratic
tetrahedra; Popp et al. (2012) suggest :math:`1/5` and give the conditions :math:`\alpha > 0` (six-node
triangle) and :math:`\alpha > 1/8` (eight-node quadrilateral), which both values satisfy. :math:`1/3` gives
the shallower negative dip of the transformed corner function, which matters where a face is only partly
covered. ``CONQUAD9`` needs no transformation and gets none. Since
the transformed basis is not biorthogonal to the untransformed one, :math:`D` is not diagonal for
quadratic faces; it is used in full.


Nodal normals
~~~~~~~~~~~~~

The normal of a non-mortar node is the average of the unit element normals of the adjacent faces,
each evaluated at the node's own natural coordinate, normalized again (Popp et al. 2010; the same average is written out with unit normals by
Farah 2018):

.. math::

    \boldsymbol{n}_I = \frac{\sum_{e \ni I} \boldsymbol{n}_I^e}{\left\| \sum_{e \ni I}
    \boldsymbol{n}_I^e \right\|} , \qquad
    \boldsymbol{n}_I^e = \frac{\boldsymbol{x}_{,\xi}(\boldsymbol{\xi}_I^e) \times
    \boldsymbol{x}_{,\eta}(\boldsymbol{\xi}_I^e)}{\left\| \boldsymbol{x}_{,\xi}(\boldsymbol{\xi}_I^e)
    \times \boldsymbol{x}_{,\eta}(\boldsymbol{\xi}_I^e) \right\|} .

Evaluated at the node, the normal follows the curvature of quadratic faces. Across a sharp edge -- a
step between a shaft and a head, say -- it averages normals that are far apart and belongs to neither
face: a contact surface should not be carried across such an edge in one piece.


.. _mortar-boundary-rows:

Boundary rows and the choice of sides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A non-mortar face that the other surface covers only in part integrates its basis functions over the
covered piece alone, so the entries of its rows in :math:`D` and :math:`C` become small. Under
``formulation=lagrange`` each row is therefore scaled by the nodal factor :math:`\kappa_I` of Popp,
Seitz, Gee & Wall (2013), Section 4.2 -- the integrated fraction of the node's faces, one for a fully
covered node -- and the pressure is recovered as :math:`\bar\lambda_I / \kappa_I`. One factor per row
leaves the constraint unchanged. The penalty form needs no scaling: its pressure is a pointwise law of
the penetration. For quadratic faces the factor integrates the transformed shape functions, which is
this implementation's extension of a first-order definition.

Which surface is the non-mortar one matters where the surfaces differ in extent. With the **smaller**
surface on the non-mortar side every one of its faces is fully covered, and no boundary row arises at
all (Cichosz & Bischoff 2011 show the effect by exchanging the sides). The decks
``MortarContactPunchEdge`` (favourable choice) and ``MortarContactPartialOverlap`` (unfavourable) are
the same configuration with the two choices.


Explicit dynamics with the mortar contact
-----------------------------------------

Only ``formulation=penalty`` can be integrated explicitly. ``lagrange`` is refused, since its
multipliers carry no inertia and an explicit increment solves no system; ``augmentedLagrange`` is
refused, since its outer loop runs only after a converged implicit equilibrium and would silently
never run.

The geometry is rebuilt every ``contact-update-frequency`` increments of the explicit solver rather
than in every time step, since the segmentation dominates the cost of the constraint and a node moves
far less than a face size in between. Where a contact node has moved, since the last build, by more
than the search boxes were widened for, the geometry is rebuilt at once, so that a fast approach
cannot close a gap the search left out.

A penalty stiffness adds its own frequency, which the element-based critical time step does not
contain. The contact stiffness of one node is rank one, :math:`(\varepsilon_N / |D_{II}|)
\boldsymbol{v}_I \boldsymbol{v}_I^{\mathsf T}` with :math:`\boldsymbol{v}_I = \boldsymbol{w}_I
\otimes \boldsymbol{n}_I`, and Gershgorin's theorem bounds the largest eigenvalue of
:math:`\boldsymbol{M}^{-1}\boldsymbol{K}` (Kulak 1989; Ghesquière-Diérickx et al. 2026):

.. math::

    \omega^2 \le \varepsilon_N R , \qquad
    R = \max_a \frac{1}{m_a} \sum_I \frac{|v_{I,a}| \, \|\boldsymbol{v}_I\|_1}{|D_{II}|} ,
    \qquad
    \Delta t \le \frac{2}{\sqrt{\varepsilon_N R}} .

The explicit solver fixes its time increment from the elements and constraints when it builds its
equation system (at the start of the step, and again after a refinement). The contact geometry is
not built at that point, so the bound does not lower the time increment; instead the stiffness is
adapted to it. Without a given ``penaltyStiffness`` it is derived as half the largest value the
current time increment can carry, :math:`\varepsilon_N = 2/(\Delta t^2 R)`, and re-derived at every
geometry rebuild; a given value that exceeds the bound is reported. With friction the tangential
stick spring shares the bound row by row.


Mesh refinement and restart
---------------------------

The constraint follows an h-adaptive refinement of either surface: the surface element generator
regenerates the contact elements, and the constraint rebuilds its contact system from them before the
next increment. A node that survives the refinement keeps its state (frictional history, augmented
estimate, active-set entry, multiplier); a new node takes the frictional history and the augmented
estimate interpolated from the old face it lies on, and the history is returned onto the friction cone
at the next rebuild. Under ``formulation=lagrange`` the scalar variables are re-created. A restart of a
refined run replays the refinement first and then restores the contact state by node label.


.. _mortar-usage:

Using the mortar contact
------------------------

Setting up the two surfaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both surfaces are element sets of contact elements. They are usually generated from surfaces of the
solids by :mod:`~edelweissfe.generators.surfaceelementgenerator` with ``facets=wholeFace``, which
creates one contact element per solid face on the solid's own nodes, collected in the element set
``<name>_facets``:

.. list-table::
    :header-rows: 1
    :widths: 40 30 30

    * - solid elements
      - face
      - contact element
    * - two-dimensional, linear
      - two-node line
      - ``CONLINE2``
    * - two-dimensional, quadratic
      - three-node line
      - ``CONLINE3``
    * - ``hexa8``
      - four-node quadrilateral
      - ``CONQUAD4``
    * - ``hexa20``
      - eight-node quadrilateral
      - ``CONQUAD8``

The constraint also accepts ``CONQUAD9``, ``CONTRI3`` and ``CONTRI6``, but since EdelweissFE has no
solid elements with such faces, they can only come from a hand-written element set.

The constraint checks the two sets when it is created and when the geometry is first built:

* both must contain contact elements only, and they must not share a node -- two surfaces glued
  together are not a contact problem (use a tie constraint);
* each surface must be consistently wound, and the non-mortar normals must point towards the mortar
  surface; otherwise a warning names the offending facets (inverted facets or swapped sides).

**Which surface is the non-mortar one** is the most important decision of the setup. The constraint
acts on the non-mortar side: its nodes carry the gaps, the pressures and, under
``formulation=lagrange``, the unknowns. Put the **smaller** surface there, so that it is covered by
the other one everywhere (see :ref:`mortar-boundary-rows`); a non-mortar surface that extends beyond
a loaded mortar surface attracts part of the force onto its nodes outside the contact zone (see
:ref:`mortar-limitations`).

A contact surface should not run across a sharp edge of the body (see the nodal normals above). The
constraint registers its non-mortar nodes, in the order of its results, as the node set
``<name>_nonMortarNodes``.


A first example
~~~~~~~~~~~~~~~

Two blocks generated by ``boxGen`` (here ``lower`` and ``upper``), the top face of the lower block
as the non-mortar side and the bottom face of the upper one as the mortar side, with the default
Lagrange multiplier formulation::

    *modelGenerator, generator=surfaceElementGenerator, name=gen1
    surface = lower_top
    name    = nonMortarSurf
    facets  = wholeFace

    *modelGenerator, generator=surfaceElementGenerator, name=gen2
    surface = upper_bottom
    name    = mortarSurf
    facets  = wholeFace

    *constraint, name=contact, type=mortarcontact
    nonMortarSurface=nonMortarSurf_facets
    mortarSurface=mortarSurf_facets
    formulation=lagrange

Nothing else has to be set: ``cn`` (and, for the penalty forms, ``penaltyStiffness``) are derived from
the adjacent materials and the mesh, and the values used are written to the journal once, when the
geometry is first built. Where the surfaces start apart, ``penaltyStiffness`` is derived at the first
build at which they overlap and reported then.


Options
~~~~~~~

The complete descriptions are listed under :doc:`constraints` (``mortarcontact``); in short:

.. list-table::
    :header-rows: 1
    :widths: 24 14 62

    * - option
      - default
      - meaning
    * - ``nonMortarSurface``
      - required
      - element set of the non-mortar (slave) surface
    * - ``mortarSurface``
      - required
      - element set of the mortar (master) surface
    * - ``field``
      - ``displacement``
      - the field the constraint acts on
    * - ``formulation``
      - ``lagrange``
      - ``lagrange`` (multipliers, exact) or ``penalty``
    * - ``geometryUpdate``
      - ``frozen``
      - ``frozen`` (geometry of the last converged state, per increment) or ``incomplete`` (rebuilt
        at every Newton iterate); implicit solvers only, explicit dynamics always uses ``frozen``
    * - ``cn``
      - 0 (derived)
      - parameter of the active-set test, ``lagrange`` only; a stress per length
    * - ``penaltyStiffness``
      - 0 (derived)
      - :math:`\varepsilon_N`, pressure per unit penetration, ``penalty`` only
    * - ``augmentedLagrange``
      - ``False``
      - outer correction loop for ``penalty``; implicit solvers only
    * - ``augmentationTolerance``
      - 1e-6
      - relative change of the pressure estimate that ends the outer loop
    * - ``maxAugmentations``
      - 10
      - largest number of outer iterations per increment
    * - ``mu``
      - 0
      - Coulomb friction coefficient, ``penalty`` only
    * - ``tangentPenalty``
      - 0 (= :math:`\varepsilon_N`)
      - stick stiffness, a traction per unit slip
    * - ``slipTolerance``
      - 0 (off)
      - elastic slip as a fraction of the facet size; replaces ``tangentPenalty`` by a
        pressure-proportional stiffness (implicit solvers)

An option that the chosen formulation does not use (``cn`` under ``penalty``, ``penaltyStiffness``
under ``lagrange``) is reported as ignored.


Which formulation
~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 28 24 24 24

    * -
      - ``lagrange``
      - ``penalty``
      - ``penalty`` + ``augmentedLagrange``
    * - non-penetration
      - exact
      - up to :math:`p/\varepsilon_N`
      - up to ``augmentationTolerance``
    * - extra unknowns
      - one per non-mortar node
      - none
      - none
    * - result depends on :math:`\varepsilon_N`
      - --
      - yes
      - no
    * - friction (``mu > 0``)
      - no
      - yes
      - yes
    * - implicit solvers
      - yes
      - yes
      - yes
    * - explicit dynamics (``NED``)
      - no
      - yes
      - no

* ``lagrange`` is the reference when no friction is needed. Its unknowns make the system a saddle
  point; they are ordinary scalar variables of the model and are therefore part of the convergence
  check, the restart and the regression references.
* ``penalty`` keeps the system size. The interface behaves like a stiff layer: the penetration is the
  pressure divided by :math:`\varepsilon_N`, uniformly over the interface. It is the only choice for
  friction and for explicit dynamics.
* ``penalty`` with ``augmentedLagrange`` removes that compliance at the cost of re-solving the
  equilibrium a few times per increment.


Choosing the parameters
~~~~~~~~~~~~~~~~~~~~~~~

``cn``
    Best left at its default, the smallest Young's modulus of the materials adjacent to the interface.
    It does not change the converged solution, only how quickly the active set settles. If the journal
    reports that no Young's modulus was found (e.g. for a material without one), a fallback is used
    and ``cn`` should be set. Since its unit is a stress per length, the default suits models whose
    lengths are of order one; for a model in much larger or smaller units it may need scaling.

``penaltyStiffness``
    Left at 0 it is derived as :math:`\varepsilon_N = 100\,E/h`, with :math:`E` the smallest adjacent
    Young's modulus and :math:`h` the characteristic face size of the non-mortar surface, at the first
    overlap of the surfaces. The literature gives the scale, not a
    value: Wriggers (2006), Remark 6.4, finds a penalty of at least 100 times the axial stiffness
    necessary and recommends choosing it with the mesh size; Ghesquière-Diérickx et al. (2026) quote
    :math:`\alpha E/h` with :math:`\alpha` between :math:`10^1` and :math:`10^4`. A larger value
    reduces the penetration and worsens the conditioning of the system. It is the same quantity as the
    ``penalty`` of :mod:`~edelweissfe.constraints.nodetodeformablesurfacepenalty` and
    :mod:`~edelweissfe.constraints.surfacetodeformablesurfacepenalty`, so a value may be carried
    between them. Under explicit dynamics it is bounded by the time increment (see the section on
    explicit dynamics above).

``augmentationTolerance``, ``maxAugmentations``
    The defaults suit most cases. With the augmented loop the converged result no longer depends on
    :math:`\varepsilon_N`; a larger :math:`\varepsilon_N` only makes the loop converge in fewer
    iterations. Reaching ``maxAugmentations`` is reported: the increment was then accepted on a
    pressure estimate that had not settled.

``geometryUpdate``
    ``frozen`` is the robust default: quadratic Newton convergence and the cheapest assembly; the
    error of holding the geometry over an increment shrinks with the increment size. ``incomplete``
    makes the result independent of the increment size, for large sliding or rotation within an
    increment, at a linear Newton rate and with the risk described under :ref:`mortar-limitations`.

``mu``, ``tangentPenalty``, ``slipTolerance``
    With friction, set ``slipTolerance`` (e.g. 0.005, the default of the Abaqus penalty friction) for
    real interfaces: it lets every sticking node slip elastically by up to that fraction of the facet
    size, whatever its pressure, and keeps the stick/slip state of lightly loaded nodes from
    chattering. ``tangentPenalty`` is used only without ``slipTolerance``; its default follows
    :math:`\varepsilon_N`.

Element type
    ``CONQUAD8`` is the quadratic face to prefer. ``CONQUAD9`` receives no basis transformation, and
    its nodal weights can turn negative under partial coverage (reported).


Implicit solvers
~~~~~~~~~~~~~~~~

The constraint needs no solver option of its own. What happens inside the solver:

* The active set is re-decided in every Newton iteration (semi-smooth Newton); once it has settled
  it is frozen for the rest of the increment. If it has not settled after 20 iterations it is frozen
  as it is, and this is reported -- the increment then converged on a set that was not confirmed; a
  smaller increment is the remedy.
* Under ``frozen`` the geometry is rebuilt once per increment; a cutback re-attempt starts from the
  same converged state and the same geometry. A gap that closes within one increment by more than the
  search margin (0.15 of the smallest face plus the motion of the previous increment) is not seen
  until the next increment; such a motion is reported, and smaller increments are the remedy. The
  first increment has no previous motion to go by.
* Under ``formulation=penalty`` an increment is never accepted on its first, extrapolated state, since
  the contact forces of that state have not been equilibrated.
* The augmented loop runs after each converged equilibrium of an increment; the solver then iterates
  again until the estimate has settled.
* Under ``formulation=lagrange`` the residual of a multiplier row is a weighted gap, a length times an
  area, checked against the absolute tolerance of the scalar variables (``fluxResidualTolerance``,
  entry ``scalar variables``, changed with ``*updateConfiguration``). For models far above unit
  length this bound can fall below the rounding of the gap, and the tolerance has to be raised.


With the explicit solver
~~~~~~~~~~~~~~~~~~~~~~~~

With the explicit solver ``NED`` use ``formulation=penalty`` (with or without ``mu``). The solver
option ``contact-update-frequency`` sets after how many increments the geometry is rebuilt at the
latest; it is rebuilt earlier where the nodes move beyond the search margin. A smaller value follows
the changing geometry more closely and costs more. ``penaltyStiffness`` is best left at 0: it is
then derived from the stability bound of the current time increment and re-derived when the
geometry is rebuilt. A given value above that bound is reported together with the factor by which it
or the ``courant-number`` would have to be lowered::

    *solver, solver=NED, name=theSolver
    expect-second-order-fields="displacement"
    courant-number=0.8
    contact-update-frequency=10

    *constraint, name=contact, type=mortarcontact
    nonMortarSurface=nonMortarSurf_facets
    mortarSurface=mortarSurf_facets
    formulation=penalty


Output of the contact results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The nodal results are available as functions of the constraint, ordered like the node set
``<name>_nonMortarNodes``, and can be written with ``>>fromExpression`` like any other field output:

``getNormalPressures()``
    contact pressure, positive in compression (under ``lagrange`` the multiplier, recovered as a
    traction);
``getTangentialTractions()``
    frictional traction vector in global components, shape ``(nodes, dimension)``;
``getSlipStates()``
    0 open, 1 stick, 2 slip;
``getGaps()``
    normal opening, positive open, negative penetrating.

For example (the constraint named ``contact``)::

    *fieldOutput
    >>fromExpression, name=contactPressure, nSet=contact_nonMortarNodes, expression='model.constraints["contact"].getNormalPressures()'
    >>fromExpression, name=contactSlipState, nSet=contact_nonMortarNodes, expression='model.constraints["contact"].getSlipStates()'

At a node whose faces are covered only by a thin strip, the pressure is the coefficient of a dual
shape function over a small support and can exceed the pressure around it; the force the node
transmits is correct. The resultant contact force is most easily checked through the reaction forces
of the supports.


Messages and what to do about them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each warning is written once per constraint.

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - message (short)
      - meaning and remedy
    * - surfaces share nodes (error)
      - the two sets touch; contact needs two separate meshes
    * - inconsistently wound / normals point away from the mortar surface
      - inverted facets or swapped sides; check the surfaces passed to the generator
    * - no Young's modulus found, fallback used
      - set ``cn`` or ``penaltyStiffness`` explicitly
    * - active set frozen by the iteration cap
      - the active set did not settle; reduce the increment size
    * - converged active set not self-consistent
      - nodes would change their state after convergence; reduce the increment size
    * - augmented Lagrange stopped at ``maxAugmentations``
      - raise ``maxAugmentations`` or ``penaltyStiffness``, or reduce the increment
    * - contacting nodes with < 1 % coverage
      - non-mortar nodes at the edge of the contact zone; put the smaller surface on the non-mortar side
    * - nodes with negative mortar weight
      - quadratic faces under partial coverage; prefer ``CONQUAD8``, or the smaller surface as non-mortar
    * - non-convex sub-segment / Gauss point projection not converged / degenerate overlap
      - strongly distorted faces or slivers of overlap; the result is less accurate there
    * - penalty stiffness needs a smaller time increment
      - explicit dynamics: lower ``penaltyStiffness`` or ``courant-number`` by the factor given
    * - contact nodes moved by more than the search reserve within an increment
      - implicit, ``frozen``: a contact closing within the increment may be missed; reduce the increment
        size


Verification
------------

* **Patch tests** -- a constant pressure across a non-matching interface, against the closed-form
  uniaxial solution, in two and three dimensions and in both element orders
  (``MortarContactPatch``, ``MortarContactPatchQuadratic``, ``MortarContactPatchHexa8``,
  ``MortarContactPatchHexa20``): displacements to round-off, contact pressure exact at every node.
  ``MortarContactPatchHexa8`` works at a hundred times the length scale of the others.
* **Unilateral condition** -- ``MortarContactSeparation`` (closing, separating and closing again),
  ``MortarContactTilted`` (an interface partly closed and partly open; ``tests/test_mortarsignorini.py``
  checks the Signorini conditions of its converged state), ``MortarContactPartialOverlap`` and
  ``MortarContactPunchEdge`` (partial coverage, the two choices of sides).
* **Penalty and augmented Lagrange** -- ``MortarContactPenaltyHexa20``.
* **Friction** -- ``MortarContactFrictionSlide`` (implicit) and ``NEDMortarContactFriction``
  (explicit); ``tests/test_mortarfriction.py``, and ``tests/test_contactfriction_consistency.py`` for the
  shared law across the three contact formulations.
* **Explicit dynamics** -- ``NEDMortarContact``; the stability bound in ``tests/test_mortarcontact.py``.
* **Tangent** -- against central differences of the residual with the geometry held fixed, for all
  seven contact element types, both formulations, and friction in stick and slip
  (``tests/test_mortarcontact.py``); the increment-size independence of ``incomplete`` in
  ``tests/test_mortargeometryupdate.py``.
* **Building blocks** -- clipping, triangulation, coupling matrices and their row-sum identity
  :math:`\sum_K D_{IK} = \sum_J C_{IJ}`, coverage, dual basis and basis transformation
  (``tests/test_mortarcontact.py``, ``tests/test_contactelement.py``).
* **Refinement and restart** -- ``tests/test_contact_amr_restart.py``.


.. _mortar-limitations:

Known limitations
-----------------

Slivers at the end of a loaded surface
    Where the mortar surface ends while still transmitting full pressure (a flat punch, a bearing
    plate) and the non-mortar surface continues beyond it, the faces just outside the punch tilt under
    load and acquire a sliver of overlap. Their nodes are constrained like any other, and part of the
    force (a few percent, decreasing slowly under refinement) lands on them. The total force is
    unaffected. It does not arise with the smaller surface on the non-mortar side.

Non-smooth geometry under ``geometryUpdate=incomplete``
    At slivers and where edges of the two meshes coincide, the clipped geometry changes
    non-smoothly with the displacements; rebuilt in every Newton iteration, this can keep the
    iteration from converging (in particular for very stiff penalties), where ``frozen`` converges.

Curved edges of quadratic faces
    The integration cells follow the chords of the linear sub-segments, so a uniform pressure is
    transferred exactly only up to the quadrature of the cells that straddle a curved edge.

Stopping criterion of the augmented loop
    Where the penalty term alone already satisfies the condition to rounding, the relative change of
    the estimate stops decreasing, and the loop runs to ``maxAugmentations``; the result is correct.

Length scale of the multiplier rows
    The residual of a multiplier row is a weighted gap (a length times an area) with an absolute
    convergence bound. For models far above unit length the bound can fall below the rounding of the
    gap; raising ``fluxResidualTolerance`` of the scalar variables is then the remedy.

Not implemented
    Self-contact, friction under ``formulation=lagrange``, condensation of the multipliers.


Implementation of the mortar coupling
-------------------------------------

.. automodule:: edelweissfe.utils.mortarcoupling


Sources of the mortar formulation
---------------------------------

* B. I. Wohlmuth, *A mortar finite element method using dual spaces for the Lagrange multiplier*,
  SIAM J. Numer. Anal. **38** (2000) 989--1012.
* B. Yang, T. A. Laursen, X. Meng, *Two dimensional mortar contact methods for large deformation
  frictional sliding*, Int. J. Numer. Methods Eng. **62** (2005) 1183--1225.
* M. A. Puso, T. A. Laursen, J. Solberg, *A segment-to-segment mortar contact method for quadratic
  elements and large deformations*, Comput. Methods Appl. Mech. Eng. **197** (2008) 555--566.
* S. Hüeber, B. I. Wohlmuth, *A primal-dual active set strategy for non-linear multibody contact
  problems*, Comput. Methods Appl. Mech. Eng. **194** (2005) 3147--3166.
* M. Hintermüller, K. Ito, K. Kunisch, *The primal-dual active set strategy as a semismooth Newton
  method*, SIAM J. Optim. **13** (2002) 865--888.
* A. Popp, M. W. Gee, W. A. Wall, *A finite deformation mortar contact formulation using a primal-dual
  active set strategy*, Int. J. Numer. Methods Eng. **79** (2009) 1354--1391.
* A. Popp, M. Gitterle, M. W. Gee, W. A. Wall, *A dual mortar approach for 3D finite deformation
  contact with consistent linearization*, Int. J. Numer. Methods Eng. **83** (2010) 1428--1465.
* M. Gitterle, A. Popp, M. W. Gee, W. A. Wall, *Finite deformation frictional mortar contact using a
  semi-smooth Newton method with consistent linearization*, Int. J. Numer. Methods Eng. **84** (2010)
  543--571.
* T. Cichosz, M. Bischoff, *Consistent treatment of boundaries with mortar contact formulations using
  dual Lagrange multipliers*, Comput. Methods Appl. Mech. Eng. **200** (2011) 1317--1332.
* A. Popp, B. I. Wohlmuth, M. W. Gee, W. A. Wall, *Dual quadratic mortar finite element methods for 3D
  finite deformation contact*, SIAM J. Sci. Comput. **34** (2012) B421--B446.
* A. Popp, A. Seitz, M. W. Gee, W. A. Wall, *Improved robustness and consistency of 3D contact
  algorithms based on a dual mortar approach*, Comput. Methods Appl. Mech. Eng. **264** (2013) 67--80.
* S. Hartmann, *Kontaktanalyse dünnwandiger Strukturen bei großen Deformationen*, Dissertation,
  Bericht Nr. 49, Institut für Baustatik und Baudynamik, Universität Stuttgart, 2007.
* P. Farah, *Mortar methods for computational contact mechanics including wear and general volume
  coupled problems*, Dissertation, Technische Universität München, 2018.
* P. Wriggers, *Computational Contact Mechanics*, 2nd ed., Springer, 2006.
* I. E. Sutherland, G. W. Hodgman, *Reentrant polygon clipping*, Commun. ACM **17** (1974) 32--42.
* D. A. Dunavant, *High degree efficient symmetrical Gaussian quadrature rules for the triangle*,
  Int. J. Numer. Methods Eng. **21** (1985) 1129--1148.
* C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2005.
* R. F. Kulak, *Critical time step estimation for three-dimensional explicit impact analysis*, Argonne
  National Laboratory, CONF-8907161, 1989.
* T. Ghesquière-Diérickx, J.-F. Molinari, G. Anciaux, *Stability of extrinsic cohesive zone model with
  penalty-based contact in explicit dynamic fragmentation simulations*, Mech. Mater. **215** (2026)
  105581.
