Mortar contact: theory
======================

This page documents the theory of the segment-to-segment (mortar) contact constraint implemented in
EdelweissFE: the discretization of the two surfaces into geometry-only contact elements
(:mod:`~edelweissfe.elements.contactelement.element`,
:mod:`~edelweissfe.generators.surfaceelementgenerator` with ``facets=wholeFace``), the segmentation
of their overlap and the mortar coupling matrices it assembles into
(:mod:`~edelweissfe.constraints.mortarcontact`), the dual basis that keeps the multiplier field
local, the semi-smooth treatment of the unilateral condition, and the three formulations that
enforce it.

It is the counterpart of :doc:`contacttheory`, which documents the node-to-surface stack. The two
differ in what they constrain: a node-to-surface method asks whether each slave node has penetrated
the opposite surface, this one integrates the non-penetration condition over the overlap of the two
surfaces. That difference is the reason both exist, and it is worth stating before any formula.


Why integrate rather than sample
--------------------------------

Consider a flat interface whose two sides are discretized differently -- a common situation
wherever two meshed parts meet. Under a constant pressure the exact solution is a uniform state of
stress on both sides. A node-to-surface method weights each slave node by a tributary area that is a
property of the SLAVE mesh alone; the opposite surface it presses against has a different node
spacing, so the forces it transmits are distributed according to one mesh and received according to
another. The result is a pressure that oscillates from node to node even though the exact one is
constant. Refining either mesh does not remove the oscillation, it only shortens its wavelength.

A mortar method instead requires that the gap vanish in a weighted sense::

    g_I = integral over Gamma of  Phi_I * (gap)  dGamma  =  0

with one such condition per node :math:`I` of the non-mortar surface, and :math:`\Phi_I` a basis
function belonging to that node. The integral runs over the OVERLAP of the two surfaces, so both
discretizations enter it. This is what lets the method pass the patch test on a non-matching
interface exactly, which is verified in ``testfiles/edelweiss-only/MortarContactPatch*``.

The price is that every nodal quantity acquires a weight. Writing

.. math::

    D_{II} = \int_\Gamma \Phi_I \, \mathrm{d}\Gamma

for the *nodal mortar weight* -- the integral of node :math:`I`'s basis function over the part of
the surface actually covered by the opposite one -- the following hold:

* :math:`g_I / D_{II}` is the physical opening at the node, not :math:`g_I` itself
* :math:`\lambda_I \, D_{II}` is a nodal force, and :math:`\lambda_I` alone is a pressure only where
  :math:`D_{II}` is the node's full tributary area
* :math:`D_{II} = 0` means no material opposite the node at all, where neither a pressure nor an
  opening is defined

Keeping these apart is not pedantry: a node on the edge of the covered region can have a weight that
is a small fraction of its faces' area, and reading its multiplier as a pressure there reports a
value several times the true one while the force it transmits is perfectly ordinary. The deck
``testfiles/edelweiss-only/MortarContactPartialOverlap`` measures exactly this.


Segmentation of the overlap
---------------------------

The integral above is evaluated over the overlap of the two surfaces, which has to be constructed.
For each face of the non-mortar surface:

#. candidate faces of the mortar surface are found through a bounding-volume hierarchy, with a
   search margin that is the larger of a fraction of the local extent and an absolute length;
#. each candidate is projected into the tangent plane of the non-mortar face, which is the auxiliary
   plane the clipping happens in;
#. the two projected polygons are clipped against each other by the Sutherland--Hodgman algorithm;
#. the resulting polygon is fan-triangulated, and each triangle is integrated with a rule exact for
   the product of the two faces' shape functions.

Sutherland--Hodgman clips against the infinite half-plane of each clip edge in turn, which is exact
for a convex clip window and loses area for a non-convex one. Quadratic faces are therefore split
into convex sub-cells BEFORE clipping rather than clipped whole. Where a sub-cell degenerates -- a
sliver overlap along a shared edge, for instance -- the local mass matrix used for the dual basis
becomes ill-conditioned, and the implementation falls back to the reference-element dual
coefficients, reporting that it has done so. Biorthogonality then holds over the full face only, and
the nodal weight of such a node can take any sign.


The dual basis
--------------

The multiplier field is discretized in a basis :math:`\Phi` that is biorthogonal to the standard
basis :math:`\tilde N`:

.. math::

    \int_\Gamma \Phi_a \, \tilde N_b \, \mathrm{d}\Gamma = \delta_{ab} \int_\Gamma \tilde N_a \,
    \mathrm{d}\Gamma

This is what makes the coupling matrix of the non-mortar side diagonal, so that a nodal multiplier
belongs to exactly one node and the constraint stays local. Without it the multipliers of
neighbouring nodes are coupled and cannot be eliminated node by node. The dual coefficients are
computed per face from the element's own quadrature rather than taken from a reference table,
because they depend on the geometry of the face.

Biorthogonality alone is not enough: the weights :math:`D_{II}` must also be positive, and for the
serendipity and quadratic-triangle types they are not. The corner functions of an eight-node
quadrilateral integrate to :math:`-A/12` over the full face and those of a six-node triangle to
exactly zero. The implementation therefore applies a basis transformation that shifts a fraction
:math:`\alpha = 1/3` of each mid-side function onto its adjacent corners before building the dual
basis. Both integrals are measured against their closed form in
``edelweissfe/elements/contactelement/test_element.py``.

The transformation has a limit that matters in a segment-based method. It makes the full-face
integral positive, but at :math:`\alpha = 1/3` the transformed corner function is not pointwise
non-negative -- that would require :math:`\alpha \geq 3/8`. An overlap that falls entirely inside
the region where the function is negative can therefore still produce a negative nodal weight on the
consistent path. The nine-node quadrilateral is excluded from the transformation altogether, since
its corner integrals are already positive, and keeps no pointwise non-negativity either. This is why
the sign of the weight is carried explicitly through the active-set decision rather than assumed.


Nodal normals
-------------

The gap is measured along a normal defined at the nodes of the non-mortar surface rather than per
face, so that it varies continuously along a curved interface instead of jumping between faces. The
nodal normal is the normalized sum of the element normals of the adjacent faces, each evaluated at
that node's own natural coordinate and weighted by its magnitude, which for a surface element is its
local area measure.

The construction has a failure mode worth naming: across a sharp edge -- the ninety-degree step
between a shaft and a head, say -- it averages two normals that are ninety degrees apart and returns
one at forty-five degrees to both, which belongs to neither surface. A contact surface should not be
carried across such an edge in one piece.


The unilateral condition
------------------------

The Signorini conditions -- non-negative pressure, non-negative gap, and no pressure where there is a
gap -- are enforced through a single non-smooth complementarity function

.. math::

    C_{n,I} = p_I - \max\left(0,\; p_I - c_n \, g_I / D_{II}\right) = 0

so that node :math:`I` is active exactly where :math:`p_I - c_n \, g_I / D_{II} > 0`. A semi-smooth
Newton method treats this by re-deciding the active set in every iteration and converging the outer
loop once the set stops changing.

The penalty formulations need one safeguard the multiplier one does not. There the pressure is an
unknown of the system and a node that touches nothing solves to zero; in the penalty branch the
pressure is made out of the gap, so on a load-free interface -- two surfaces coinciding, nothing
applied yet -- a stiffness of 1e6 turns the rounding of the geometry into a pressure of 1e-10, and a
bare sign test reads that as contact. The weighted gap is therefore compared against a noise floor
of its own scale, the interface diameter times the largest nodal weight, taken relative by the same
factor the nodal weights themselves use. Below that floor there is no gap and no contact, and the
augmented outer loop has nothing to correct.

``cn`` is purely algorithmic: the gap vanishes at an active node on convergence, so the converged
solution is the same for every admissible value. Two properties of it are not obvious:

* its **unit** is a stress per length, since it multiplies an opening and is added to a pressure.
  A value that works for a model measured in metres is therefore wrong for the same model measured
  in millimetres.
* the admissible values form a **band**. Below it the active set never settles and the increment
  cuts back; far above it the set oscillates between two states forever. The order of the Young's
  modulus of the softer body sits inside that band for a model whose lengths are of order one, which
  is what the default derivation uses.

Two safeguards surround the iteration. The set is frozen for the remainder of an increment once it
has settled, and also once a discrete state already visited in that increment recurs -- a cycle can
otherwise repeat indefinitely without either converging or failing. Freezing by the iteration cap
rather than by convergence is reported, because the converged solution of such an increment is not
guaranteed to satisfy the conditions above.


The three formulations
----------------------

All three share the entire geometric pipeline -- search, projection, clipping, segmentation, nodal
normals and the coupling matrices -- and differ only in how the discrete condition is enforced, which
makes their results directly comparable.

``lagrange``
    The multipliers are unknowns of the global system, which becomes a saddle-point problem. The
    condition is satisfied exactly at convergence. Because the multipliers are scalar variables of
    the system, they appear in the reference solutions of the regression decks, so those decks
    check the contact pressures as well as the displacements.

``penalty``
    The multipliers are eliminated in favour of a pressure proportional to the weighted gap. Note
    the unit of that proportionality: the weighted gap carries length times area, so it is not the
    penalty parameter of a pointwise formulation and cannot be chosen by the same rules of thumb.
    The two are related through the nodal weight. The weighted form is used because it needs no
    division by that weight, which cannot be assumed to stay away from zero.

``penalty`` with ``augmentedLagrange``
    The pressure estimate is corrected in an outer loop that runs after each converged equilibrium
    and never inside it. This recovers most of the accuracy of the saddle-point form without its
    extra unknowns, and the converged result stops depending on the penalty stiffness, which then
    only sets how fast the outer loop converges. Measured on the quadratic patch test, the error
    falls from 1.5e-04 to 3e-11 at the same stiffness.

The multipliers of the saddle-point form are not condensed out of the system. A correct condensation
is a Schur complement of the already assembled non-mortar displacement row -- which carries the
surrounding bulk stiffness, not merely the constraint's own contribution -- and replaces that row
with the linearised constraint. It would require the constraint interface to replace rows rather
than only add to them.


The tangent
-----------

The geometry -- the segmentation, the coupling matrices and the nodal normals -- is evaluated once per
increment and held fixed for its Newton iterations, rather than recomputed per iteration. The
tangent is consistent with the residual that follows from that frozen geometry, which is what
governs how the iteration converges, and is verified against finite differences of the assembled
forces in ``edelweissfe/constraints/test_mortarcontact.py``. The terms omitted by the freezing are
the derivatives of the coupling matrices and the nodal normals with respect to the configuration;
they affect the rate of convergence rather than the converged answer, and the accuracy cost is the
one of evaluating the residual itself at the configuration of the previous increment.


Known limitations
-----------------

These are measured, not suspected.

Nodal weights at the edge of the covered region
    Where the mortar surface ends ON a face boundary of the non-mortar surface, the adjacent
    uncovered face touches it along a line. Clipping returns a sliver of near-zero rather than zero
    area, and the node acquires a tiny weight and a multiplier of order 1e5 whose last digits differ
    between nodes that symmetry says are identical. The nodal force that results is not negligible.
    Ending the mortar surface inside a face rather than on its boundary is well posed and is what
    ``testfiles/edelweiss-only/MortarContactPartialOverlap`` measures.

The stopping criterion of the augmented outer loop
    Convergence is measured as the change of the pressure estimate relative to that estimate's own
    magnitude. Where the penalty term alone already satisfies the condition to rounding, the
    estimate stops moving and both sides of the comparison fall to noise together, so the ratio
    never decreases and the loop always runs to ``maxAugmentations``. The result is correct; the
    report that it did not converge is not.

Length scale
    The residual of a multiplier row is the weighted gap, whose dimension is length times area,
    while its convergence bound is absolute rather than relative to a flux. Beyond roughly two
    decades above unit length that bound falls below the rounding noise of the gap and becomes
    unreachable no matter how exact the solution is; the run then exhausts its cutbacks. Raising
    ``fluxResidualTolerance`` for the scalar variables is the remedy.

Sharp edges
    A contact surface carried across a sharp edge produces nodal normals that belong to neither of
    the two faces meeting there. Such surfaces should be split.


Sources of the formulation
--------------------------

The formulation is not original to EdelweissFE. The sources below are the ones the implementation
follows; the code itself carries no citations, so that its comments stand on their own, and this is
where the attribution belongs. Each entry names what it underpins here.

*Dual Lagrange multipliers and their basis*

* B. I. Wohlmuth: *A mortar finite element method using dual spaces for the Lagrange multiplier*,
  SIAM Journal on Numerical Analysis 38 (2000), 989--1012. -- The dual space that keeps the
  multiplier field local.
* B. P. Lamichhane, R. P. Stevenson, B. I. Wohlmuth: *Higher order mortar finite element methods in
  3D with dual Lagrange multiplier bases*, Numerische Mathematik 102 (2005), 93--121.
* A. Popp, B. I. Wohlmuth, M. W. Gee, W. A. Wall: *Dual quadratic mortar finite element methods for
  3D finite deformation contact*, SIAM Journal on Scientific Computing 34 (2012), B421--B446. --
  The basis transformation for quadratic surfaces, and the reason the nine-node quadrilateral needs
  none.
* T. Cichosz, M. Bischoff: *Consistent treatment of boundaries with mortar contact formulations
  using dual Lagrange multipliers*, Computer Methods in Applied Mechanics and Engineering 200
  (2011), 1317--1332.

*Segment-to-segment contact*

* M. A. Puso, T. A. Laursen: *A mortar segment-to-segment contact method for large deformation solid
  mechanics*, Computer Methods in Applied Mechanics and Engineering 193 (2004), 601--629. -- The
  penalty form in terms of the weighted gap.
* M. A. Puso, T. A. Laursen, J. Solberg: *A segment-to-segment mortar contact method for quadratic
  elements and large deformations*, Computer Methods in Applied Mechanics and Engineering 197
  (2008), 555--566. -- The augmented outer loop, advanced only once equilibrium has converged.
* A. Popp, M. W. Gee, W. A. Wall: *A finite deformation mortar contact formulation using a
  primal-dual active set strategy*, International Journal for Numerical Methods in Engineering 79
  (2009), 1354--1391.
* A. Popp, M. Gitterle, M. W. Gee, W. A. Wall: *A dual mortar approach for 3D finite deformation
  contact with consistent linearization*, International Journal for Numerical Methods in Engineering
  83 (2010), 1428--1465. -- The averaged nodal normal.
* P. W. Farah: *Mortar Methods for Computational Contact Mechanics Including Wear and General Volume
  Coupled Problems*, Dissertation, TU München, 2018.
* P. Farah, A. Popp, W. A. Wall: *Segment-based vs. element-based integration for mortar methods in
  computational contact mechanics*, Computational Mechanics 55 (2015), 209--228. -- Why the overlap
  is segmented rather than integrated on the parent element.

*The semi-smooth treatment of the unilateral condition*

* T. De Luca, F. Facchinei, C. Kanzow: *A semismooth equation approach to the solution of nonlinear
  complementarity problems*, Mathematical Programming 75 (1996), 407--439.
* M. Hintermüller, K. Ito, K. Kunisch: *The primal-dual active set strategy as a semismooth Newton
  method*, SIAM Journal on Optimization 13 (2002), 865--888.
* S. Hüeber, B. I. Wohlmuth: *A primal-dual active set strategy for non-linear multibody contact
  problems*, Computer Methods in Applied Mechanics and Engineering 194 (2005), 3147--3166. -- The
  lower end of the admissible band for the complementarity parameter, and its linear dependence on
  the stiffness.
* M. Gitterle, A. Popp, M. W. Gee, W. A. Wall: *Finite deformation frictional mortar contact using a
  semi-smooth Newton method with consistent linearization*, International Journal for Numerical
  Methods in Engineering 84 (2010), 543--571. -- The upper end of that band, where the active set
  begins to chatter.

*Geometry and quadrature*

* I. E. Sutherland, G. W. Hodgman: *Reentrant polygon clipping*, Communications of the ACM 17
  (1974), 32--42. -- The clipping algorithm, including its restriction to convex clip windows.
* D. A. Dunavant: *High degree efficient symmetrical Gaussian quadrature rules for the triangle*,
  International Journal for Numerical Methods in Engineering 21 (1985), 1129--1148.
* C. Ericson: *Real-Time Collision Detection*, Morgan Kaufmann, 2004. -- The bounding-volume
  hierarchy of the candidate search.
