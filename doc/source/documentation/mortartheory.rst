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

The page is in three parts. The first states what is constrained and how that becomes a system of
equations. The second is about the geometry: how the two surfaces are turned into an integration
domain, in which basis the multiplier lives, and what has to be done differently where a surface
ends. The third is for using it -- the parameters, the place of the constraint inside the solver,
what is verified against what, and where the formulation is known to fall short.


The formulation
---------------

What the method constrains, how that becomes a system of equations, and what the tangent of
that system does and does not contain.


Why integrate rather than sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

for the *nodal mortar weight* -- the integral of node :math:`I`'s basis function over its own facets
-- the following hold:

* :math:`g_I / D_{II}` is the physical opening at the node, not :math:`g_I` itself
* :math:`\lambda_I \, D_{II}` is a nodal force, and :math:`\lambda_I` is the pressure
* :math:`D_{II} = 0` means no material opposite the node at all, where neither a pressure nor an
  opening is defined

Over its own facets, not over the covered part of them: that distinction is the subject of the
section on boundary rows below, and it is what keeps the second line true where coverage is partial.
How much of a node is actually opposed is reported separately, as its *coverage*, because it can no
longer be read off the weight. The deck ``testfiles/edelweiss-only/MortarContactPartialOverlap``
tabulates weight, coverage, multiplier and force side by side across such an interface.


The unilateral condition
~~~~~~~~~~~~~~~~~~~~~~~~

The Signorini conditions -- non-negative pressure, non-negative gap, and no pressure where there is a
gap -- are enforced through a single non-smooth complementarity function

.. math::

    C_{n,I} = p_I - \max\left(0,\; p_I - c_n \, g_I / D_{II}\right) = 0

so that node :math:`I` is active exactly where :math:`p_I - c_n \, g_I / D_{II} > 0`. A semi-smooth
Newton method treats this by re-deciding the active set in every iteration and converging the outer
loop once the set stops changing. Identifying the primal-dual active set strategy *as* a semi-smooth
Newton method, and with it the local superlinear convergence, is due to Hintermüller, Ito and
Kunisch; the complementarity function itself goes back to De Luca, Facchinei and Kanzow.

Which gap enters the indicator is a choice, and the one made here departs from the sources. Gitterle,
Popp, Gee and Wall write :math:`C_{n,j} = z_{n,j} - \max(0, z_{n,j} - c_n \tilde g_j)` with
:math:`\tilde g_j` the mortar-WEIGHTED gap, which carries length times area; above, the pointwise
opening :math:`g_I / D_{II}` is used instead, which carries a length. Both are admissible -- the
indicator only decides a branch, and either gap vanishes at convergence -- but only the length-valued
form makes :math:`c_n \, g_I / D_{II}` a pressure comparable to :math:`p_I`, and only for it is the
recommendation below on the order of :math:`c_n` dimensionally meaningful at all.

The penalty formulations need one safeguard the multiplier one does not. There the pressure is an
unknown of the system and a node that touches nothing solves to zero; in the penalty branch the
pressure is made out of the gap, so on a load-free interface -- two surfaces coinciding, nothing
applied yet -- a stiffness of 1e6 turns the rounding of the geometry into a pressure of 1e-10, and a
bare sign test reads that as contact. The weighted gap is therefore compared against a noise floor
of its own scale, the interface diameter times the largest nodal weight, taken relative by the same
factor the nodal weights themselves use. Below that floor there is no gap and no penetration: the
node is *closed*, it carries no pressure, and the augmented outer loop has nothing to correct.

**A closed node is an active node**, in both branches. The complementarity function has a kink where
its indicator vanishes -- :math:`p_I = 0` together with a gap that is zero up to the noise floor --
and a semi-smooth Newton method may use any element of the generalized derivative there:
Hintermüller, Ito and Kunisch (2002, Eq. 3.2, Lemma 3.1 and Remark 3.1) give the derivative of
:math:`\max(0, y)` as 1 for :math:`y > 0`, 0 for :math:`y < 0` and an arbitrary :math:`\delta` at
:math:`y = 0`; their own active-set strategy is the one with :math:`\delta = 0`. The choice is free
for the local convergence result but not in practice. Putting the kink on the inactive side removes every closed
node from the tangent, and a body carried by the contact alone then has a singular tangent whenever
its gap is exactly closed -- at a cold start from a touching configuration, and at the start of every
increment that follows a load-free one, which resolves an initial interference to exactly zero. The
iteration usually recovers, but not without damage: measured on the self-weight cases of the contact
study with C3D20R, the first Newton correction reached :math:`10^{10}`, and the cancellation that
followed left rounding of order :math:`10^{-5}` in the near-zero-energy modes of the
reduced-integrated elements -- a pressure scatter of 2 to 7 % on problems whose exact pressure is
uniform, in the penalty branch AND the multiplier branch. With the kink on the active side the same
runs are exact to round-off (below :math:`10^{-8}`) and take two Newton iterations per increment. The
pressure is unaffected by the choice: a closed node keeps its spring in the tangent, but its pressure
stays :math:`\max(0, p_\text{trial})`, and it is released as soon as the iteration opens it beyond
the floor or, in the multiplier branch, produces a tensile multiplier.

``cn`` is purely algorithmic: the gap vanishes at an active node on convergence, so the converged
solution is the same for every admissible value. Two properties of it are not obvious:

* its **unit** is a stress per length, since it multiplies an opening and is added to a pressure.
  A value that works for a model measured in metres is therefore wrong for the same model measured
  in millimetres.
* the admissible values form a **band**. Below it the active set never settles and the increment
  cuts back; far above it the set oscillates between two states forever. The order of the Young's
  modulus of the softer body sits inside that band for a model whose lengths are of order one, which
  is what the default derivation uses.

On where that band comes from, precisely, since neither source covers this case as it stands. That
the parameter cannot affect the answer is Gitterle, Popp, Gee and Wall, without qualification. The
lower end and its proportionality to the stiffness are Hüeber and Wohlmuth, who report it as an
observation rather than a result -- "it seems to be that :math:`c_0` depends linearly on :math:`E`" --
and, more importantly, report it for their INEXACT strategy, which updates the active set after each
multigrid sweep. They state that the parameter has no influence at all when the linear problems are
solved exactly, which is what EdelweissFE does. What makes the value matter here is something their
quasi-linear setting does not have: the set is re-decided per Newton iteration on a geometry that
changes from increment to increment. The band is real and measured, but it is measured here rather
than inherited.

Two safeguards surround the iteration. The set is frozen for the remainder of an increment once it
has settled, and also once a discrete state already visited in that increment recurs -- a cycle can
otherwise repeat indefinitely without either converging or failing. Freezing by the iteration cap
rather than by convergence is reported, because the converged solution of such an increment is not
guaranteed to satisfy the conditions above.


The three formulations
~~~~~~~~~~~~~~~~~~~~~~

All three share the entire geometric pipeline -- search, projection, clipping, segmentation, nodal
normals and the coupling matrices -- and differ only in how the discrete condition is enforced, which
makes their results directly comparable.

``lagrange``
    The multipliers are unknowns of the global system, which becomes a saddle-point problem. The
    condition is satisfied exactly at convergence. Because the multipliers are scalar variables of
    the system, they appear in the reference solutions of the regression decks, so those decks
    check the contact pressures as well as the displacements.

``penalty``
    The multipliers are eliminated in favour of a pressure proportional to the penetration. The
    weighted gap carries length times area, so the stiffness that multiplies it is divided by the
    nodal weight, node by node: the pressure is then :math:`\varepsilon_N` times the pointwise
    penetration, and ``penaltyStiffness`` is the same pointwise quantity, in the same unit, as the
    ``penalty`` of the node- and segment-based penalty constraints, chosen by the same rules of
    thumb. The division is applied to the stiffness rather than to the gap, because the weight
    cannot be assumed to stay away from zero; see *The law is a pointwise one* below.

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
~~~~~~~~~~~

How the geometry -- the segmentation, the coupling matrices and the nodal normals -- enters the
residual and the tangent is set by ``geometryUpdate``.

**consistent**, the default under the implicit solvers. The geometry is rebuilt at every Newton
iterate, at the iterate itself, and the tangent carries its full linearisation, as in Popp,
Gitterle, Gee and Wall (2010), Section 4.3 and Appendix A, and in two dimensions Popp, Gee and Wall
(2009), Appendix A. Every quantity D, C and the weighted gap are built from is differentiated with
respect to the nodal coordinates, and the derivatives are chained down
(:mod:`~edelweissfe.utils.mortarlinearization`):

* the element normal at each node and their unweighted average, Eqs. (A1)--(A3) of Popp et al.
  (2010);
* the auxiliary plane and the orthogonal projection of the nodes into it, Eq. (A17);
* the vertices of the clipped polygon, each either a surviving subject vertex or the intersection of
  a subject edge with the line of a clip edge, Eq. (A19);
* the fan triangles, their Jacobian, Eq. (A7), and the Gauss points in them, Eq. (A16);
* the natural coordinates of every Gauss point on both sides, Eqs. (A12)--(A14);
* the dual coefficients, Eq. (A23), with the basis transformation constant -- it is "defined once
  at t = 0" and stays "unaltered during the entire finite deformation process" (Popp, Wohlmuth, Gee
  and Wall 2012, p. B432) -- and the whole-facet integrals through the surface Jacobian, Eqs.
  (A24)--(A26);
* D and C, Eqs. (A4)--(A5), the weighted gap, Eqs. (A8)--(A11), and the derivative of the contact
  forces, the matrix of their Eq. (51).

Three steps of the geometry built here differ from the reference, and their derivatives follow from
the chain rule rather than from the paper: the auxiliary plane belongs to a slave sub-cell and takes
that sub-cell's own normal instead of the interpolated nodal normal of Eq. (A15); a Gauss point is
mapped back by inverting the in-plane isoparametric map (three dimensions) or by a closest-point
projection (two dimensions) instead of along the auxiliary normal, Eq. (A12); and the dual
coefficients carry the row weighting of Cichosz and Bischoff. The in-plane basis of the auxiliary
plane is varied without rotation about its normal, which no quantity leaving the plane depends on.
As in the reference, a derivative holds for a fixed layout: which vertices the polygon has, how it
is triangulated, which facet a Gauss point is attributed to and whether a facet takes the
reference-element fallback is decided at each iterate and not differentiated. An entry of D or C
below the noise floor keeps its column and its derivative with the value exactly zero: a coupling
that vanishes because the two meshes coincide along one direction does not vanish in the
neighbouring configuration, and its derivative is not zero.

The residual is the same as under ``frozen``: the contact forces are :math:`-\lambda_I D_{IK} n_I`
and :math:`+\lambda_I C_{IJ} n_I`, and the variation of D and C in the virtual work is neglected, as
in Popp et al. (2010), Section 7. The tangent is the exact derivative of this residual, and it is
non-symmetric, under ``penalty`` as well.

Measured against central differences of the residual with the geometry rebuilt at every
perturbation, on a tilted, curved, laterally shifted pair of surfaces, for all seven contact element
types, both formulations and friction in stick and in slip: :math:`\max |K - K_{FD}| / \max |K|`
between :math:`10^{-10}` and :math:`1.1 \cdot 10^{-7}`. Without the geometric terms the same
comparison gives :math:`5.3 \cdot 10^{-2}` under ``penalty`` and 1.1 under ``lagrange`` (CONQUAD4).
Once the active set has settled the Newton iteration converges quadratically; on
``MortarContactTilted`` the residual of the first increment runs 1.4, :math:`6.3 \cdot 10^{-3}`,
:math:`2.1 \cdot 10^{-7}`, :math:`1.3 \cdot 10^{-15}`.

**frozen**. The geometry is built once per increment from the last converged configuration and held
fixed for that increment's Newton iterations. The tangent omits the derivatives of D, C and the
normals and is exact for the residual with the geometry frozen: against central differences with
the geometry frozen, :math:`\max |K - K_{FD}| / \max |K| = 5.3 \cdot 10^{-10}`. The residual,
however, belongs to the configuration at the start of the increment, which makes the scheme
staggered and first order in the increment size -- an error of the increment, not of the iteration,
which shrinks with the step and not with the tolerance. Measured as the largest deviation of the
final state from the consistent one at the smallest increment, relative to its largest entry:

====================================  ===========  ===========  ===========  ===========  ===========
maxInc                                0.5          0.25         0.125        0.0625       0.03125
====================================  ===========  ===========  ===========  ===========  ===========
``MortarContactTilted``, frozen       1.4e-2       5.7e-3       2.5e-3       1.2e-3       5.7e-4
``MortarContactPartialOverlap``,      1.6e-2       9.2e-3       4.3e-3       2.1e-3       1.0e-3
frozen
both decks, consistent                < 4e-13      < 4e-13      < 4e-13      < 4e-13      0
====================================  ===========  ===========  ===========  ===========  ===========

``frozen`` is not the *incomplete linearisation* Popp et al. (2010) measure in their Table I -- a
live geometry carried with a tangent that omits the derivatives of the nodal normal and of the two
coupling matrices, which costs them 52 Newton steps against 16 with the full linearisation under the
same fixed-point active-set strategy (8 with the semi-smooth one). Here the tangent is exact for the
residual it belongs to; it is the residual that lags.


Friction
~~~~~~~~

With ``mu > 0`` the ``penalty`` formulation adds Coulomb friction, regularised by a tangential
penalty in the same way the normal condition is. The law -- elastic stick predictor, radial return
onto the cone, consistent tangent, history, anti-chattering freeze, ``slipTolerance`` -- is the one
all contact formulations of EdelweissFE share (:mod:`~edelweissfe.utils.contactfriction`, described in
the *Coulomb friction* section of :doc:`contacttheory`): the penalty-regularised law with the
traction as history and the radial return of Simo and Laursen (1992), Eqs. (3.9) and
(3.12)--(3.14), and Wriggers (2006), Eqs. (10.121)--(10.126) and Box 14; in the mortar setting it is
the law of Yang, Laursen and Meng (2005), their Eqs. (39) and (47)--(49), in the three-dimensional
form of Puso, Laursen and Solberg (2008), their Eqs. (16)--(21). It needs no unknowns
of its own and is therefore available under the implicit and the explicit solvers alike;
``formulation=lagrange`` with ``mu > 0`` is refused. What is mortar-specific is how a non-mortar
node maps onto that law, and the objective slip correction below.

The weighted slip increment is built from the same weights as the weighted gap, :math:`w_a = D_{IK}`
on the non-mortar and :math:`w_a = -C_{IJ}` on the mortar nodes, projected onto the tangent plane
:math:`P_T = I - n_I \otimes n_I` of the nodal normal. Divided by :math:`|D_{II}|` it is the
pointwise slip of the node, so the node enters the shared law as a contact point with

.. math::

   c = \frac{w}{|D_{II}|}, \qquad A = |D_{II}|, \qquad
   \Delta u_T = P_T \sum_a c_a \, \Delta u_a = \frac{s_I}{|D_{II}|},

   t^{tr}_I = P_T \, t_{I,n} - \varepsilon_T \, \Delta u_T ,\qquad
   t_I = \begin{cases}
   t^{tr}_I & |t^{tr}_I| \le \mu p_I \quad \text{(stick)} \\
   \mu p_I \, t^{tr}_I / |t^{tr}_I| & \text{otherwise (slip)}
   \end{cases}

:math:`\Delta u` is the increment since the last converged state under the implicit solvers and one
time step's displacement under the explicit one. That is the slip measure of the frozen geometry; the
consistent one replaces it by the objective slip below. :math:`\varepsilon_T` (``tangentPenalty``, by
default the resolved :math:`\varepsilon_N`) is a pointwise modulus exactly like
:math:`\varepsilon_N`: a sticking node carries :math:`\varepsilon_T` times the relative tangential
displacement. The nodal forces :math:`A\,c_a t_I = w_a t_I` are the gradient of the convex stick
energy, so the sign of :math:`D_{II}` needs no special treatment: only its magnitude enters. A node
counts as without pressure -- no friction, no tangent -- within the noise floor of the weighted gap.
The characteristic facet size of ``slipTolerance`` is taken from the mortar weights at the first
geometry rebuild, :math:`h = (\sum_I |D_{II}| / n_{facets})^{1/(d-1)}`.

Under ``geometryUpdate=consistent`` the slip is the objective one of Gitterle, Popp, Gee and Wall
(2010), Eq. (53), at the iterate,

.. math::

   \Delta u_T = P_T \, \frac{\big[ -(D - D_n)\, x_s + (C - C_n)\, x_m \big]_I}{|D_{II}|} ,

with :math:`D_n, C_n` the coupling matrices of the last converged configuration and everything else
at the iterate; it is the measure of Yang et al., Eq. (47), as well. A rigid rotation leaves D and C
unchanged and so produces no slip, and the history needs no correction between increments. Gitterle
et al. treat two dimensions; in three the slip is projected onto the tangent plane, as the weighted
relative tangential velocity of Farah (2018), Eq. (3.43). Its linearisation is their Eq. (A3) with
the normal, the coupling matrices and the pressure all differentiated, and it is the chain rule of
this implementation for the tangent plane in three dimensions.

Under ``geometryUpdate=frozen`` the slip measure within an increment is the frozen-geometry one,
:math:`D \Delta u_s - C \Delta u_m`, which keeps the tangent exact. On its own it is not objective: a rigid rotation :math:`\theta` of a
penetrating pair turns the weighted gap vector :math:`g_I = \sum_K D_{IK} x_K - \sum_J C_{IJ} x_J`
and reads as a slip of about :math:`\theta` times the penalty gap, which in stick would accumulate
over the increments. The objective measure of Yang et al., Eq. (47), and of Gitterle et al. (2010),
Eq. (53), :math:`-(D_{n+1} - D_n) x_s + (C_{n+1} - C_n) x_m`, differs from the frozen one by exactly
the change of :math:`g_I` (Yang et al., Eqs. (45)--(46); Gitterle et al., Eqs. (50)--(51)), and the coupling matrices of the end of an increment are what the next
geometry rebuild computes. The accepted traction is therefore corrected there (the optional gap
correction of the shared history transport),

.. math::

   t_{I,n} \leftarrow P_T \Big( t_{I,n} + \varepsilon_T \Big( \frac{g_I^{new}}{|D_{II}^{new}|}
   - \frac{g_I^{old}}{|D_{II}^{old}|} \Big) \Big),

and returned onto the cone of the accepted pressure. Integrated over any number of increments the
slip is then the objective one; what remains is the error of the frozen normal, first order in one
increment and not accumulated. A pure slide of a flat pair leaves :math:`g_I` unchanged and is not
corrected. Measured on two blocks rotated rigidly in stick by 0.01 per increment, the frozen measure
alone leaves :math:`0.7\,p\,\varepsilon_T/\varepsilon_N` after seven increments, the corrected one
never more than the single increment's :math:`0.1`. The node-to-surface and integrated formulations
need no such correction: their reprojection onto the closest point is itself the objective frame.

With a constant modulus the elastic range :math:`\mu p / \varepsilon_T` is, on a lightly loaded
interface, far below one Newton correction (about 1e-7 mm against 1e-5 mm in the first increments of
the POT pull-out), so that the stick/slip state is decided by noise; ``slipTolerance`` (0.005 in the
POT decks) is the remedy there.

The nodal results -- pressure, frictional traction, stick/slip state and opening -- are available
through ``getNormalPressures``, ``getTangentialTractions``, ``getSlipStates`` and ``getGaps``,
ordered like the node set ``<name>_nonMortarNodes`` that the constraint registers, so that they can
be written as ``fromExpression`` field outputs and exported to Ensight per node.

Under explicit dynamics the stick spring enters the stability bound alongside the normal one,
row by row, :math:`\omega^2 \le \max_a (\varepsilon_N r_{N,a} + \varepsilon_T r_{T,a})`, and a
derived :math:`\varepsilon_N` is chosen so that the two together stay at half the limit.


Surface discretisation and integration
--------------------------------------

How the two surfaces are turned into an integration domain: what is clipped against what, in
which basis the multiplier lives, along which direction the gap is measured, and what has to be
done differently where a surface ends.


Segmentation of the overlap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
for a convex clip window and loses area for a non-convex one. The restriction is on the WINDOW
alone; the subject polygon may be anything, since the algorithm as published is "applicable to any
polygon, convex or concave, planar or not". (The word *reentrant* in that paper's title describes
the reentrant CODE -- one clipping stage called repeatedly, once per clip edge -- and not a reentrant
polygon, which is the opposite of what the title suggests on first reading.) Quadratic faces are
therefore split into convex sub-cells BEFORE clipping rather than clipped whole. Their Appendix B
offers the other way out, decomposing a concave polygon into convex pieces; that is deliberately not
implemented here, because a contact facet that comes out concave in the auxiliary plane is a mesh
problem worth reporting rather than a case worth accommodating. Where a sub-cell degenerates -- a
sliver overlap along a shared edge, for instance -- the local mass matrix used for the dual basis
becomes ill-conditioned, and the implementation falls back to the reference-element dual
coefficients, reporting that it has done so. Biorthogonality then holds over the full face only, and
the nodal weight of such a node can take any sign.


The dual basis
~~~~~~~~~~~~~~

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
exactly zero. The implementation therefore applies a basis transformation before building the dual
basis, which shifts a fraction :math:`\alpha` of each mid-side function onto the two corners
adjacent to it:

.. math::

    \tilde N_c = N_c + \alpha \left( N_{m_1} + N_{m_2} \right) , \qquad
    \tilde N_m = \left( 1 - 2\alpha \right) N_m , \qquad
    \alpha = \tfrac{1}{3} ,

where :math:`m_1, m_2` are the mid-side nodes of the two edges meeting at corner :math:`c`. As a
matrix this is column-stochastic -- every column sums to one -- which is what preserves the partition
of unity: the transformed functions still reproduce a constant exactly, so the method still
transfers a constant pressure exactly. What changes is only how the total is distributed between
corner and mid-side nodes.

The transformation is that of Popp, Wohlmuth, Gee and Wall; the value :math:`\alpha = 1/3` is
Farah's. This is worth stating plainly, because the two sources give the same transformation with
DIFFERENT values, and a reader who checks only the first will find :math:`1/5`. Both derive the same
admissibility requirement -- integral positivity needs :math:`\alpha > 0` for a six-node triangle and
:math:`\alpha > 1/8` for an eight-node quadrilateral, and :math:`\alpha < 1/2` keeps the mid-side
integrals positive -- and both values satisfy it.

What decides between them here is that this method integrates over PARTS of a face. Recomputed for
the eight-node corner function, :math:`\int \tilde N_c \,\mathrm{d}\Gamma = 8\alpha/3 - 1/3`, which
is indeed positive from :math:`\alpha > 1/8`; but the pointwise minimum of that function over the
reference square is :math:`-0.0706` at :math:`\alpha = 1/5` against :math:`-1/324 = -0.00309` at
:math:`\alpha = 1/3`. Since it is an overlap confined to the region where the function is negative
that turns a nodal weight negative, the shallower dip is the safer choice, and :math:`1/3` is kept
for that reason rather than by inheritance.

Measured on a straight-edged element of area :math:`A`, the corner integral
:math:`\int \tilde N_c \, \mathrm{d}\Gamma`:

=============  =======================  =====================
Element        untransformed            transformed
=============  =======================  =====================
``CONQUAD8``   :math:`-A/12`            :math:`+5A/36`
``CONTRI6``    exactly :math:`0`        :math:`+2A/9`
``CONLINE3``   :math:`+A/6`             :math:`+7A/18`
=============  =======================  =====================

``CONLINE3`` is in the table although its untransformed integral is already positive: there the
transformation is applied for the stronger property described below, not to repair the sign. Both
columns are checked against their closed form in
``tests/test_contactelement.py``.

The transformation has a limit that matters in a segment-based method. It makes the full-face
integral positive, but at :math:`\alpha = 1/3` the transformed corner function is not pointwise
non-negative -- that would require :math:`\alpha \geq 3/8`. An overlap that falls entirely inside
the region where the function is negative can therefore still produce a negative nodal weight on the
consistent path. The nine-node quadrilateral is excluded from the transformation altogether, since
its corner integrals are already positive, and keeps no pointwise non-negativity either. This is why
the sign of the weight is carried explicitly through the active-set decision rather than assumed.


Nodal normals
~~~~~~~~~~~~~

The gap is measured along a normal defined at the nodes of the non-mortar surface rather than per
face, so that it varies continuously along a curved interface instead of jumping between faces:

.. math::

    n_I = \frac{\sum_{e \ni I} \; n_I^e}
               {\left\| \sum_{e \ni I} \; n_I^e \right\|} ,
    \qquad
    n_I^e = \frac{x_{,\xi}(\xi_I^e) \times x_{,\eta}(\xi_I^e)}
                 {\left\| x_{,\xi}(\xi_I^e) \times x_{,\eta}(\xi_I^e) \right\|}

summed over the faces :math:`e` adjacent to node :math:`I`, each cross product evaluated at that
node's OWN natural coordinate :math:`\xi_I^e` within the face. Note the two normalisations. Each
face's contribution is brought to unit length BEFORE the sum, so that the faces enter with equal
weight, and the sum is normalised again at the end, which is what makes the field continuous across
face boundaries. This is the averaged nodal normal of Popp, Gitterle, Gee and Wall, written out as
Eq. (4.41) of Farah, who likewise sums the *unit* normals of the adjacent elements.

Summing the un-normalised cross products instead would weight each face by its surface Jacobian, and
that is measurably worse rather than better: on a graded cylinder patch with a neighbour size ratio
of two, the nodes with unequal neighbours come out at 3.82 degrees weighted against 1.91 degrees
unweighted for ``CONQUAD4``, and 0.0211 against 0.0147 degrees for ``CONQUAD8``/``CONQUAD9``. The
larger face averages over a wider arc and is therefore the poorer estimate of the normal, so giving
it more weight pulls the result away from the exact one. The weighting would also depend on the
element type rather than on the geometry, since the surface Jacobian carries the size of the
reference element: a triangle and a quadrilateral of EQUAL area contribute in the ratio 8:1.

All these variants coincide wherever the contributions at a node are parallel -- a flat non-mortar
surface -- or of equal length and symmetric about the node, since any positive weighting then
normalises to the same direction. The difference appears only on a curved surface whose faces differ
in size.

Evaluating at the node's own coordinate rather than at the face centre is what ties the normal to
the geometry at that node, and it is what the quadratic element types buy. Measured against the
analytical normal of a cylindrical patch, halving the facet size takes the angular error from
0.0278 to 0.0035 degrees for the quadratic types -- a factor of 7.95, i.e. third order -- while a
single facet-constant normal leaves them at the linear types' own 7.1620 to 3.5810 degrees, a
factor of 2. A facet-constant normal assigns one direction to a corner and to a mid-side node of
the same face, so the curvature of that face never reaches the normal at all.

The construction has a failure mode worth naming: across a sharp edge -- the ninety-degree step
between a shaft and a head, say -- it averages two normals that are ninety degrees apart and returns
one at forty-five degrees to both, which belongs to neither surface. A contact surface should not be
carried across such an edge in one piece.


Boundary rows
~~~~~~~~~~~~~

A non-mortar facet that the other surface covers only in part integrates its basis functions over
the covered piece alone. Its nodal weight then decreases with the SQUARE of the covered area, since
both the domain of integration and the integrand shrink with it, while the force that has to cross
there decreases only linearly -- the master's force is distributed to the nodes of the boundary facet
by a lever arm which shortens at the same rate. The multiplier, being the ratio of the two, grows as
the reciprocal of the covered area. Those values are assembled into the stiffness matrix, whose
condition number degrades with them, and with it the convergence of the Newton iteration; in the
worst case it does not converge at all.

The remedy is due to Cichosz and Bischoff, whose own diagnosis this is. It has two parts, their
Eqs. (41) and (43) to (45), and both are implemented here:

#. the dual basis of such a facet is built by imposing biorthogonality over the COVERED area rather
   than over the whole facet -- "the underlying integrals now have to be evaluated on the contact
   area instead of the entire slave element" -- which is what restores the row-sum identity between
   the two coupling matrices at the boundary; without it the two are integrated over different
   domains and the identity, hence the invariance of the weighted gap under a rigid translation, is
   lost;
#. each row is then weighted by the reciprocal of its own covered integral, so that the weighted
   nodal weight is the whole-facet integral again.

Both coupling matrices carry the same factor, so the row-sum identity survives and with it the
normalised gap :math:`g_I / D_{II}`, whose numerator and denominator are scaled alike. The
multiplier is rescaled inversely and becomes a pressure of ordinary magnitude. No case distinction
is needed either, since on a fully covered facet the covered and the whole integral coincide and the
factor is exactly one -- which is their point as well: once the weighting is in place, "there is no
difference between slave mortar integrals in inner and boundary elements", and the separate
treatment of a boundary element that the second part would otherwise need "can be skipped".

EdelweissFE generalises the weighting in one respect. Cichosz and Bischoff define the weighting factor
parametrically, as the reciprocal of :math:`\int N_k \,\mathrm{d}\xi` over the covered interval,
which for their straight two-node elements returns the row to the whole-element value. The ratio of
the two PHYSICAL integrals used here lands on the same value for those elements and extends to a
curved facet, whose Jacobian does not cancel out of a parametric expression.

For a fixed active set the weighting is a rescaling of the constraint and the displacement solution
is identical. Across the active-set decision it is not. The activation test compares the pressure
against :math:`c_n` times the normalised gap; the gap is invariant under the rescaling and the
pressure is not, so a node close to the threshold can switch differently. That is intended: without
the rescaling the quantity entering that comparison at a thinly covered node would be a force
divided by a vanishing area, set against a genuine pressure on the other side of the test. Measured on a curved indenter, three nodes at the edge of the contact zone move,
by at most 3.4 % of the peak pressure, while the agreement with the analytical Hertz solution stays
at 1.0 %.

Measured on a facet covered to :math:`8 \cdot 10^{-6}` of its area, the multiplier falls from
:math:`2.1 \cdot 10^{5}` to 10.8, against a contact pressure of 10 elsewhere on the same interface.

The weighting is applied only where the covered integral is positive. A quadratic facet covered in
the region where its transformed corner function is negative does not provide a measure to divide
by, and the factor would there flip the sign of the row -- an equivalent constraint, but one that
silently redefines what the sign of the multiplier means, which the active-set indicator reads.
Those rows are left unweighted and keep the sign handling they were written for.


The choice of sides
~~~~~~~~~~~~~~~~~~~

Which of the two surfaces is made the non-mortar one is not a matter of taste where the surfaces
differ in extent. Putting the SMALLER one there makes every one of its facets fully covered, and no
boundary row arises at all. The same model with the sides exchanged produces them in numbers.

This is not one of the two remedies below but something better: it is the case in which they are
not needed at all, and it is the one to reach for whenever the geometry allows it. Cichosz and
Bischoff establish it by example rather than by derivation -- their Section 6.1 punches a small
block onto a larger one, runs the same model twice with the sides exchanged, and compares the
contact pressure and the condition number of the stiffness matrix. The two decks
``MortarContactPunchEdge`` and ``MortarContactPartialOverlap`` are the same configuration with the
two choices, and their headers carry the measurements.


The discrete system, step by step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above assembles into two sparse matrices and one scalar unknown per non-mortar node. It
is worth writing the result down, because the shape of it explains where each of the quantities
discussed so far ends up.

Let :math:`K` run over the nodes of the non-mortar surface, :math:`J` over those of the mortar
surface, and :math:`n_I` be the nodal normal of non-mortar node :math:`I`. The two coupling
matrices are integrals over the overlap of the two surfaces,

.. math::

    D_{IK} = \int_\Gamma \Phi_I \, N_K \, \mathrm{d}\Gamma , \qquad
    C_{IJ} = \int_\Gamma \Phi_I \, N_J \, \mathrm{d}\Gamma ,

with :math:`\Phi_I` the dual basis function of node :math:`I` and :math:`N` the standard shape
functions of the respective surface. Note that :math:`D` is not diagonal for quadratic elements:
biorthogonality holds against the TRANSFORMED basis, and lumping :math:`D` to its diagonal would
destroy the exact transfer of a constant pressure. What is diagonal in the relevant sense is the
row sum,

.. math::

    D_{II}^{\text{row}} = \sum_K D_{IK} = \int_\Gamma \Phi_I \, \mathrm{d}\Gamma ,

which is the nodal weight referred to throughout. The weighted gap of node :math:`I` follows from
the current positions :math:`x` as

.. math::

    g_I = -\sum_K D_{IK} \, (x_K \cdot n_I) + \sum_J C_{IJ} \, (x_J \cdot n_I) .

Its translational invariance rests on a single identity, :math:`\sum_K D_{IK} = \sum_J C_{IJ}`,
which holds because both are integrated over the SAME domain -- the overlap, not the respective
facets. Translating both bodies by the same vector leaves :math:`g_I` unchanged only because of it,
and it is checked directly in the unit tests.

Collecting the weights of one node as :math:`w_a = +D_{IK}` on the non-mortar nodes and
:math:`w_a = -C_{IJ}` on the mortar ones, and writing :math:`v_a = w_a n_I`, the whole nodal
contribution is rank one:

.. math::

    g_I = -\sum_a v_a \cdot x_a , \qquad
    f_a = -\lambda_I \, v_a .

**Saddle-point form.** With ``formulation=lagrange`` the multipliers :math:`\lambda_I` are unknowns
of the global system, one scalar variable per non-mortar node. An active node contributes the
constraint row :math:`g_I = 0` and its transpose, an inactive one the trivial row
:math:`\lambda_I = 0`:

.. math::

    \begin{bmatrix} K_{uu} & B^{T} \\ B & 0 \end{bmatrix}
    \begin{bmatrix} \Delta u \\ \lambda \end{bmatrix}
    = \begin{bmatrix} r_u \\ -g \end{bmatrix} ,
    \qquad B_{Ia} = v_a ,

for the active rows. The off-diagonal blocks are each other's transpose, which is what makes the
assembled system symmetric; it is indefinite, so the linear solver has to cope with a saddle point.

Because the multipliers are ordinary scalar variables of the model, they appear in the state vector
the regression runner compares against its reference solution. The reference of a
``formulation=lagrange`` deck therefore holds the nodal contact pressures as well as the
displacements, and a pressure distribution that changed would fail the comparison.

**Penalty form.** With ``formulation=penalty`` no unknown is added. The pressure is a function of
the current state,

.. math::

    \lambda_I = \max\left(0, \; z_I + \kappa_I \, g^{\text{pen}}_I\right) \operatorname{sgn} D_{II} ,
    \qquad g^{\text{pen}}_I = -g_I \operatorname{sgn} D_{II} ,
    \qquad \kappa_I = \frac{\varepsilon_N}{|D_{II}|} ,

with :math:`z_I` the augmentation estimate, zero for pure penalty. The tangent is then the rank-one
:math:`K_{ab} \mathrel{+}= \kappa_I \, v_a v_b`, symmetric and positive semi-definite, and the system
keeps the size and the definiteness it had without contact.

**The law is a pointwise one.** Since :math:`g^{\text{pen}}_I = |D_{II}| \, \delta_I` with
:math:`\delta_I` the pointwise penetration, the law above is exactly
:math:`\lambda_I = \varepsilon_N \delta_I`: a pressure per unit opening, the same quantity and the
same unit as the ``penalty`` of :doc:`the node- and segment-based penalty constraints
</documentation/contacttheory>`, so a value can be carried between them.

**Why the stiffness is node-wise.** A single stiffness multiplying the weighted gap would give
every node the pointwise stiffness :math:`\varepsilon_I = \kappa D_{II}`, i.e. a stiffness
proportional to its own tributary area -- a factor of four between a corner node and an interior one
on a plain :math:`4\times4` conforming interface. A uniform pressure could then not produce a
uniform penetration, the interface would not stay plane, and the contact patch test would fail at
every finite stiffness, on a conforming mesh. Dividing by :math:`D_{II}` removes exactly that: the
penetration is :math:`p/\varepsilon_N` at every node, so the pressure a uniform gap produces is
uniform. Measured on the two-block patch test, the relative pressure spread is
:math:`1.7\cdot10^{-12}` on a *non-conforming* mesh, with a pure penalty and no augmentation.

The scale factor is the :math:`\kappa_A` of Yang, Laursen & Meng (2005), their Eqs. (36) and (37),
:math:`1/\sum_D n_{AD}`, the row sum of the slave mortar matrix in the reference configuration,
introduced there, in their words, "to cause the gap function :math:`g_A` to have the proper units
of length, which is of crucial importance when implementing penalty methods in particular". Here it
is the nodal weight :math:`|D_{II}|` of the current geometry rebuild, which is that row sum for the
dual basis. Their Section 8.1 passes the patch test on non-conforming meshes to machine precision
with a penalty (in two dimensions).

**Why only the stiffness is divided, not the gap.** :math:`D_{II}` cannot be assumed to stay away
from zero -- a partially covered CONQUAD9, a CONQUAD8 corner at :math:`\alpha = 1/3` and the sliver
fallback all produce negative or near-zero weights. The *gap* is therefore carried in its weighted
form everywhere, and the division appears only in the *stiffness*, where it can be floored: a node
whose weight falls below the relative floor of :attr:`currentWeightTolerance` receives no penalty
spring at all rather than an unbounded one, which is also the physically right answer, since a node
with no coverage transmits no force. Only the magnitude of the weight enters
:math:`\kappa_I`; its sign is carried by :math:`g^{\text{pen}}_I` and by :math:`\lambda_I`.


Use and verification
--------------------

What a user has to choose, how the constraint sits inside the solver, what is checked against
what, and where the formulation is known to fall short.


Choosing the parameters
~~~~~~~~~~~~~~~~~~~~~~~

* ``formulation`` is the first choice and the one with the largest consequences. ``lagrange``
  satisfies the condition exactly and is the reference; its cost is a saddle-point system with one
  extra unknown per non-mortar node. ``penalty`` keeps the system size and definiteness but
  satisfies the condition only approximately: measured on the quadratic patch test, the error in
  the displacement field is 1.5e-04 at :math:`\varepsilon_N = 10^6` and falls by exactly one decade
  per decade of stiffness -- 1.5e-05, then 1.5e-06. That error is pure interface compliance and
  uniform over the interface; the contact *pressure* stays constant, which is what makes the patch
  test pass at any finite stiffness. Switching ``augmentedLagrange`` on at the same stiffness
  brings the displacement error to 3e-11, four orders better, and removes the dependence on the
  stiffness altogether. Pure penalty is a reasonable choice where the compliance is acceptable,
  and it is the only one an explicit solver can integrate.

* ``cn`` enters the activation test only. It is purely algorithmic -- the opening vanishes at an
  active node on convergence, so the converged solution is the same for every admissible value --
  but two of its properties are easy to get wrong. Its UNIT is a stress per length, because it
  multiplies an opening and is added to a pressure; a value that suits a model measured in metres
  is wrong for the same model in millimetres. And the admissible values form a band whose ends fail
  differently: below it the active set never settles and the increment cuts back, far above it the
  set oscillates between two states indefinitely. Left at its default of 0 it is taken as the
  smallest Young's modulus among the materials adjacent to the two surfaces, which sits inside that
  band for a model whose lengths are of order one, and the derived value is reported once.

* ``penaltyStiffness`` is :math:`\varepsilon_N`, a pointwise contact stiffness per unit area -- a
  pressure per unit opening. It is the same quantity as the ``penalty`` of
  ``nodeToDeformableSurfacePenalty`` and ``surfaceToDeformableSurfacePenalty``, so a value may be
  carried between them and a rule of thumb for a classical penalty parameter applies unchanged.
  Left at 0 it is derived as a hundred times the smallest adjacent Young's modulus divided by the
  characteristic face size, :math:`\varepsilon_N = 100\,E/h`. No nodal weight enters that rule, and
  none should: the weights differ by a factor of four across an ordinary interface, so a mean weight
  cannot stand in for them; the per-node normalisation belongs in the stiffness and is applied
  there. The literature gives the scale of the value, not the value: Wriggers (2006), Remark 6.4,
  finds that a penalty "at least 100 times EA" is needed even in a two-bar example and that it has
  to be chosen as a function of the mesh size, and Ghesquière-Diérickx et al. (2026), Eq. (12),
  quote :math:`\alpha E/h` with :math:`\alpha` between :math:`10^1` and :math:`10^4` as the common
  guideline. The derived number is reported. With ``augmentedLagrange`` enabled the converged result does not depend on it
  at all; it then only sets how fast the outer loop converges.

  **Under explicit dynamics it is bounded from above** and is derived from a different rule
  entirely, because there the stiffness competes with the stability limit of the time integrator
  rather than with the conditioning of a linear solve. See the next section.

* ``augmentationTolerance`` and ``maxAugmentations`` bound the outer loop. The tolerance is a
  RELATIVE change of the pressure estimate from one augmentation to the next, which makes it
  independent of the unit the model is measured in. Reaching the cap is reported; it means the
  increment converged on an estimate that had not settled, and the remedy is a larger cap, a larger
  stiffness, or a smaller increment.

* **Which surface is the non-mortar one** is a parameter in all but name, and on interfaces of
  unequal extent it is the most consequential one on this list. Put the SMALLER surface there. Every
  one of its facets is then fully covered and no boundary row arises at all. With the sides the
  other way round, a surface that ends while still transmitting pressure produces them in numbers:
  measured on a half-width punch, nodes outside the punch take 7 % of the transmitted force,
  falling to 2 % under eightfold refinement, in all three formulations alike. See the section on the
  choice of sides.

* **Element type.** ``CONQUAD8`` is the quadratic surface element to prefer. ``CONQUAD9`` receives
  no basis transformation -- its corner integrals are already positive, so it needs none for the
  full element -- and therefore has no pointwise non-negativity either, which lets its nodal weights
  turn negative under partial coverage. ``CONQUAD8`` at :math:`\alpha = 1/3` is not immune to that
  either, but reaches it only for overlaps confined to the small region where its transformed corner
  function is negative.

  Which types are actually reachable is narrower than the list of types the constraint supports.
  :mod:`~edelweissfe.generators.surfaceelementgenerator` emits a contact element from a face of a
  solid element, so it can only emit the faces the solid elements of EdelweissFE have: ``CONLINE2``
  and ``CONLINE3`` from the two-dimensional ones, ``CONQUAD4`` from ``hexa8`` and ``CONQUAD8`` from
  ``hexa20``. ``CONQUAD9``, ``CONTRI3`` and ``CONTRI6`` are implemented throughout the constraint
  and the contact element, but their source elements -- a 27-node hexahedron and the tetrahedra --
  do not exist in EdelweissFE, so no generator call can produce them. They are reachable today only
  from a hand-written element set, and are there so that the constraint does not have to be revisited
  when such an element is added.


Inside the solver
~~~~~~~~~~~~~~~~~

**Under ``geometryUpdate=consistent`` the geometry follows the iterate.** The segmentation, the
coupling matrices and the nodal normals are rebuilt at every assembly, at the configuration the
solver hands in, together with their derivatives; nothing geometric is carried from one iterate to
the next. With friction, the coupling matrices of the last converged configuration are built once
more at the start of every increment attempt, since the objective slip measures the change from
them.

**Under ``geometryUpdate=frozen`` the geometry is frozen, and the SOLVER decides how often it is
rebuilt.** The segmentation, the coupling matrices and the nodal normals are evaluated from the last
CONVERGED configuration and held fixed until the solver's connectivity tick --
``updateConnectivity`` -- asks for them again. Under an implicit solver that tick comes once per
increment. Under explicit dynamics, which always runs on the frozen geometry because it assembles no
tangent, it comes every ``contact-update-frequency`` increments instead, for reasons given in the
next section.

The reference is deliberately the converged state and not the incoming iterate: solvers
extrapolate the previous increment before the first assembly, and freezing at that predictor would
make the converged contact solution depend on a solver switch that must not influence it. A
cutback re-attempt starts from the same converged state, and so from the same frozen geometry; the
per-increment bookkeeping is reset with it.

Routing the rebuild through the tick rather than through a new increment number is what separates
the two cadences. The per-increment bookkeeping -- the active set, the augmentation counter, the
iteration count -- still follows the increment, so wherever the solver ticks once per increment the
two coincide exactly and nothing about the implicit behaviour changes.

What the frozen geometry costs is the subject of the section on the tangent above. What it buys is
that the contact contribution of one increment is a smooth function of the displacements, with no
re-segmentation happening underneath the iteration, and an assembly that is cheaper by the
segmentation. Measured per assembly, on ``MortarContactPatchHexa20`` 37 ms frozen against 218 ms
consistent, on ``MortarContactPenaltyHexa20`` 12 ms against 214 ms, on ``MortarContactFrictionSlide``
2.2 ms against 19 ms; the consistent geometry spends that on rebuilding the segmentation with its
derivatives at every iterate.

**The active set is re-decided in every Newton iteration.** This is the semi-smooth Newton method,
and it is what gives the active set superlinear convergence rather than a fixed-point character.
Three safeguards surround it:

* the set is frozen for the remainder of an increment once it has settled -- unchanged over two
  iterations -- so that the iteration can converge on a fixed set rather than chase a moving one;
* it is also frozen when a discrete state already visited in this increment recurs, because a cycle
  would otherwise repeat indefinitely without either converging or failing;
* and a hard cap of 20 iterations remains as a last resort. Reaching it is reported, because the
  increment then converges onto a set that was never confirmed and its solution is not guaranteed
  to satisfy the conditions above.

**Two hooks on the constraint interface** exist for the augmented formulation, and both are no-ops
for every other constraint:

* ``augmentConstraint`` runs one outer iteration after the Newton loop of the increment has
  converged -- never inside it, since it is the converged violation that is being corrected. It
  returns whether another equilibrium iteration is needed.
* ``requiresCorrectionBeforeConvergence`` is true on the first assembly of an increment under
  ``formulation=penalty``, whose pressure is a function of the state rather than an unknown: the
  forces of the extrapolated state that opens the increment have never been equilibrated, so the
  increment may not be declared converged on them. Without this an increment can be accepted at
  iteration 0: the field-correction criterion is then satisfied by an ABSENT correction rather than
  a small one.

**The multipliers are ordinary scalar variables** of the model. They are assembled into the global
system, they appear in restart state, and they are part of what the regression runner compares
against its reference -- which is why a ``formulation=lagrange`` deck checks the contact pressures
and not only the displacements.


Mesh refinement and restart
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The constraint follows an h-adaptive refinement of either surface.** It is a
:class:`~edelweissfe.models.meshdependent.MeshDependent`: when a refinement touches the solid
elements under one of its surfaces, the surface element generator regenerates every contact element
of that surface (``facets=wholeFace`` included) on the solids' own nodes, and the constraint rebuilds
its contact system from the new element sets -- node lists, degrees of freedom, reference
coordinates, undeformed normals -- before the next increment. The geometry is discarded and rebuilt
at the next assembly. A node that survives the
refinement is the same object before and after, and keeps its state: the frictional history, the
augmented pressure estimate, the active-set entry, the reference gap vector of the objective slip
correction and, under ``formulation=lagrange``, the converged multiplier. A node created by the
refinement takes the frictional history (traction, pressure, contact status) and the augmented
pressure estimate interpolated from the old non-mortar element it lies on, with that element's shape
functions at its location (Wriggers 2006, Section 14.7.2, Eqs. 14.83-14.86); its active-set entry
and multiplier start at zero and are found by the first iteration. At the first rebuild after the
refinement the history is carried into the new tangent planes and returned onto the cone of its
pressure, since a transferred traction need not satisfy the slip condition (Perić et al. 1996,
Remark 4.3), but without the objective slip correction: its reference gap vector belongs to the old
coupling matrices, and the difference would be the change of discretisation, not slip. The penalty
stiffness derived from the element size and the characteristic length of ``slipTolerance`` are
determined once, on the surfaces of the first assembly, and stay as they are.
The number of multipliers follows the non-mortar nodes, so the constraint re-creates its scalar
variables itself; the refresh reports a changed footprint and the solver rebuilds its equation
system. Hanging nodes of the refined surface take part like any other node, their degrees of freedom
eliminated by the hanging-node constraint of the refinement.

**A restart reproduces the refined interface and its state.** The checkpoint holds the refinement
history; reading it replays the refinements, so the resumed model has the refined surfaces again,
and only then is the contact state restored. Since that happens before any solver has refreshed the
mesh dependents, the constraint first catches up with the refined surfaces itself and then matches
the checkpoint to its non-mortar nodes by node label -- including the multipliers, which as scalar
variables are restored by name, i.e. by index, and no longer identify a node once the surface was
refined, and the nodal normals of the last rebuild, against which the first rebuild after the
restart checks for a crease. The node-to-surface and the integrated contact constraints do the same,
by slave node and by slave facet and quadrature point.

Both are verified end to end in ``tests/test_contact_amr_restart.py``, with the node-to-surface
constraint, the integrated constraint, the mortar penalty formulation with friction and the mortar
Lagrange formulation: a frictional drag during which both contact surfaces are refined; the same run
truncated after the refinement and resumed from its checkpoint, which ends with the displacements of
the uninterrupted run; and a traction field linear in the coordinates, imposed before each
refinement, which every new point carries exactly afterwards.


Explicit dynamics with the mortar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only ``formulation=penalty`` survives an explicit increment, and the other two are refused rather
than quietly reinterpreted.

* ``lagrange`` needs its multipliers as unknowns of a system an explicit increment never solves.
  They would also carry no inertia, so the explicit update would never move them: the constraint
  would appear active in every output and enforce nothing (Wriggers 2006, p. 352: "there is no mass
  associated with the Lagrange multipliers"). The explicit dynamic solver already
  refuses any constraint carrying its own scalar variables, and the constraint repeats the refusal
  at its own explicit entry point.
* ``augmentedLagrange`` is the dangerous one. Its outer update runs from ``augmentConstraint``,
  which only the implicit solvers call, so under an explicit solver it would simply never run --
  and the constraint would degrade to the pure penalty form without failing and without saying so.
  It is refused for that reason.

**The segmentation is throttled, not run per time step.** Every explicit time step is an increment,
and rebuilding the geometry at each of them is not affordable: measured on the constraint, the
segmentation and the nodal normals are 93 to 98 % of its cost -- 26 ms per rebuild on a small
linear interface and 246 ms on a quadratic one -- against time steps counted in millions. The
rebuild therefore follows ``updateConnectivity``, which the explicit solver ticks every
``contact-update-frequency`` increments. On ``testfiles/edelweiss-only/NEDMortarContact``, 314 time
steps at a frequency of 10 produce 32 segmentations rather than 314, and change the result by
0.14 % -- the staggering the throttle buys, an error of the update interval rather than of the
formulation.

What makes the throttle defensible is the same argument the node-to-surface constraint's search
rests on: between two rebuilds a slave node moves :math:`\lVert v \rVert \, \Delta t \, f`, orders
of magnitude below a facet dimension.

.. _mortar-explicit-stability:

The stability limit on the penalty stiffness
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An explicit solver derives its critical time step from the elements -- a wave speed and a
characteristic length per element, the CFL condition of the MESH. A penalty contact adds its own
frequency on top of that, and it is *not* in that estimate (Wriggers 2006, p. 352: stiff penalty
springs make the Courant criterion lead to small time steps). A stiffness chosen for accuracy can therefore put the integration past the stability limit
with nothing to report it. Measured on ``NEDMortarContact`` with everything else held fixed:

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - :math:`\varepsilon_N`
     - final ``dispLowerMax``
   * - :math:`1.6 \cdot 10^{5}`
     - :math:`7.78 \cdot 10^{-1}` -- unstable
   * - :math:`1.2 \cdot 10^{5}`
     - :math:`1.57 \cdot 10^{-2}` -- the constraint warns here, one step early
   * - :math:`8 \cdot 10^{4}`
     - :math:`1.29 \cdot 10^{-2}`
   * - :math:`2 \cdot 10^{4}`
     - :math:`9.24 \cdot 10^{-3}`
   * - :math:`5 \cdot 10^{3}`
     - :math:`4.38 \cdot 10^{-3}`

The bound the constraint imposes follows from the structure of its own stiffness. The penalty
contribution of one non-mortar node is RANK ONE,

.. math::

   \boldsymbol{K}_I = \kappa_I\, \boldsymbol{v}_I \boldsymbol{v}_I^{\mathsf T},
   \qquad \boldsymbol{v}_I = \boldsymbol{w}_I \otimes \boldsymbol{n}_I ,
   \qquad \kappa_I = \frac{\varepsilon_N}{\lvert D_{II} \rvert} ,

so the assembled contact stiffness is :math:`\boldsymbol{K} = \varepsilon_N \sum_I
\lvert D_{II}\rvert^{-1} \boldsymbol{v}_I \boldsymbol{v}_I^{\mathsf T}`, and Gershgorin's theorem
bounds the largest eigenvalue of :math:`\boldsymbol{M}^{-1}\boldsymbol{K}` by the largest scaled
absolute row sum, as in Kulak (1989), Eqs. (11)--(12), and Ghesquière-Diérickx et al. (2026),
Eq. (18):

.. math::
   :label: mortar-stability-factor

   \omega^2 \;\le\; \varepsilon_N \, \underbrace{\max_a \frac{1}{m_a}
   \sum_I \frac{\lvert v_{I,a} \rvert \, \lVert \boldsymbol{v}_I \rVert_1}
   {\lvert D_{II} \rvert}}_{=:\, R}

The :math:`1/\lvert D_{II}\rvert` sits inside the sum over the nodes and cannot be pulled out of
it: the weights differ by a factor of four across an ordinary interface, and a bound taken with a
mean weight would be wrong by that factor at the stiffest node -- which is precisely the node that
sets the stable increment.

:math:`R` carries the geometry and the inertia and not the stiffness, which is what makes it usable
in both directions. With the central-difference stability limit :math:`\Delta t_{cr} =
2/\sqrt{\lambda_{\max}}` (Kwon et al., Eqs. 20--21; Ghesquière-Diérickx et al., Eq. 17),

.. math::

   \Delta t \;\le\; \frac{2}{\sqrt{\varepsilon_N R}}
   \qquad\Longleftrightarrow\qquad
   \varepsilon_N \;\le\; \frac{4}{\Delta t^{2} R}

Gershgorin OVERestimates the eigenvalue, so the bound is conservative -- it may ask for a slightly
smaller time step, or a slightly softer stiffness, than strictly necessary. That is the right
direction to err in. The exact rank-one eigenvalue :math:`\kappa_I \, \boldsymbol{v}_I^{\mathsf T}
\boldsymbol{M}^{-1} \boldsymbol{v}_I` would be exact for one node in isolation and UNDERestimate as
soon as two nodes share a degree of freedom, which is the normal case. Every non-mortar node is
counted, not only the currently active ones: the active set changes from increment to increment,
and a bound that held only for the set that happened to be active when it was computed would not be
a bound (Ghesquière-Diérickx et al. likewise include all potential contact facets). Kulak shows on
two impacting bars that the Rayleigh quotient of the contact element underestimates the exact
eigenvalue (by 39 % when contact and bar stiffness are equal) while the Gershgorin bound
overestimates it. The bound covers the contact on its own: the solver takes the smaller of it and
the elements' own critical time step, as Kulak does (his Eq. 14), but his row sums also contain the
stiffness of the elements adjoining the contact nodes, so the minimum here is not a bound of the
elements and the contact together.

**What the constraint does with it.** Two cases, and neither of them is silent:

* ``penaltyStiffness`` **not given.** It is derived from the stability limit rather than from the
  material, at half of :math:`4/(\Delta t^2 R)` -- the factor of two in the stiffness is a factor
  of :math:`\sqrt 2` in the frequency, and it is the margin for the geometry moving between two
  connectivity ticks. Re-derived at every rebuild, since :math:`R` follows the segmentation. This
  is the same idea as the ``SOFT=1`` contact of LS-DYNA, which takes its stiffness from the nodal
  masses and the time step rather than from the material constants, except that here it follows
  from the eigenvalue bound above rather than from a rule of thumb. On ``NEDMortarContact`` it
  lands at :math:`6.8 \cdot 10^{5}`.
* ``penaltyStiffness`` **given.** The value is honoured, and measured against the bound. An
  exceedance is reported together with the time increment that would carry it and the stiffness
  that this increment would. Reported rather than refused, because the bound is conservative and
  refusing would block models that are in fact stable.

The reported bound is also returned to the solver through
``ConstraintBase.computeCriticalTimeStepForExplicitDynamics``, which defaults to infinity, so that a
solver can take the minimum over its constraints as it does over its elements. Every other
constraint inherits that default; the node-to-surface and the integrated penalty contact do not
override it, so their penalty stiffness is not covered by the solver's time step.

.. rubric:: References for this section

* Y.-J. Kwon, J.-G. Kim, S. S. Cho, J. A. González: *A General Bipenalty Formulation for Explicit
  Contact-Impact Analysis With a Parameter Selection Criterion.* International Journal for
  Numerical Methods in Engineering 126 (2025) e7614. -- The central-difference stability limit,
  Eqs. (20)--(21).
* T. Ghesquière-Diérickx, J.-F. Molinari, G. Anciaux: *Stability of extrinsic cohesive-zone model
  with penalty-based contact in explicit dynamic fragmentation simulations*, Mechanics of Materials
  215 (2026) 105581.
  -- The penalty scale :math:`\alpha E/h`, Eq. (12); the critical time step and its Gershgorin bound
  over all potential contact facets, Eqs. (17)--(18).
* R. F. Kulak: *Critical time step estimation for three-dimensional explicit impact analysis*,
  Argonne National Laboratory, report CONF-8907161, 1989. -- The Gershgorin bound of the critical
  time step of penalty contact from nodal row sums, Eqs. (11)--(12), combined with the elements'
  own limit by the minimum, Eq. (14).
* LS-DYNA support documentation, *SOFT option* and *Contact stiffness calculation*,
  https://www.dynasupport.com/howtos/contact/soft-option -- with ``SOFT=1`` the contact stiffness
  is ``max(SLSFAC*SFS*k0, SOFSCL*k1)``, where ``k1`` is "calculated from nodal masses and the
  solution time step" and ``k0`` from the bulk modulus and the element dimensions; the practical
  precedent for the derivation above.


What is verified, and how
~~~~~~~~~~~~~~~~~~~~~~~~~

Every number below is measured on the cases named, not estimated.

**The patch test** establishes the property the whole method exists for: a constant pressure
crossing an interface whose two sides share no nodes. Four decks cover it, in two and three
dimensions and in both element orders, each against its closed-form uniaxial solution. Measured
deviation of the displacement field, and of the contact pressure, which is exact because the
multipliers are scalar variables of the system:

=================================  ====================  ====================  ==================
Deck                               max lateral           max axial deviation   contact pressure
=================================  ====================  ====================  ==================
``MortarContactPatch`` (2D)        8e-17                 2e-16 of 0.02         10.000000000
``MortarContactPatchQuadratic``    7e-16                 2e-15 of 0.02         10.000000000
``MortarContactPatchHexa8``        1e-14                 5e-14 of 2            10.000000000
``MortarContactPatchHexa20``       3e-15                 7e-15 of 0.02         10.000000000
=================================  ====================  ====================  ==================

The lateral constraints of those decks fix the symmetry planes only, never the whole model, so the
lateral displacements are a result of the formulation rather than a boundary condition read back.
``MortarContactPatchHexa8`` works at a hundred times the length scale of the others, which is what
notices a tolerance inside the constraint that compares a length against an absolute bound.

**The unilateral condition** is covered by two further decks. ``MortarContactSeparation`` walks one
model through closing, separating and closing again in three steps: in the separated state every
multiplier is exactly 0 and the non-mortar block is left behind at rest to 3.1e-18, and the
re-closed state is exact again to 4e-16 -- so the active set can empty and refill.
``MortarContactPartialOverlap`` leaves part of the non-mortar surface uncovered and tabulates
weight, coverage, multiplier and force across the interface; ``MortarContactPunchEdge`` is the same
configuration with the favourable choice of sides, where the coverage is exactly 1.000 at every node
and the multipliers run from 9.34 in the interior to 22.97 at the rim.

``MortarContactTilted`` has an interface that is partly closed and partly open, with nodes that
have a counterpart and are nevertheless separated; ``tests/test_mortarsignorini.py`` checks the
Signorini conditions of its converged state.

**Friction** is covered by ``MortarContactFrictionSlide``, a pressed block dragged across a larger
one under the implicit solver so that every contact node passes from stick into slip, and by
``NEDMortarContactFriction``, the same under explicit dynamics; ``tests/test_mortarfriction.py``
checks that the drag ends on the friction cone and that the explicit run transmits a frictional
force. The consistency of the friction law across the three contact formulations is checked in
``tests/test_contactfriction_consistency.py``.

**The penalty branch and the outer loop** are covered by ``MortarContactPenaltyHexa20``, the
quadratic patch test with the multipliers eliminated. It is the only deck that reaches the two
solver hooks. Its header carries the comparison against the same model with the loop switched off.

**The tangent** is checked against central differences of the assembled forces with the active set
held fixed, since across a change of the set there is no tangent to compare. Under
``geometryUpdate=consistent`` the geometry is rebuilt at every perturbation
(``tests/test_mortarlinearization.py``): all seven contact element types, both formulations, and
friction in stick and slip with and without ``slipTolerance``, on a tilted, curved, laterally shifted
pair of surfaces, agree to :math:`\max |K - K_{FD}| / \max |K| \le 1.1 \cdot 10^{-7}`; every
building block of the linearisation is checked on its own as well. Under ``geometryUpdate=frozen``
the geometry is held at the start of the increment, which is the residual that tangent belongs to;
measured on an interpenetrating, laterally shifted pair of quadratic facets with every node in
contact: :math:`\max |K - K_{FD}| / \max |K| = 5.3 \cdot 10^{-10}`.

**The building blocks** are covered by unit tests beside the code rather than by decks: the clipping
against analytic overlap areas including the area a non-convex window loses, the triangulation, the
facet normals and the tangent basis, the coupling matrices and their row-sum identity, the coverage
bookkeeping, the noise floor of the penalty activation, and -- at element level -- cardinal shape
functions for all seven types, exact quadrature of the element measure, biorthogonality of the dual
basis recomputed from the element's own quadrature, and the measured proof that the untransformed
serendipity corner integral is :math:`-A/12` and the quadratic triangle's is exactly zero.

**Against an analytical solution.** The formulation is compared against Hertzian line contact on a
parabolic indenter, a case with a curved, load-determined contact zone. Quadratic elements reproduce
the peak pressure to **1.0 %**, and -- the decisive observation -- that figure does not move under
mesh refinement: 1.01 % on the medium mesh, 1.00 % on a mesh eight times finer. The residual is
therefore not a discretisation error but the price of a finite body standing in for the half-space
Hertz assumes: the block is about nine contact half-widths wide, twelve deep, and rigidly clamped at
its base, all of which stiffen it, and a stiffer foundation gives a higher peak pressure. Linear
elements reach 6.6 % on the medium mesh and are still converging at 4.3 % on the fine one.

These curved cases are not part of the shipped test suite. They take 8 s and 89 s respectively,
which is out of proportion to a suite that runs in about a minute, and reproducing them needs a
parabolically warped mesh that no model generator provides.

**Under explicit dynamics**, ``NEDMortarContact`` is the shipped deck. Its decisive quantity is
``dispLowerMax``: nothing but the contact loads the lower block, so it is identically zero unless
the constraint's forces reach the equations, and a run that merely completes proves nothing. The
throttle is measured on the same deck, before and against after: 314 time steps produce 314
segmentations without it and 32 with it at ``contact-update-frequency = 10``, changing the result
from 1.20688e-2 to 1.20520e-2, or 0.14 %.

The stability bound is covered by unit tests rather than by a deck, in
``tests/test_mortarcontact.py``. Two of them state the properties that make the
bound a bound rather than a number: that :math:`\Delta t \le 2/\sqrt{\varepsilon_N R}` holds, checked by
quadrupling the stiffness and requiring the reported limit to halve exactly; and that a stiffness
derived for a given increment satisfies the bound it was derived for, which is the property the
whole mechanism exists to guarantee. A third checks that :math:`\varepsilon_N` follows
:math:`1/\Delta t^2`, and a fourth that a constraint which does not implement the hook leaves the
mesh's time step untouched.


Known limitations
~~~~~~~~~~~~~~~~~

These are measured, not suspected.

Force reaching nodes outside a contact surface that ends under pressure
    Where the mortar surface ends while still transmitting full pressure -- a flat punch, an anchor
    head, a bearing plate -- the non-mortar surface kinks at that edge under load. The adjacent
    facets, which lie entirely outside the mortar surface, tilt, and the projection of the mortar
    surface leaks a sliver onto them. Their nodes acquire a coverage of order :math:`10^{-5}`, and a
    node is constrained exactly like a fully covered one regardless of coverage: coverage sets how
    the contact force is distributed, never whether a node is tied.

    Measured on a half-width punch (the geometry of ``MortarContactPunchEdge`` with the sides
    exchanged, hexa8): 7.1 % of the transmitted force lands on nodes outside the punch, and the share
    falls under uniform refinement -- 5.0 % at twice the resolution, 3.3 % at four times, 2.2 % at
    eight. It falls, but slowly, roughly with the square root of the element size. The total force
    is unaffected, being fixed by equilibrium; what is wrong is where it ends up. The constraint
    itself is satisfied throughout, the normalised gap at such a node measuring
    :math:`9 \cdot 10^{-13}`, the level of the Newton tolerance, as at every other node.

    It is NOT confined to ``formulation=lagrange``. Since the penalty law is pointwise, a node on a
    sliver carries the pressure its penetration gives it, not a sliver's share of it, and the same
    model misplaces 7.08 % with ``penalty`` and with ``augmentedLagrange`` alike, and the share
    hardly depends on :math:`\varepsilon_N` (7.1 % at :math:`10^5`, 7.08 % at :math:`10^6`; only a
    very soft :math:`10^4` gives 7.7 %). A stiffness of :math:`10^9` does not converge on this model
    at all, see the next entry.

    Two things bound the problem. It requires the pressure to be at full value where the surface
    ends, so a contact that tapers off between curved bodies is unaffected -- a cylindrical
    indenter measured 0 at every refinement. And it does not arise at all when the smaller surface
    is the non-mortar one, which is the remedy named above.

    No treatment of it is implemented. The weighting of the boundary rows does not address it and
    cannot: a rescaling of the constraint leaves the solution invariant, and it is the solution that
    is at issue here, not the conditioning.

A non-smooth geometry at slivers, under ``geometryUpdate=consistent``
    The geometry is a smooth function of the displacements only for a fixed layout, and at a sliver
    the layout changes under displacements of the order of the rounding. On the model of the
    previous entry, a Gauss point of the sliver lies on the edge shared by two non-mortar facets;
    a perturbation of :math:`10^{-7}` moves 11 such points into the neighbouring facet (see
    :meth:`~edelweissfe.constraints.mortarcontact.Constraint._containingFacet`) and changes the
    residual there by 3 %. No tangent describes such a jump. The frozen geometry meets it only from
    one increment to the next; the consistent one meets it within the Newton iteration. Measured on
    that model: ``lagrange`` converges at every refinement, and ``penalty`` up to
    :math:`\varepsilon_N = 10^6`, i.e. :math:`10^3 E`; at :math:`10^9` the iteration diverges at
    every increment size, while ``geometryUpdate=frozen`` converges. With the smaller surface on the
    non-mortar side there is no sliver, and the same stiffness, and :math:`10^{12}`, converge.

Reproducibility under threaded BLAS, ``geometryUpdate=consistent``
    The linearisation is assembled by matrix products that numpy hands to BLAS. With more than one
    BLAS thread their summation order varies from run to run, and two runs of the same deck then
    agree to rounding rather than bit for bit; a frictional run, whose stick/slip decisions follow
    that rounding, measured :math:`9 \cdot 10^{-6}` of a displacement of 0.4. With
    ``OMP_NUM_THREADS=1`` the results are reproducible exactly.

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
follows, and each entry names what it underpins here. Where a comment in the code states a formula,
a bound or a quoted phrase that comes from one of them, it names the source at that point too, so
that a reader checking a single line does not have to reconstruct which work it belongs to.

*Dual Lagrange multipliers and their basis*

* B. I. Wohlmuth: *A mortar finite element method using dual spaces for the Lagrange multiplier*,
  SIAM Journal on Numerical Analysis 38 (2000), 989--1012. -- The dual space that keeps the
  multiplier field local.
* B. P. Lamichhane, R. P. Stevenson, B. I. Wohlmuth: *Higher order mortar finite element methods in
  3D with dual Lagrange multiplier bases*, Numerische Mathematik 102 (2005), 93--121.
* A. Popp, B. I. Wohlmuth, M. W. Gee, W. A. Wall: *Dual quadratic mortar finite element methods for
  3D finite deformation contact*, SIAM Journal on Scientific Computing 34 (2012), B421--B446. --
  The basis transformation for quadratic surfaces (their Section 4.4.1, Eqs. (4.6) and (4.9)), its
  admissibility bounds, and the reason the nine-node quadrilateral needs none -- "no basis
  transformation is needed in the case of quad9 surfaces, because the corresponding shape functions
  already satisfy" the positivity condition. They suggest :math:`\alpha = 1/5`; the value used here
  is Farah's, for the reason given under *The dual basis*. The transformation is constant, "defined
  once at t = 0", p. B432, so it carries no derivative.
* T. Cichosz, M. Bischoff: *Consistent treatment of boundaries with mortar contact formulations
  using dual Lagrange multipliers*, Computer Methods in Applied Mechanics and Engineering 200
  (2011), 1317--1332. -- The partially covered facet at the boundary of the contact area: the
  quadratic decay of its nodal weight, the reciprocal growth of its multiplier and the resulting
  loss of conditioning, and both remedies used here -- biorthogonality over the covered area, and
  the weighting of the boundary rows. Their Section 4 and Eqs. (41) to (45). The choice of sides is not one of
  those but their Section 6.1, which shows by example what exchanging the two surfaces does to
  the contact pressure and to the condition number.

*Segment-to-segment contact*

* M. A. Puso, T. A. Laursen: *A mortar segment-to-segment contact method for large deformation solid
  mechanics*, Computer Methods in Applied Mechanics and Engineering 193 (2004), 601--629. -- The
  penalty form in terms of the weighted gap, their Eq. (24). Their Section 3.1 also reports the
  seven-point rule used on the clipped cells here as "more than sufficient when compared to higher
  order nine and thirteen point schemes", with the caveat that the choice "is somewhat problem
  dependent".
* M. A. Puso, T. A. Laursen, J. Solberg: *A segment-to-segment mortar contact method for quadratic
  elements and large deformations*, Computer Methods in Applied Mechanics and Engineering 197
  (2008), 555--566. -- The augmented outer loop, their Eq. (18), advanced only once equilibrium has
  converged; and the three-dimensional friction trial state with the re-projected history, Eqs.
  (16)--(21). Their Fig. 3 is also the decomposition of a quadratic facet into linear sub-cells used
  here: four quadrilaterals for the nine-node patch, four triangles for the six-node patch, and four
  triangles plus one quadrilateral for the serendipity eight-node patch.
* B. Yang, T. A. Laursen, X. Meng: *Two dimensional mortar contact methods for large deformation
  frictional sliding*, International Journal for Numerical Methods in Engineering 62 (2005),
  1183--1225. -- The penalty-regularised Coulomb law of the friction option, its trial state and
  return map, Eqs. (39) and (47)--(49); and the nodal scale factor of the penalty normalisation.
* J. C. Simo, T. A. Laursen: *An augmented Lagrangian treatment of contact problems involving
  friction*, Computers & Structures 42 (1992), 97--116. -- The penalty-regularised friction law,
  Eqs. (3.9.1)--(3.9.5), and its trial state with the traction as history and radial return,
  Eqs. (3.12)--(3.14); no friction without pressure, p. 104.
* P. Wriggers: *Computational Contact Mechanics*, 2nd ed., Springer, 2006. -- The radial return of
  the friction law, Section 10.5.2, Eqs. (10.121)--(10.126) and Box 14; the non-symmetric slip
  tangent, p. 359 with Eq. (9.47); the penalty parameter as a function of the mesh size, Remark 6.4;
  the projected multiplier update, Eq. (10.79); Lagrange multipliers and stiff penalties under
  explicit time integration, p. 352; the transfer of the frictional history to the nodes a mesh
  refinement creates, Section 14.7.2, Eqs. (14.83)--(14.86).
* D. Perić, Ch. Hochard, M. Dutko, D. R. J. Owen: *Transfer operators for evolving meshes in small
  strain elasto-plasticity*, Computer Methods in Applied Mechanics and Engineering 137 (1996)
  331--344. -- Remark 4.3: a transferred state need not satisfy the yield condition; here, the
  transferred traction is returned onto the friction cone after a mesh refinement.
* M. Gitterle, A. Popp, M. W. Gee, W. A. Wall: *Finite deformation frictional mortar contact using a
  semi-smooth Newton method with consistent linearization*, International Journal for Numerical
  Methods in Engineering 84 (2010), 543--571. -- The objective weighted slip increment, Eq. (53),
  which the friction option uses under ``geometryUpdate=consistent`` and which the frozen-geometry
  slip measure approximates; its linearisation, Appendix A, Eq. (A3).
* A. Popp, M. W. Gee, W. A. Wall: *A finite deformation mortar contact formulation using a
  primal-dual active set strategy*, International Journal for Numerical Methods in Engineering 79
  (2009), 1354--1391. -- The consistent linearisation in two dimensions, Appendix A.
* A. Popp, M. Gitterle, M. W. Gee, W. A. Wall: *A dual mortar approach for 3D finite deformation
  contact with consistent linearization*, International Journal for Numerical Methods in Engineering
  83 (2010), 1428--1465. -- The averaged nodal normal; the consistent linearisation of the
  geometry, Section 4.3 and Appendix A, Eqs. (A1)--(A26); the neglected variation of the coupling
  matrices, Section 7; the cost of an incomplete linearisation, Table I.
* P. Farah: *Mortar Methods for Computational Contact Mechanics Including Wear and General Volume
  Coupled Problems*, Dissertation, TU München, 2018. -- The averaged nodal normal written out as
  Eq. (4.41), over the *unit* normals of the adjacent elements; the weighted relative tangential
  velocity in three dimensions, Eq. (3.43); and the transformation parameter
  :math:`\alpha = 1/3` of his Section 6.2.3.2, Eq. (6.20), which is the value used here.
* P. Farah, A. Popp, W. A. Wall: *Segment-based vs. element-based integration for mortar methods in
  computational contact mechanics*, Computational Mechanics 55 (2015), 209--228. -- Why the overlap
  is segmented rather than integrated on the parent element: segment-based integration is "the best
  available integration scheme ... with regard to accuracy". Two qualifications belong with that
  sentence. It is drawn there for frictional contact, where they find element-based integration more
  sensitive than in the frictionless case treated here; and their own recommendation is neither
  extreme but a boundary-segmentation compromise, which this implementation does not use.

*The semi-smooth treatment of the unilateral condition*

* T. De Luca, F. Facchinei, C. Kanzow: *A semismooth equation approach to the solution of nonlinear
  complementarity problems*, Mathematical Programming 75 (1996), 407--439.
* M. Hintermüller, K. Ito, K. Kunisch: *The primal-dual active set strategy as a semismooth Newton
  method*, SIAM Journal on Optimization 13 (2002), 865--888. -- Eq. (3.2), Lemma 3.1 and
  Remark 3.1: the generalized derivative of the max operator, with an arbitrary value where its
  argument vanishes; the basis for treating a closed node as active.
* S. Hüeber, B. I. Wohlmuth: *A primal-dual active set strategy for non-linear multibody contact
  problems*, Computer Methods in Applied Mechanics and Engineering 194 (2005), 3147--3166. -- The
  lower end of the admissible band for the complementarity parameter, and its linear dependence on
  the stiffness -- reported there as an observation, and for their inexact strategy; see *The
  unilateral condition* for why that does not carry over unchanged.
* M. Gitterle, A. Popp, M. W. Gee, W. A. Wall: *Finite deformation frictional mortar contact using a
  semi-smooth Newton method with consistent linearization*, International Journal for Numerical
  Methods in Engineering 84 (2010), 543--571. -- The complementarity function in its contact form,
  their Eq. (55), and the weighted gap of their Eq. (36); that the parameter cannot affect the
  accuracy of the solution; and the upper end of the band, where the active set begins to chatter.

*Geometry and quadrature*

* I. E. Sutherland, G. W. Hodgman: *Reentrant polygon clipping*, Communications of the ACM 17
  (1974), 32--42. -- The clipping algorithm, including its restriction to convex clip *windows*;
  the subject polygon is unrestricted, and "reentrant" in the title refers to the code rather than to
  a polygon. Their Appendix B decomposes a concave polygon into convex pieces, which is deliberately
  not used here.
* D. A. Dunavant: *High degree efficient symmetrical Gaussian quadrature rules for the triangle*,
  International Journal for Numerical Methods in Engineering 21 (1985), 1129--1148. -- The
  seven-point symmetric rule of degree five on the triangle, applied to each clipped cell.
* C. Ericson: *Real-Time Collision Detection*, Morgan Kaufmann, 2004. -- The bounding-volume
  hierarchy of the candidate search.
* A. Konyukhov, K. Schweizerhof: *On the solvability of closest point projection procedures in
  contact analysis: analysis and solution strategy for surfaces of arbitrary geometry*, Computer
  Methods in Applied Mechanics and Engineering 197 (2008), 3045--3056. -- That the closest-point
  projection of a Gauss point back onto a facet need not be solvable at all, and under which
  conditions it fails; the constraint reports the failures rather than hiding them.
