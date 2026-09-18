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
~~~~~~~~~~~

The geometry -- the segmentation, the coupling matrices and the nodal normals -- is evaluated once per
increment and held fixed for its Newton iterations, rather than recomputed per iteration. The
tangent is consistent with the residual that follows from that frozen geometry, and that is the
tangent the iteration needs: it is the residual the solver actually assembles.

Two measurements separate what is exact from what is approximate, both on the same interpenetrating
pair of quadratic facets with every node in contact.

Against finite differences of the residual **with the geometry frozen**, which is the tangent's own
claim, the agreement is
:math:`\max |K - K_{FD}| / \max |K| = 5.3 \cdot 10^{-10}` -- exact to round-off.

Against finite differences **with the geometry recomputed at every perturbation**, the difference is
0.6 % of :math:`\max |K|` on a flat interface. That is the size of the terms the freezing omits: the
derivatives of the coupling matrices and of the nodal normals with respect to the configuration.
They are not part of the assembled tangent, and where the geometry of the overlap changes strongly
with the displacement they are larger than this.

What follows from the omission is a distinction worth keeping straight, because the literature
measures a different thing under a similar name. Popp, Gitterle, Gee and Wall report an *incomplete
linearisation* -- a LIVE geometry carried with a tangent that omits exactly these terms, the nodal
normal and the two coupling matrices -- and it costs them 52 Newton steps against 8, which is why
they call full linearisation indispensable. That is not this scheme. Here the geometry is frozen, so
the assembled tangent is exact for the residual that is actually assembled, and the Newton rate is
not what suffers: the first measurement above says so.

The cost lies in the residual instead. The geometry entering it belongs to the previous converged
configuration, which makes the scheme staggered and first-order in the increment size -- an error of
the increment, not of the iteration, and one that shrinks with the step rather than with the
tolerance. A formulation that re-segmented every iteration and carried the full linearisation would
remove it, at the price of a considerably more involved tangent.


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
``edelweissfe/elements/contactelement/test_element.py``.

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
the geometry at that node. Measured against the analytical normal of a cylindrical patch resolved by
quadratic elements, the angular error falls by about a factor of eight for every halving of the
facet size, i.e. at third order.

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

One generalisation is this implementation's own. Cichosz and Bischoff define the weighting factor
parametrically, as the reciprocal of :math:`\int N_k \,\mathrm{d}\xi` over the covered interval,
which for their straight two-node elements returns the row to the whole-element value. The ratio of
the two PHYSICAL integrals used here lands on the same value for those elements and extends to a
curved facet, whose Jacobian does not cancel out of a parametric expression.

For a fixed active set the weighting is a rescaling of the constraint and the displacement solution
is identical. Across the active-set decision it is not. The activation test compares the pressure
against :math:`c_n` times the normalised gap; the gap is invariant under the rescaling and the
pressure is not, so a node close to the threshold can switch differently. This is an improvement
rather than a side effect: before the rescaling the quantity entering that comparison at a thinly
covered node was a force divided by a vanishing area, set against a genuine pressure on the other
side of the test. Measured on a curved indenter, three nodes at the edge of the contact zone move,
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

This is the first remedy Cichosz and Bischoff name, and it is the one to reach for whenever the
geometry allows it. The two decks ``MortarContactPunchEdge`` and ``MortarContactPartialOverlap``
are the same configuration with the two choices, and their headers carry the measurements.


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

    \lambda_I = \max\left(0, \; z_I + \kappa \, g^{\text{pen}}_I\right) \operatorname{sgn} D_{II} ,
    \qquad g^{\text{pen}}_I = -g_I \operatorname{sgn} D_{II} ,

with :math:`z_I` the augmentation estimate, zero for pure penalty. The tangent is then the rank-one
:math:`K_{ab} \mathrel{+}= \kappa \, v_a v_b`, symmetric and positive semi-definite, and the system
keeps the size and the definiteness it had without contact.

Note where the nodal weight does NOT appear: the penalty law multiplies the WEIGHTED gap, so no
division by :math:`D_{II}` occurs anywhere in it. That is deliberate. In the alternative form
:math:`\lambda = \varepsilon \, g_I / D_{II}` the weight sits in the denominator of the contact
pressure, and it cannot be assumed to stay away from zero.


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
  the displacement field is 1.5e-04 at :math:`\kappa = 10^6` and falls by exactly one decade per
  decade of stiffness -- 1.5e-05, then 1.5e-06 -- while the contact pressure spreads across the
  interface instead of staying constant. Switching ``augmentedLagrange`` on at the same stiffness
  brings that error to 3e-11, four orders better, and removes the dependence on the stiffness
  altogether. There is little reason to run pure penalty except to see what the augmentation buys.

* ``cn`` enters the activation test only. It is purely algorithmic -- the opening vanishes at an
  active node on convergence, so the converged solution is the same for every admissible value --
  but two of its properties are easy to get wrong. Its UNIT is a stress per length, because it
  multiplies an opening and is added to a pressure; a value that suits a model measured in metres
  is wrong for the same model in millimetres. And the admissible values form a band whose ends fail
  differently: below it the active set never settles and the increment cuts back, far above it the
  set oscillates between two states indefinitely. Left at its default of 0 it is taken as the
  smallest Young's modulus among the materials adjacent to the two surfaces, which sits inside that
  band for a model whose lengths are of order one, and the derived value is reported once.

* ``penaltyStiffness`` carries a unit that is NOT the one a pointwise penalty formulation has. The
  gap it multiplies is the weighted gap and carries length times area, so a value taken from a rule
  of thumb for a classical penalty parameter is wrong by the size of a face. Left at 0 it is derived
  as the smallest adjacent Young's modulus divided by the characteristic face size and the mean
  nodal weight -- a contact stiffness per unit area of one adjacent element layer. That derivation
  is an engineering rule of this implementation rather than an established value, and the derived
  number is reported. With ``augmentedLagrange`` enabled the converged result does not depend on it
  at all; it then only sets how fast the outer loop converges.

* ``augmentationTolerance`` and ``maxAugmentations`` bound the outer loop. The tolerance is a
  RELATIVE change of the pressure estimate from one augmentation to the next, which makes it
  independent of the unit the model is measured in. Reaching the cap is reported; it means the
  increment converged on an estimate that had not settled, and the remedy is a larger cap, a larger
  stiffness, or a smaller increment.

* **Which surface is the non-mortar one** is a parameter in all but name, and on interfaces of
  unequal extent it is the most consequential one on this list. Put the SMALLER surface there. Every
  one of its facets is then fully covered and no boundary row arises at all. With the sides the
  other way round, a surface that ends while still transmitting pressure produces them in numbers:
  measured on a half-width punch, nodes outside the punch take 7 % of the transmitted force, rising
  to 21 % under eightfold refinement. See the section on the choice of sides.

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

**The geometry is frozen once per increment.** The segmentation, the coupling matrices and the
nodal normals are evaluated once, from the last CONVERGED configuration, and held fixed for the
Newton iterations of that increment. The reference is deliberately the converged state and not the
incoming iterate: solvers extrapolate the previous increment before the first assembly, and
freezing at that predictor would make the converged contact solution depend on a solver switch that
must not influence it. A cutback re-attempt resets the frozen state along with everything else,
since the geometry of a diverged attempt belongs to that attempt.

What this costs is the subject of the section on the tangent above. What it buys is that the
contact contribution of one increment is a smooth function of the displacements, with no
re-segmentation happening underneath the iteration.

**The active set is re-decided in every Newton iteration.** This is the semi-smooth Newton method,
and it is what gives the active set superlinear convergence rather than a fixed-point character.
Two safeguards surround it, both necessary and both reported when they fire:

* the set is frozen for the remainder of an increment once it has settled, so that the iteration
  can converge on a fixed set rather than chase a moving one;
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
* ``requiresCorrectionBeforeConvergence`` reports that the constraint has changed its own state
  since the last correction, so the increment may not be declared converged on it. Without this an
  increment can be accepted at iteration 0 on the extrapolated state: the field-correction criterion
  is then satisfied by an ABSENT correction rather than a small one.

**The multipliers are ordinary scalar variables** of the model. They are assembled into the global
system, they appear in restart state, and they are part of what the regression runner compares
against its reference -- which is why a ``formulation=lagrange`` deck checks the contact pressures
and not only the displacements.


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
``MortarContactPatch`` (2D)        1e-16                 2e-16 of 0.02         10.000000000
``MortarContactPatchQuadratic``    1e-16                 6e-16 of 0.02         10.000000000
``MortarContactPatchHexa8``        2e-14                 7e-14 of 2            10.000000000
``MortarContactPatchHexa20``       3e-15                 1e-14 of 0.02         10.000000000
=================================  ====================  ====================  ==================

The lateral constraints of those decks fix the symmetry planes only, never the whole model, so the
lateral displacements are a result of the formulation rather than a boundary condition read back.
``MortarContactPatchHexa8`` works at a hundred times the length scale of the others, which is what
notices a tolerance inside the constraint that compares a length against an absolute bound.

**The unilateral condition** is covered by two further decks. ``MortarContactSeparation`` walks one
model through closing, separating and closing again in three steps: in the separated state every
multiplier is exactly 0 and the non-mortar block is left behind at rest to 1.3e-18, and the
re-closed state is exact again to 8e-16 -- so the active set can empty and refill.
``MortarContactPartialOverlap`` leaves part of the non-mortar surface uncovered and tabulates
weight, coverage, multiplier and force across the interface; ``MortarContactPunchEdge`` is the same
configuration with the favourable choice of sides, where the coverage is exactly 1.000 at every node
and the multipliers run from 9.33 in the interior to 23.01 at the rim.

**The penalty branch and the outer loop** are covered by ``MortarContactPenaltyHexa20``, the
quadratic patch test with the multipliers eliminated. It is the only deck that reaches the two
solver hooks. Its header carries the comparison against the same model with the loop switched off.

**The tangent** is checked against central differences of the assembled forces, with the active set
and the geometry frozen -- across a change of the set there is no tangent to compare, and the
geometry is evaluated once per increment, so the residual this tangent belongs to is the frozen one.
Measured on an interpenetrating, laterally shifted pair of quadratic facets with every node in
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

    Measured on a half-width punch: 7 % of the transmitted force lands on nodes outside the punch,
    and the share GROWS under refinement -- 14 % at twice the resolution, 18 % at four times, 21 % at
    eight. The total force is unaffected, being fixed by equilibrium; what is wrong is where it
    ends up. The constraint itself is satisfied exactly throughout, the normalised gap at such a
    node measuring :math:`-2 \cdot 10^{-16}`, as at every other node.

    Three things bound the problem. It is confined to ``formulation=lagrange``: under penalty the
    pressure is proportional to the penetration, so a sliver carries a sliver's force, and the same
    model misplaces :math:`3 \cdot 10^{-9}` instead of 7 %. It requires the pressure to be at full
    value where the surface ends, so a contact that tapers off between curved bodies is unaffected --
    a cylindrical indenter measured 0 at every refinement. And it does not arise at all when the
    smaller surface is the non-mortar one, which is the remedy named above.

    No treatment of it is implemented. The weighting of the boundary rows does not address it and
    cannot: a rescaling of the constraint leaves the solution invariant, and it is the solution that
    is at issue here, not the conditioning.

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
  is Farah's, for the reason given under *The dual basis*.
* T. Cichosz, M. Bischoff: *Consistent treatment of boundaries with mortar contact formulations
  using dual Lagrange multipliers*, Computer Methods in Applied Mechanics and Engineering 200
  (2011), 1317--1332. -- The partially covered facet at the boundary of the contact area: the
  quadratic decay of its nodal weight, the reciprocal growth of its multiplier and the resulting
  loss of conditioning, and both remedies used here -- biorthogonality over the covered area, and
  the weighting of the boundary rows. Their Section 4 and Eqs. (41) to (45); the choice of sides is
  the advice of their Example 6.1.

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
  converged. Their Fig. 3 is also the decomposition of a quadratic facet into linear sub-cells used
  here: four quadrilaterals for the nine-node patch, four triangles for the six-node patch, and four
  triangles plus one quadrilateral for the serendipity eight-node patch.
* A. Popp, M. W. Gee, W. A. Wall: *A finite deformation mortar contact formulation using a
  primal-dual active set strategy*, International Journal for Numerical Methods in Engineering 79
  (2009), 1354--1391.
* A. Popp, M. Gitterle, M. W. Gee, W. A. Wall: *A dual mortar approach for 3D finite deformation
  contact with consistent linearization*, International Journal for Numerical Methods in Engineering
  83 (2010), 1428--1465. -- The averaged nodal normal.
* P. Farah: *Mortar Methods for Computational Contact Mechanics Including Wear and General Volume
  Coupled Problems*, Dissertation, TU München, 2018. -- The averaged nodal normal written out as
  Eq. (4.41), over the *unit* normals of the adjacent elements; and the transformation parameter
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
  method*, SIAM Journal on Optimization 13 (2002), 865--888.
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
