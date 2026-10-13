#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Manuel Hradsky manuel.hradsky@uibk.ac.at
#
#  This file is part of EdelweissFE.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.models.meshdependent import MeshDependent
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.caseinsensitivedict import CaseInsensitiveDict
from edelweissfe.utils.contactfriction import (
    CoulombFriction,
    characteristicFacetSize,
    frictionCoefficientField,
    matchRestartKeys,
    slipToleranceField,
    tangentPenaltyField,
)
from edelweissfe.utils.mortargeometry import (
    buildBoundingVolumeHierarchy,
    clipLineSegments,
    queryBoundingVolumeHierarchy,
    sutherlandHodgmanClip,
    tangentBasis,
    toPlaneCoordinates,
    triangulatePolygon,
)
from edelweissfe.utils.schema import buildSchemaFromOptions, schemaField

"""
A unilateral contact constraint between two deformable surfaces, enforced segment by segment rather
than node by node. The surfaces are represented by geometry-only contact elements
(:mod:`~edelweissfe.elements.contactelement.element`, typically created via
:mod:`~edelweissfe.generators.surfaceelementgenerator` with ``facets=wholeFace``), one per face of
the underlying solids. One of them is designated the non-mortar side and carries the constraint;
the other is projected onto it.

Where a node-to-surface method asks whether each slave node has penetrated, this one integrates the
non-penetration condition over the overlap of the two surfaces and asks that the result vanish. The
quantity it constrains is therefore not a pointwise gap but a WEIGHTED one, an integral of the gap
against a basis function belonging to a non-mortar node. That is what lets the method transfer a
constant pressure across an interface whose two sides do not share nodes -- a node-to-surface method
cannot, because its weighting is a property of the node rather than of the surface. The price is
that every quantity per node acquires a weight: the nodal mortar weight ``D_II``, the integral of
that node's basis function over its own facets. A weighted gap divided by ``D_II`` is the physical
opening, a multiplier times ``D_II`` is a nodal force, and a node whose weight is zero has no
material opposite it at all.

Over its OWN facets, not over the part of them the other surface covers: a facet that is only partly
covered would otherwise integrate to a weight that vanishes quadratically with the covered area,
while the force that has to cross there vanishes only linearly, and the multiplier -- the ratio of
the two -- would grow without bound. Those rows are therefore rescaled so that their weight is the
whole-facet integral again, which leaves the constraint untouched (both coupling matrices carry the
same factor) and keeps the multiplier a pressure of ordinary magnitude. How much of a node is
actually opposed is reported separately as ``currentCoverage``, since it can no longer be read off
the weights.

The multiplier is discretised in a DUAL basis, biorthogonal to the standard one. This is what keeps
the coupling matrix of the non-mortar side diagonal, so that a nodal multiplier belongs to exactly
one node and the constraint stays local. The dual basis is built per face from the element's own
quadrature rather than taken from a table, because it depends on the face's geometry.

SIGN CONVENTION. ``lambda_I`` is the NEGATIVE traction of the non-mortar surface, so it is positive
in compression, and the force it applies to the node is ``-lambda_I * D_IK * n_I``, directed against
the outward normal. Where the nodal weight is positive -- the regular case -- ``lambda_I`` therefore
IS the nodal contact pressure and may be read as one. Where it is not, ``p_n = lambda_I * sgn(D_II)``
restores that reading. Carrying the sign explicitly matters because it decides a branch rather than
scaling a result: the activation test below sits inside a maximum, where a sign error selects the
wrong branch instead of producing a slightly wrong number.

The Signorini conditions -- non-negative pressure, non-negative gap, and no pressure where there is a
gap -- are enforced through a single non-smooth complementarity function::

    C_n,I = p_I - max(0, p_I - c_n * g_I / D_II) = 0
        <=>  active  iff  s_n,I = p_I - c_n * g_I / D_II > 0

which a semi-smooth Newton method solves by re-deciding the active set in EVERY iteration and
converging the outer loop once the set stops changing. ``c_n`` is purely algorithmic: the gap
vanishes at an active node on convergence, so the converged solution does not depend on it. It does
have to be of the order of a stress divided by a length, and the admissible values form a band --
too small and the active set never settles, too large and it oscillates between two sets forever.
Defaulting it to the smallest Young's modulus adjacent to the interface places it inside that band
for a model whose lengths are of order one.

THREE FORMULATIONS of the same discrete condition are available, differing only in how it is
enforced and sharing the entire geometric pipeline -- search, projection, clipping, segmentation,
nodal normals, and the coupling matrices:

``lagrange``
    The multipliers are unknowns of the global system, which becomes a saddle-point problem. The
    condition is satisfied exactly at convergence.

``penalty``
    The multipliers are eliminated and replaced by a pressure proportional to the penetration. The
    condition is then satisfied only approximately, the error falling in proportion to the penalty
    stiffness. Note the unit: the weighted gap carries length times area, so the stiffness that
    multiplies it is divided by the nodal weight ``D_II``, node by node. The pressure is then
    ``eps_N`` times the pointwise penetration, and ``penaltyStiffness`` is the classical pointwise
    penalty parameter -- the same quantity, in the same unit, as the ``penalty`` of the node- and
    segment-based penalty constraints, chosen by the same rules of thumb.

``penalty`` with ``augmentedLagrange``
    The same, with the pressure estimate corrected in an outer loop that runs after each converged
    equilibrium rather than inside it. This recovers most of the accuracy of the saddle-point form
    without its extra unknowns, and the converged result no longer depends on the penalty stiffness,
    which only sets how fast the outer loop converges.

COULOMB FRICTION (``mu > 0``) is available for ``penalty`` only, with or without the augmentation,
under the implicit and the explicit solvers: a tangential penalty stick predictor from the converged
nodal traction and the mortar-weighted slip increment, and a radial return onto ``|t_T| <= mu p``.
See :meth:`Constraint._assembleFriction` for the law, :meth:`Constraint._carryFrictionHistory` for
the objective correction of the slip at each geometry rebuild, and
:meth:`Constraint._freezeSlipStateIfChattering` for the anti-chattering rule of the implicit
iteration.

RESULTS PER NON-MORTAR NODE are published for output. The constraint registers the node set
``<name>_nonMortarNodes`` (in the order of the contact system), so that the accessors
:meth:`Constraint.getNormalPressures`, :meth:`Constraint.getTangentialTractions`,
:meth:`Constraint.getSlipStates` and :meth:`Constraint.getGaps` can be requested as a
``fromExpression`` field output over that set and exported to Ensight like any nodal result::

    *fieldOutput
    >>fromExpression, name=pN, nSet=contact_nonMortarNodes,
    expression='model.constraints["contact"].getNormalPressures()'

The multipliers of the saddle-point form are NOT condensed out of the system. Doing so correctly
requires a Schur complement of the already assembled non-mortar displacement row -- which carries the
surrounding bulk stiffness, not merely this constraint's own contribution -- and replaces that row
with the linearised constraint. The tempting shortcut, evaluating the multiplier from the current
gap and applying it as a force, is NOT that elimination: it acts as a penalty whose stiffness is the
reciprocal nodal weight, a quantity of order one against a bulk stiffness of order 1e4, and
under-transmits the contact force by more than an order of magnitude on a real problem while still
satisfying a single-state unit test that merely checks that a PRESCRIBED multiplier produces
matching forces. A correct condensation would need the constraint interface to replace rows rather
than only add to them.
"""


@dataclass(frozen=True)
class MortarContactSchema:
    """The options this constraint accepts, owned by this module and never mutated from outside
    it.

    Name resolution, case folding, type coercion and defaults are all applied by
    :func:`~edelweissfe.utils.schema.buildSchemaFromOptions`, so an option is spelled once, here,
    and nowhere else.
    """

    nonMortarSurface: str | None = schemaField(
        description=(
            "The element set of contact elements (CONQUAD4/CONQUAD8/CONLINE2/CONLINE3) forming the "
            "non-mortar (slave) surface, e.g. the '<name>_facets' set of a surfaceElementGenerator run "
            "with facets=wholeFace. This is the side the constraint is enforced on: its nodes carry the "
            "weighted gap and the contact pressure, and the mortar side is projected onto it."
        ),
        dtype=str,
        default=None,
        required=True,
    )

    mortarSurface: str | None = schemaField(
        description=(
            "The element set of contact elements forming the mortar (master) surface. It carries no "
            "unknowns of its own; the contact forces reach it through the projection onto the non-mortar "
            "side. The two sets must be disjoint."
        ),
        dtype=str,
        default=None,
        required=True,
    )

    field: str = schemaField(
        description=("The field this constraint acts on (e.g. displacement)."),
        dtype=str,
        default="displacement",
    )

    cn: float = schemaField(
        description=(
            "Complementarity parameter of the semi-smooth activation test s_n = p_n - cn * g_sep, where "
            "g_sep is the weighted gap divided by the nodal mortar weight, i.e. the physical opening at the "
            "node, and a node counts as in contact where s_n is positive. It is purely algorithmic: the "
            "opening vanishes at an active node on convergence, so the converged solution is the same for "
            "every admissible value, and only the path to it differs. Its UNIT is a stress per length, "
            "because it is added to a pressure after multiplying an opening -- which also means a value that "
            "works for a model measured in metres is wrong for the same model measured in millimetres. The "
            "admissible values form a band whose two ends fail differently: below it the active set never "
            "settles and the increment cuts back, far above it the set oscillates between two states forever. "
            "The order of the Young's modulus of the softer body sits comfortably inside that band for a "
            "model whose lengths are of order one. Leave it at 0, the default, to take the smallest Young's "
            "modulus among the materials adjacent to the two surfaces; the derived value is reported once so "
            "it stays checkable."
        ),
        dtype=float,
        default=0.0,
    )

    formulation: str = schemaField(
        description=(
            "How the non-penetration condition is enforced. 'lagrange', the default, keeps the multipliers "
            "as unknowns of the global system, which becomes a saddle-point problem and satisfies the "
            "condition exactly at convergence. 'penalty' eliminates them in favour of a pressure proportional "
            "to the penetration, which satisfies the condition only to an error falling in proportion to "
            "penaltyStiffness -- unless augmentedLagrange is switched on as well, which recovers the accuracy "
            "without the extra unknowns. Both share the entire geometric pipeline (search, projection, "
            "clipping, segmentation, nodal normals and the coupling matrices) and differ in the constraint "
            "law alone, so their results are directly comparable."
        ),
        dtype=str,
        default="lagrange",
    )

    penaltyStiffness: float = schemaField(
        description=(
            "Stiffness of the penalty pressure, used only for formulation=penalty: the pressure is "
            "penaltyStiffness times the pointwise penetration. It is the classical pointwise penalty "
            "parameter, in the same unit as the penalty of the node- and segment-based penalty constraints, "
            "so a value can be carried between them. Internally the weighted gap (length times area) is "
            "multiplied by penaltyStiffness divided by the nodal mortar weight, node by node; the division is "
            "applied to the stiffness rather than to the gap because that weight cannot be assumed to stay "
            "away from zero on a partially covered quadratic face. Leave it at 0, the default, to derive it as "
            "100 times the smallest adjacent Young's modulus divided by the characteristic face size (under "
            "explicit dynamics, from the stability limit of the time increment instead). That derivation is an "
            "engineering rule of this implementation rather than an established value, and the derived number "
            "is reported once so it stays checkable. With augmentedLagrange enabled the converged result does "
            "not depend on this stiffness at all; it then only sets how fast the outer loop converges."
        ),
        dtype=float,
        default=0.0,
    )

    augmentedLagrange: bool = schemaField(
        description=(
            "Add an outer correction loop on top of formulation=penalty: after each converged equilibrium, "
            "and only then, the nodal pressure estimate is advanced by the penalty stiffness times the "
            "remaining weighted gap, and equilibrium is re-established. OFF by default, so formulation="
            "penalty on its own is the pure penalty form. Switching it on removes the penalty method's "
            "characteristic sensitivity: the converged solution then satisfies the condition to the "
            "augmentation tolerance rather than in proportion to the stiffness, and stops depending on that "
            "stiffness. It needs an equilibrium iteration to nest inside and is therefore unavailable to an "
            "explicit solver. Ignored for formulation=lagrange, which enforces the condition exactly anyway."
        ),
        dtype=bool,
        default=False,
    )

    augmentationTolerance: float = schemaField(
        description=(
            "Termination tolerance of the augmented outer loop, as a RELATIVE change of the nodal pressure "
            "estimate from one augmentation to the next: max|p_new - p_old| <= tol * max|p_new|. The relative "
            "form is used rather than a bound on the remaining gap because it is dimensionless and therefore "
            "unaffected by the length unit the model is measured in. Its limitation follows from the same "
            "property: where the penalty term alone already satisfies the condition to rounding, the estimate "
            "stops moving, both sides of the comparison fall to noise together, and the ratio no longer "
            "decreases -- the loop then runs to maxAugmentations although it has in fact converged."
        ),
        dtype=float,
        default=1e-06,
    )

    maxAugmentations: int = schemaField(
        description=(
            "Safeguard cap on the number of augmented-Lagrangian outer iterations per increment. Reaching it "
            "is reported: the increment then converged on a pressure estimate that had not settled, so its "
            "constraint violation is whatever the last augmentation left - larger than the tolerance asked "
            "for."
        ),
        dtype=int,
        default=10,
    )

    mu: float = frictionCoefficientField(
        " Used only for formulation=penalty, at every non-mortar node, with the mortar-weighted "
        "tangential slip (Yang, Laursen & Meng 2005; Puso, Laursen & Solberg 2008); it works under the "
        "implicit and the explicit solvers alike."
    )

    tangentPenalty: float = tangentPenaltyField(
        "penaltyStiffness",
        " The default follows eps_N as resolved, including its derivation from the stability limit under "
        "explicit dynamics.",
    )

    slipTolerance: float = slipToleranceField()


def mapPlaneToNatural(el, coords_2d, point_2d, max_iter=10, tol=1e-12):
    """Map a 2D local plane coordinate point_2d to the element's natural space.

    Returns (local_coords, converged).

    Termination. This is a (Gauss-)Newton solve for the natural coordinate, so the
    quantity it actually drives to zero is the UPDATE, not the residual. For 2D
    surface elements the two coincide as long as the point lies in the element's
    plane - which it does, since both facets are projected into the same auxiliary
    plane. For 1D line elements in 2D space they do NOT: the Gauss point is
    generated on the SLAVE line and then mapped into the MASTER element, so with an
    open contact gap g the smallest attainable residual is exactly g and a residual
    test can never be satisfied. The least-squares step nevertheless returns the
    exact orthogonal projection after the first pass; without an update-based test
    the loop just burns all max_iter passes to return it. Measured on the test
    suite before this criterion was added: 189 of 780476 calls "failed" this way,
    every one of them a 2D call at an open gap, every returned coordinate exact.

    The update test is also the scale-invariant one - the natural coordinate is
    dimensionless, whereas the residual carries the model's length unit.

    `converged` is True when either criterion was met. It is False only when the
    iteration ran out of passes or hit a singular Jacobian - i.e. when the returned
    coordinate is the last iterate rather than a solution. Nothing downstream can
    tell those apart on its own (the value looks like any other), so the caller
    counts them and reports them; see the `gp_projection_failed` diagnostic in
    computeMortarCouplingMatrices. That the closest-point projection need not be
    solvable at all, and under which conditions it fails, is the subject of
    Konyukhov & Schweizerhof (2008); the local Newton used here is the standard one
    of the surrounding solids.

    Implementation note. The linear algebra is written out in scalars instead of
    calling np.linalg.norm / np.linalg.solve. Those are 2x2 and 2-vector operations
    called several hundred thousand times per increment, where the numpy dispatch
    overhead is many times the arithmetic (measured on hertz_hex20_fine: 5.0 s in
    np.linalg.solve and 3.3 s in np.linalg.norm out of 31 s of segmentation).

    This is NOT bit-identical to what it replaces, and the difference was measured
    rather than assumed. The 2x2 solve follows LAPACK dgesv's operation order -
    partial pivoting with IDAMAX tie-breaking towards the first row, the reciprocal
    of the pivot formed once as dgetf2 does, then forward/back substitution - and
    still differs from np.linalg.solve for 39 % of random 2x2 systems, by up to
    5.5e-15 relative (dividing instead of using the reciprocal is worse: 9.9e-12).
    math.sqrt(r0*r0 + r1*r1) differs from np.linalg.norm for 8 % of random 2-vectors,
    by 1 ulp. Propagated through the Newton iteration and the segment quadrature,
    that shows up as ~1e-12 relative in D and C and ~2e-10 in the converged
    multipliers, with the active set unchanged (hertz_hex20_medium). The whole
    verification suite passes with unchanged tolerances, but this is a deliberate
    trade of exact reproducibility against roughly 10 % wall clock - if the
    reproducibility matters more, this function is the single place to revert.
    """
    el_type = el.elType.upper()

    # 1D line elements
    if "LINE" in el_type:
        xi = 0.0
        local_coords = np.zeros(1)
        for _ in range(max_iter):
            local_coords[0] = xi
            N = el.getShapeFunctions(local_coords)
            x_mapped = N @ coords_2d
            r0 = x_mapped[0] - point_2d[0]
            r1 = x_mapped[1] - point_2d[1]
            if math.sqrt(r0 * r0 + r1 * r1) < tol:
                return local_coords, True
            dN = el.getShapeFunctionDerivatives(local_coords)  # shape (1, n_nodes)
            J = dN @ coords_2d  # shape (1, dim_of_coords_2d)
            j0, j1 = J[0, 0], J[0, 1]
            J_norm = j0 * j0 + j1 * j1
            if J_norm < 1e-14:
                return local_coords, False
            delta = (j0 * r0 + j1 * r1) / J_norm
            xi -= delta
            local_coords[0] = xi
            if abs(delta) < tol:
                return local_coords, True
        return local_coords, False

    # 2D surface elements (Quads and Triangles)
    if "TRI" in el_type:
        xi = eta = 1.0 / 3.0
    else:
        xi = eta = 0.0
    local_coords = np.array([xi, eta])

    for _ in range(max_iter):
        local_coords[0] = xi
        local_coords[1] = eta
        N = el.getShapeFunctions(local_coords)
        x_mapped = N @ coords_2d
        r0 = x_mapped[0] - point_2d[0]
        r1 = x_mapped[1] - point_2d[1]
        if math.sqrt(r0 * r0 + r1 * r1) < tol:
            return local_coords, True
        dN = el.getShapeFunctionDerivatives(local_coords)  # shape (2, n_nodes)
        Jm = dN @ coords_2d  # (2, 2); the Jacobian is its transpose
        a00, a01 = Jm[0, 0], Jm[1, 0]
        a10, a11 = Jm[0, 1], Jm[1, 1]

        # 2x2 LU with partial pivoting, in LAPACK dgesv's operation order.
        if abs(a10) > abs(a00):  # IDAMAX picks the first maximum, so ties keep row 0
            a00, a01, a10, a11 = a10, a11, a00, a01
            b0, b1 = r1, r0
        else:
            b0, b1 = r0, r1
        if a00 == 0.0:
            return local_coords, False  # singular column: np.linalg.solve would raise
        # dgetf2 forms the RECIPROCAL of the pivot once and multiplies by it; that is
        # not the same in floating point as dividing, so dividing here would already
        # cost bit-identity with np.linalg.solve.
        m = a10 * (1.0 / a00)
        u11 = a11 - m * a01
        if u11 == 0.0:
            return local_coords, False
        y1 = b1 - m * b0
        d1 = y1 / u11
        d0 = (b0 - a01 * d1) / a00

        xi -= d0
        eta -= d1
        local_coords[0] = xi
        local_coords[1] = eta
        if math.sqrt(d0 * d0 + d1 * d1) < tol:
            return local_coords, True
    return local_coords, False


# Decomposition of contact facets into linear sub-cells for the mortar
# segmentation (polygon clipping). The clip polygons must be simple convex
# polygons, so curved (quadratic) facets are subdivided into linear cells
# using their mid-side nodes before the clipping, which is what the segment-based
# integration needs. The curvature of the facet enters only
# through the shape function evaluation at the mapped Gauss points, never
# through the clipping geometry itself.
#
# The three quadratic decompositions below are those of Puso, Laursen & Solberg
# (2008), Fig. 3: the nine-node patch into four quadrilaterals (b), the six-node
# patch into four linear triangles (d), and the serendipity eight-node patch into
# "four linear triangle sub-segments and one quadrilateral segment" (f). They note
# what this costs and why it is acceptable: the geometrically exact description of
# the quadratic surface is given up for the integration, but "through consideration
# of simple affine maps between the parameterizations for these linear sub-segments
# and their parent quadratic elements, we retain the ability to easily integrate
# inner products of higher order shape functions".
_subCellMap = {
    "CONLINE2": [[0, 1]],
    "CONLINE3": [[0, 2], [2, 1]],
    "CONQUAD4": [[0, 1, 2, 3]],
    "CONTRI3": [[0, 1, 2]],
    "CONQUAD8": [[0, 4, 7], [4, 1, 5], [5, 2, 6], [7, 6, 3], [4, 5, 6, 7]],
    "CONQUAD9": [[0, 4, 8, 7], [4, 1, 5, 8], [8, 5, 2, 6], [7, 8, 6, 3]],
    "CONTRI6": [[0, 3, 5], [3, 4, 5], [3, 1, 4], [5, 4, 2]],
}


def _cooToCsr(vals: list, rows: list, cols: list, shape: tuple) -> csr_matrix:
    """Build a CSR matrix from lists of per-block triples, summing duplicates."""
    if not vals:
        return csr_matrix(shape)
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=shape,
    ).tocsr()


def _nonzeroRows(A: csr_matrix, n_rows: int, rtol: float = 1e-12) -> tuple[list, list]:
    """Per-row (column indices, values) of a CSR matrix, filtered RELATIVE to max|A|.

    The threshold is part of the formulation's arithmetic, not of the storage: it
    decides which coupling terms enter the weak gap and the stiffness, and a stored
    CSR entry can be below it (coo -> csr sums duplicates, it does not prune).

    It is deliberately a RELATIVE threshold. The entries of D and C are integrals
    int(Phi_I N_K) and therefore carry the unit of a facet AREA, so any absolute
    bound is a statement about the model's length unit rather than about the
    arithmetic. What has to be separated here is round-off dust - which is
    ~eps*max|A| by construction, independent of the unit - from a genuine coupling
    term, and rtol*max|A| with rtol = 1e-12 sits four orders above the dust and
    twelve below the signal for ANY length scale. (The bound this replaces was an
    absolute 1e-14; on a model of order one it is the same filter to within an order
    of magnitude, which is why the measured values in the documentation are
    unchanged.)

    Caching the VALUES next to the indices - rather than re-indexing A[I, nz] on every
    Newton iteration - is the other half of the point: the pattern and the values are
    fixed for the whole increment (frozen geometry), while applyConstraint runs once
    per iteration.
    """
    idx_per_row, val_per_row = [], []
    indptr, indices, data = A.indptr, A.indices, A.data
    # An empty matrix has no scale; the loop below then keeps nothing, which is right.
    tol = rtol * np.max(np.abs(data)) if data.size else 0.0
    for I in range(n_rows):  # noqa: E741 - I is the non-mortar node index of the formulation
        lo, hi = indptr[I], indptr[I + 1]
        cols, vals = indices[lo:hi], data[lo:hi]
        keep = np.abs(vals) > tol
        idx_per_row.append(cols[keep])
        val_per_row.append(vals[keep])
    return idx_per_row, val_per_row


def isConvexPolygon(poly_2d, tol: float = 1e-14) -> bool:
    """Whether a planar polygon given in order is convex.

    BOTH sub-cells of a segmentation pair have to be convex, for two different
    reasons:

    slave (clip window)   Sutherland-Hodgman clips against the half-plane of each
        clip edge, and the intersection of those half-planes is the polygon itself
        only if the polygon is convex. A reflex vertex therefore removes area that
        genuinely belongs to the facet, silently (measured: -2/3 of the area for a
        constructed reflex quad, see 02_polygon_clipping).

    master (subject)      Sutherland-Hodgman itself accepts any subject polygon
        ("applicable to any polygon, convex or concave", Sutherland & Hodgman 1974,
        p. 33), but the overlap it returns inherits the subject's reflex vertices,
        and `triangulatePolygon` fans from vertex 0 with a per-triangle |area|.
        For a non-convex overlap those fan triangles reach outside the polygon and
        the area is OVER-counted (measured: 0.375 instead of 0.125, i.e. +200 %).

    Triangles cannot be non-convex, so the requirement only bites on the middle
    quad of a CONQUAD8 and the four quads of a CONQUAD9.
    """
    n = len(poly_2d)
    if n < 4:
        return True
    sign = 0.0
    for i in range(n):
        a = poly_2d[(i + 1) % n] - poly_2d[i]
        b = poly_2d[(i + 2) % n] - poly_2d[(i + 1) % n]
        cross = a[0] * b[1] - a[1] * b[0]
        if abs(cross) < tol:
            continue  # collinear vertex: neither convex nor reflex
        if sign == 0.0:
            sign = cross
        elif cross * sign < 0.0:
            return False
    return True


def isInsideReferenceElement(el, local_coords, tol: float = 1e-10) -> bool:
    """Whether a natural coordinate lies in the element's reference domain (up to ``tol``).

    Used to catch a Gauss point that the segmentation attributed to a facet which does not
    contain it. The clipping works on the straight sub-cells of a facet (see _subCellMap), so
    wherever a quadratic facet has a CURVED edge the region between the chord and the true edge
    belongs to one facet geometrically and to its neighbour in the clipping. Mapped back into the
    wrong facet, such a point lands outside the reference domain and its shape functions are
    extrapolated rather than interpolated.
    """
    el_type = el.elType.upper()
    if "TRI" in el_type:
        xi, eta = local_coords[0], local_coords[1]
        return xi >= -tol and eta >= -tol and xi + eta <= 1.0 + tol
    return bool(np.all(np.abs(local_coords) <= 1.0 + tol))


def getSubCells(el) -> list[list[int]]:
    """Return the linear sub-cell decomposition (local node indices) of a contact facet."""
    el_type = el.elType.upper()
    if el_type not in _subCellMap:
        raise NotImplementedError(f"No linear sub-cell decomposition defined for element type '{el_type}'.")
    return _subCellMap[el_type]


def facetNormal(coords: np.ndarray) -> np.ndarray:
    """Unnormalized area-weighted normal of a flat linear facet (3 or 4 corner nodes).

    Used for the AUXILIARY PLANE of the segmentation only - one plane per slave
    sub-cell, and the sub-cells are linear by construction, so this is the exact
    normal of their plane rather than an approximation.

    It is deliberately NOT what computeNormals builds the nodal normal n_I from:
    one constant direction per facet cannot distinguish a mid-side node from its
    corners, which costs an order of convergence on a curved slave surface. Note
    also that its length is the facet AREA, whereas computeNormals sums unit
    normals and so carries no area weighting at all. See computeNormals.
    """
    if len(coords) == 3:
        return 0.5 * np.cross(coords[1] - coords[0], coords[2] - coords[0])
    return 0.5 * np.cross(coords[2] - coords[0], coords[3] - coords[1])


# 3-point Gauss-Legendre rule (degree 5) on [-1, 1] for 1D line segments
_lineGaussPoints = np.array(
    [
        [-np.sqrt(0.6)],
        [0.0],
        [np.sqrt(0.6)],
    ]
)
_lineGaussWeights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


# 7-point symmetric Gauss rule (degree 5) on the reference triangle, the rule
# tabulated by Dunavant (1985) for p = 5, n_g = 7, written here in closed form.
# 7 points per integration cell are enough here,
# since the nonlinear projection between the auxiliary plane and the curved
# element surfaces raises the polynomial degree of the integrand. That is also the
# experience of Puso & Laursen (2004), Section 3.1, who report that "seven point
# integration was used for most example problems and appears to be more than
# sufficient when compared to higher order nine and thirteen point schemes" -- with
# the caveat, which holds here too, that "the choice of integration rule is somewhat
# problem dependent".
_triangleGaussA = (6.0 - np.sqrt(15.0)) / 21.0
_triangleGaussB = (6.0 + np.sqrt(15.0)) / 21.0
_triangleGaussPoints = np.array(
    [
        [1.0 / 3.0, 1.0 / 3.0],
        [_triangleGaussA, _triangleGaussA],
        [_triangleGaussA, 1.0 - 2.0 * _triangleGaussA],
        [1.0 - 2.0 * _triangleGaussA, _triangleGaussA],
        [_triangleGaussB, _triangleGaussB],
        [_triangleGaussB, 1.0 - 2.0 * _triangleGaussB],
        [1.0 - 2.0 * _triangleGaussB, _triangleGaussB],
    ]
)
_triangleGaussWeights = np.array(
    [
        9.0 / 80.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 - np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
        (155.0 + np.sqrt(15.0)) / 2400.0,
    ]
)


class Constraint(ConstraintBase, MeshDependent):
    """A segment-to-segment mortar contact constraint between two deformable surfaces.

    The module docstring above states the formulation, its sign convention and the three ways the
    discrete condition can be enforced; :doc:`the mortar theory page </documentation/mortartheory>`
    gives the derivation and the sources. What follows is the object contract.

    Both surfaces are element sets of geometry-only contact elements
    (:mod:`~edelweissfe.elements.contactelement.element`), usually produced by
    :mod:`~edelweissfe.generators.surfaceelementgenerator` with ``facets=wholeFace``. The non-mortar
    set carries the constraint; under ``formulation=lagrange`` it also contributes one scalar
    multiplier variable per non-mortar node to the global system, which is what makes that
    formulation a saddle-point problem.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model tree.
    journal
        The journal to report diagnostics to. ``None`` is accepted, for a constraint built directly
        rather than from an input file; diagnostics then fall back to :mod:`warnings`.
    kwargs
        The options of :class:`MortarContactSchema`, by their input-file names. Resolution, case
        folding, coercion and defaults are applied here, so that a direct call and the route through
        :meth:`fromConstraintDefinition` see identical rules.
    """

    #: Option schema for this constraint, per OptionSchemaProvider.
    schema = MortarContactSchema

    #: Sender of this constraint's journal messages. A short category rather than the instance name,
    #: which the messages carry in their own text, so that several contact pairs share one column.
    identification = "MortarContact"

    def __init__(self, name: str, model: FEModel, journal: Journal = None, **kwargs):
        super().__init__(name, model, **kwargs)

        self.journal = journal
        self.model = model
        # AMR: rebuilt from the regenerated facets at the next refresh (see refresh).
        self._lastSeenTopologyVersion = model.topologyVersion
        model.registerMeshDependent(self)
        # Resolving names, coercing values and filling in defaults is the schema's job, done once
        # here so that a direct call from a test and the route through
        # fromConstraintDefinition see the same rules.
        kwargs = CaseInsensitiveDict(vars(buildSchemaFromOptions(self.schema, kwargs)))

        self._name = name
        # Runtime diagnostics. Initialized early so that the input checks further
        # down can already use them.
        self._warned = set()
        self.field = kwargs["field"]
        self.sizeField = getFieldSize(self.field, model.domainSize)

        non_mortar_surf_name = kwargs["nonMortarSurface"]
        mortar_surf_name = kwargs["mortarSurface"]

        self._nonMortarSurfaceSetName = non_mortar_surf_name
        self._mortarSurfaceSetName = mortar_surf_name

        self._identifySurfaces(model)

        # ------------------------------------------------------------------
        # FORMULATION. Two ways to enforce the same discrete constraint g_A = 0,
        # sharing the entire geometric pipeline above and below:
        #   'lagrange'  saddle point, the nodal multipliers are unknowns of the
        #               global system.
        #   'penalty'   the multipliers are eliminated and follow the weighted gap,
        #               a pressure proportional to the weighted gap, optionally
        #               corrected inside an augmented Lagrangian outer loop.
        # Only the LAGRANGE branch needs additional scalar unknowns.
        # ------------------------------------------------------------------
        self.formulation = str(kwargs["formulation"]).strip().lower()
        if self.formulation not in ("lagrange", "penalty"):
            raise ValueError(
                f"MortarContact '{name}': unknown formulation '{kwargs['formulation']}'. "
                f"Valid choices are 'lagrange' (default) and 'penalty'."
            )

        self._updateFootprint()
        self.active = True

        # Semi-smooth-Newton complementarity parameter of the normal contact NCP
        # of the semi-smooth activation test. Purely algorithmic,
        # but it carries a UNIT, so a fixed numeric default is only ever right for one
        # system of units. A non-positive value therefore means "derive it", which is
        # done lazily at the first assembly: sections are assigned to elements in
        # model.prepareYourself(), i.e. after the constraints are constructed, so the
        # materials are not reachable yet at this point.
        self.c_n = float(kwargs["cn"])
        self._derive_c_n = self.c_n <= 0.0

        # Penalty parameter kappa of t_A = kappa*g_A. Like c_n it carries a unit, so
        # a non-positive value means "derive it" - here not at the first assembly but
        # inside the geometry block, because the rule needs the nodal weights.
        # eps_N, a POINTWISE penalty modulus: a pressure per unit opening, the same
        # quantity (and the same unit) as the ``penalty`` of the two node/segment penalty
        # constraints. It is NOT the kappa of the weighted form -- see _resolveEpsN and
        # the "Penalty form" section of mortartheory.rst for why that form was left.
        self.epsN = float(kwargs["penaltyStiffness"])
        self._derive_epsN = self.epsN <= 0.0
        self.useAugmentedLagrange = bool(kwargs["augmentedLagrange"])
        self.augmentationTolerance = float(kwargs["augmentationTolerance"])
        self.maxAugmentations = int(kwargs["maxAugmentations"])

        # Coulomb friction, penalty branch only. eps_T <= 0 means "follow eps_N", resolved
        # lazily through _tangentPenaltyStiffness because eps_N itself may be derived only at the
        # first geometry rebuild -- and, under explicit dynamics, re-derived at every one.
        # The law and the per-node frictional state are the ones shared by all contact formulations
        # (edelweissfe.utils.contactfriction); what is mortar-specific is how a non-mortar node maps
        # onto it (see _assembleFriction) and the objective slip correction at a geometry rebuild
        # (see _carryFrictionHistory). The elastic slip of slipTolerance takes h from the mortar
        # weights at the first geometry rebuild (see _refreshFrozenGeometry).
        self._friction = CoulombFriction(
            self.nNonMortarNodes,
            model.domainSize,
            float(kwargs["mu"]),
            float(kwargs["tangentPenalty"]),
            float(kwargs["slipTolerance"]),
            owner=name,
        )
        self.mu = self._friction.mu
        self._epsTGiven = self._friction.tangentPenalty
        self.slipTolerance = self._friction.slipTolerance
        if self.mu > 0.0 and self.formulation != "penalty":
            raise ValueError(
                f"MortarContact '{name}': friction (mu > 0) is implemented for formulation=penalty only. "
                f"Set formulation=penalty."
            )
        self.friction = self._friction.enabled

        # A parameter that belongs to the OTHER formulation is silently inert, which
        # is exactly the kind of thing that costs an afternoon. Say it once.
        if self.formulation == "penalty" and float(kwargs["cn"]) > 0.0:
            self._warnOnce(
                "cn_ignored_in_penalty",
                "'cn' was given but formulation=penalty ignores it: the complementarity "
                "parameter belongs to the semi-smooth NCP of the Lagrange-multiplier "
                "branch. The penalty branch takes its regularization from "
                "'penaltyStiffness' instead.",
            )
        if self.formulation == "lagrange" and float(kwargs["penaltyStiffness"]) > 0.0:
            self._warnOnce(
                "penalty_ignored_in_lagrange",
                "'penaltyStiffness' was given but formulation=lagrange ignores it: the "
                "multipliers are unknowns there and the constraint is enforced exactly. "
                "Set formulation=penalty to use it.",
            )

        self._deriveNodalGeometry()

        # Initialize PDASS variables
        self.activeSet = np.zeros(self.nNonMortarNodes, dtype=bool)
        self.useActiveSet = True
        # Identifies the increment ATTEMPT (number, size, end time) - see
        # applyConstraint: a cutback re-attempt keeps the number but changes size.
        self.lastTimeStepKey = None
        # Whether the frozen geometry (segmentation, coupling matrices, nodal normals)
        # has to be rebuilt before it is used again. Set by updateConnectivity, which
        # the solver ticks at ITS own cadence: once per increment under the implicit
        # solvers, and only every 'contact-update-frequency' increments under explicit
        # dynamics, where a rebuild per time step would be unaffordable. Starts True so
        # that the first assembly builds it even if no solver ticked first.
        self._geometryStale = True
        # Explicit dynamics: set on the first call through applyConstraintExplicit, which no
        # implicit solver ever makes. The lumped mass arrives separately, through
        # computeCriticalTimeStepForExplicitDynamics, before the increment loop.
        self._explicit = False
        self._explicitLumpedMass = None
        self._explicitStabilityFactor = None
        self.currentIteration = 0
        # Termination + anti-cycling state of the semi-smooth (PDASS) iteration,
        # reset per increment in applyConstraint: the set is frozen for the rest
        # of the increment once it has settled, or once a discrete state already
        # visited this increment recurs.
        self.activeSetFrozen = False
        self.activeSetStableCount = 0
        self._seen_states = set()
        self._last_state = None

        # ---------------- state of the penalty / augmented branch ----------------
        # augmentedMultipliers is the nodal pressure estimate p^k_A of the augmented Lagrangian
        # of the augmented outer loop, in the PHYSICAL convention
        # p >= 0 in compression. It is zero for pure penalty, where the pressure is
        # kappa*g_A alone. augmentedMultipliersConverged carries it across increments (warm start)
        # and is what a cutback re-attempt is reset to.
        self.augmentedMultipliers = np.zeros(self.nNonMortarNodes)
        self.augmentedMultipliersConverged = np.zeros(self.nNonMortarNodes)
        self._augmentation_counter = 0
        # Last assembled weighted gap and nodal multiplier per slave node. The gap is
        # what the augmentation updates from; nodalMultipliers exists so that BOTH
        # formulations expose the contact pressure the same way - in the penalty
        # branch it is not a degree of freedom and could not be read off the
        # solution vector otherwise.
        self.currentWeakGap = np.zeros(self.nNonMortarNodes)
        self.nodalMultipliers = np.zeros(self.nNonMortarNodes)

        # ---------------- mortar-specific state of the friction law (mu > 0) ----------------
        # The rest of the frictional state (tractions, pressures, stick/slip state, anti-chattering)
        # is held by self._friction and exposed through the properties below.
        # The pointwise weighted gap VECTOR of every non-mortar node, sum_K D_IK x_K - sum_J C_IJ x_J
        # over |D_II|, as of the last geometry rebuild; NaN where it was not defined. Its change from
        # one rebuild to the next is what turns the frozen slip into the objective one.
        self.frictionGapReference = np.full((self.nNonMortarNodes, model.domainSize), np.nan)

        self._publishNonMortarNodeSet(model)

        # NOTE: self._warned is initialized at the very top of __init__, because the
        # input checks above already emit diagnostics through it. Every assumption
        # behind those diagnostics was measured over the Control_Tests and
        # patch-test suite before being wired up; the counts are recorded in the
        # documentation.

    def _identifySurfaces(self, model: FEModel):
        """Collect the contact elements of both sides and the non-mortar and mortar nodes from the
        two element sets. Called at construction and again after an AMR refinement regenerated the
        facets (see :meth:`refresh`)."""

        self.nonMortarFacets = self._collectFacets(model, self._nonMortarSurfaceSetName, "Non-mortar")
        self.mortarFacets = self._collectFacets(model, self._mortarSurfaceSetName, "Mortar")

        # Identify nodes. First-seen order, so the node numbering of the contact system is a
        # function of the facet set alone and not of dictionary iteration order.
        self.nonMortarNodes = []
        for el in self.nonMortarFacets:
            for node in el.nodes:
                if node not in self.nonMortarNodes:
                    self.nonMortarNodes.append(node)

        self.mortarNodes = []
        for el in self.mortarFacets:
            for node in el.nodes:
                if node not in self.mortarNodes:
                    self.mortarNodes.append(node)

        # The formulation assumes a fixed pair of DISJOINT surfaces. A node listed
        # on both of them is not merely questionable input, it makes the system
        # singular: for coinciding facets C = D holds exactly, hence the weak gap
        # g_I = -sum_K D_IK (x_K.n) + sum_J C_IJ (x_J.n) vanishes identically for
        # every configuration, and the corresponding lambda row of K cancels out
        # the moment such a node becomes active. Without this check the user only
        # sees an unintelligible linear-solver failure.
        # NOTE: this rules out MISDECLARED surfaces, not self-contact. Genuine
        # self-contact would additionally require an exclusion rule for a facet's
        # own and adjacent facets, an unambiguous nodal normal at doubly
        # classified nodes and a dynamic surface pairing; it is out of scope here.
        shared_nodes = set(self.nonMortarNodes) & set(self.mortarNodes)
        if shared_nodes:
            labels = sorted(node.label for node in shared_nodes)
            raise ValueError(
                f"MortarContact '{self._name}': non-mortar surface '{self._nonMortarSurfaceSetName}' and mortar "
                f"surface '{self._mortarSurfaceSetName}' share {len(labels)} node(s): {labels}. "
                f"The two contact surfaces must be disjoint - coinciding facets yield an "
                f"identically vanishing weak gap and thus a singular system."
            )

        self._nodes = self.nonMortarNodes + self.mortarNodes
        self.nNonMortarNodes = len(self.nonMortarNodes)
        self.nMortarNodes = len(self.mortarNodes)

    def _updateFootprint(self):
        """The degrees of freedom of the contact system: the field on every contact node, plus one
        multiplier per non-mortar node under formulation=lagrange."""

        self.nMultipliers = self.nNonMortarNodes if self.formulation == "lagrange" else 0
        self._nDof = self.sizeField * len(self._nodes) + self.nMultipliers
        self._fieldsOnNodes = [[self.field]] * len(self._nodes)

    def _deriveNodalGeometry(self):
        """Everything derived from the contact nodes in the undeformed configuration: the index
        lookups, the reference coordinates, the interface diameter, the undeformed normals, and the
        orientation checks of the two surfaces."""

        # Node index lookups for fast access
        self.nodeToGlobalIndex = {node: i for i, node in enumerate(self._nodes)}
        self.nonMortarNodeToIndex = {node: i for i, node in enumerate(self.nonMortarNodes)}
        self.mortarNodeToIndex = {node: i for i, node in enumerate(self.mortarNodes)}

        # Undeformed coordinates of all constraint nodes (slaves first, then masters)
        self._X = np.array([node.coordinates for node in self._nodes])

        # Largest extent of the non-mortar surface in any coordinate direction, taken once from the
        # undeformed configuration. It is the length this constraint measures other lengths against
        # wherever an absolute bound would otherwise have to be invented - the noise floor of the
        # weighted gap, and the margin of the orientation check. Taking it undeformed is deliberate:
        # a tolerance that moved with the solution would make a state's admissibility depend on how
        # it was reached.
        slaveCoordinates = self._X[: self.nNonMortarNodes]
        self._interfaceDiameter = float(np.max(np.ptp(slaveCoordinates, axis=0))) if len(slaveCoordinates) else 0.0

        # Precompute undeformed normals
        self.undeformedNormals = self.computeNormals()

        # The orientation of the contact facets is an INPUT property: it decides
        # the direction of n_I and with it the sign of gap and pressure. Nothing
        # downstream can recover from getting it wrong - a slave normal pointing
        # into its own body turns the contact into an adhesive bond that transmits
        # tension. Two independent checks, both diagnostic rather than fatal,
        # because a legitimate mesh may be split, non-manifold at its border, or
        # deliberately one-sided.
        self._checkFacetWinding(self.nonMortarFacets, self._nonMortarSurfaceSetName)
        self._checkFacetWinding(self.mortarFacets, self._mortarSurfaceSetName)
        self._checkSurfacesFaceEachOther()

    def _warnOnce(self, key: str, message: str):
        """Emit a runtime diagnostic at most once per constraint instance and cause.

        These conditions repeat every increment once they occur at all, so warning
        per occurrence would bury the message in its own repetitions. One message
        per run and cause is what makes it readable.

        Parameters
        ----------
        key
            The cause, which is what is deduplicated on.
        message
            The text, which already names the condition and what it implies.
        """
        if key in self._warned:
            return
        self._warned.add(key)
        text = f"contact '{self._name}': {message}"
        if self.journal is not None:
            self.journal.message(text, self.identification, level=1)
        else:
            # No journal: a constraint constructed directly rather than from an input file. The
            # diagnostic is worth more than the channel, so it still goes out.
            warnings.warn(f"MortarContact {text}", RuntimeWarning, stacklevel=3)

    def _publishNonMortarNodeSet(self, model: FEModel):
        """Register the non-mortar nodes, in the order of the contact system, as ``<name>_nonMortarNodes``.

        This is the set the per-node accessors (:meth:`getNormalPressures` and its siblings) line up
        with, so that a ``fromExpression`` field output over it -- and an Ensight ``>>perNode`` export
        of that output -- needs no reordering. Ensight also gives every node set its own part, so the
        contact zone shows up in a post-processor without any field output at all.

        A set of that name that already exists is left alone and reported: it may be the user's own,
        and overwriting it would silently change what another output refers to.
        """

        setName = f"{self._name}_nonMortarNodes"
        self.nonMortarNodeSetName = setName
        nodeSets = getattr(model, "nodeSets", None)
        if nodeSets is None:
            return
        if getattr(self, "_ownsNonMortarNodeSet", False):
            # Ours, published at construction: brought back into the order of the contact system
            # after a refinement (which also rewrites every tracked node set, this one included).
            current = nodeSets.get(setName)
            if current is None:
                nodeSets[setName] = NodeSet(setName, list(self.nonMortarNodes))
            else:
                current.replaceMembers(list(self.nonMortarNodes))
            return
        if setName in nodeSets:
            self._warnOnce(
                "node_set_exists",
                f"the node set '{setName}' already exists, so the non-mortar nodes are not published under "
                f"that name. A fromExpression output of this constraint's nodal results needs a set in the "
                f"order of the contact system; rename the existing set to get one.",
            )
            return
        nodeSets[setName] = NodeSet(setName, list(self.nonMortarNodes))
        self._ownsNonMortarNodeSet = True

    # ------------------------------------------------------------------ AMR

    def refresh(self, model: FEModel, change) -> bool:
        """Rebuild the contact system from the regenerated facets after an AMR refinement touched
        either surface, carrying every nodal state over by ``Node`` identity.

        The surface element generator regenerates ALL facets of a refined surface as new elements
        (``facets=wholeFace`` included), on the solid's own ``Node`` objects -- so a node that
        survives the refinement is the same object before and after, and a new node is one that did
        not exist. What is carried over for a retained non-mortar node: the frictional history
        (traction, pressure, contact status), the augmented pressure estimate, the reference gap
        vector of the objective slip correction, and under formulation=lagrange its converged
        multiplier. A new node starts at rest. Everything geometric -- segmentation, coupling
        matrices, normals, tolerances -- is invalidated and rebuilt at the next assembly from the
        last converged configuration, exactly as at a regular increment.

        Under formulation=lagrange the number of multipliers follows the non-mortar nodes, so the
        constraint's scalar variables are re-created (see
        :meth:`~edelweissfe.models.femodel.FEModel.reassignScalarVariablesOfConstraint`).

        Returns True if either surface was touched: the DOF footprint changed, so the solver must
        rebuild its equation system (:meth:`updateConnectivity` alone never says so).
        """

        recipes = getattr(model, "contactFacetRecipes", {})
        touched = False
        for setName in (self._nonMortarSurfaceSetName, self._mortarSurfaceSetName):
            recipe = recipes.get(setName)
            if recipe is not None and change.touchesSurface(recipe[0]):
                touched = True
            if setName in getattr(change, "changedElementSets", ()):
                touched = True
        if not touched:
            return False

        oldNonMortarNodes = list(self.nonMortarNodes)
        oldMultipliers = (
            np.array([variable.value for variable in self.scalarVariables])
            if self.formulation == "lagrange" and len(self.scalarVariables) == len(oldNonMortarNodes)
            else None
        )

        self._identifySurfaces(model)
        self._updateFootprint()
        self._invalidateGeometry()
        self._deriveNodalGeometry()
        self._remapNodalState(oldNonMortarNodes)
        self._publishNonMortarNodeSet(model)

        if self.formulation == "lagrange":
            variables = model.reassignScalarVariablesOfConstraint(self)
            if oldMultipliers is not None:
                old = dict(zip(oldNonMortarNodes, oldMultipliers))
                for node, variable in zip(self.nonMortarNodes, variables):
                    variable.value = old.get(node, 0.0)

        if self.journal is not None:
            self.journal.message(
                f"contact '{self._name}': rebuilt after a mesh refinement, {self.nNonMortarNodes} non-mortar and "
                f"{self.nMortarNodes} mortar nodes ({len(oldNonMortarNodes)} non-mortar before)",
                self.identification,
                level=1,
            )
        return True

    def _invalidateGeometry(self):
        """Drop every cache derived from the contact surfaces, so that the next assembly rebuilds the
        frozen geometry from scratch."""

        for attribute in (
            "_nonMortarNeighbourCache",
            "currentNormals",
            "currentD",
            "currentC",
            "currentNodalWeights",
            "currentDNonzero",
            "currentDRow",
            "currentCNonzero",
            "currentCRow",
        ):
            self.__dict__.pop(attribute, None)
        self._geometryStale = True
        self._explicitStabilityFactor = None
        # A new attempt of the current increment as far as the per-increment bookkeeping goes.
        self.lastTimeStepKey = None

    def _remapNodalState(self, oldNonMortarNodes: list):
        """Carry the per-node state over to the current non-mortar nodes by ``Node`` identity; a new
        node starts at rest."""

        index = {node: i for i, node in enumerate(oldNonMortarNodes)}
        newToOld = np.array([index.get(node, -1) for node in self.nonMortarNodes], dtype=int)
        kept = newToOld >= 0

        def remapped(array, fill):
            result = np.full((self.nNonMortarNodes,) + array.shape[1:], fill, dtype=array.dtype)
            result[kept] = array[newToOld[kept]]
            return result

        self.activeSet = remapped(self.activeSet, False)
        self.augmentedMultipliers = remapped(self.augmentedMultipliers, 0.0)
        self.augmentedMultipliersConverged = remapped(self.augmentedMultipliersConverged, 0.0)
        self.currentWeakGap = remapped(self.currentWeakGap, 0.0)
        self.nodalMultipliers = remapped(self.nodalMultipliers, 0.0)
        self.frictionGapReference = remapped(self.frictionGapReference, np.nan)
        self._friction.remap(oldNonMortarNodes, self.nonMortarNodes)

    @staticmethod
    def _collectFacets(model: FEModel, elementSetName: str, side: str) -> list:
        """The contact elements of one side, in the element set's own order.

        The set is expected to hold nothing but contact elements: a solid element among them would
        not merely be ignored, its nodes would enter the coupling matrices as if they lay on the
        interface, and the resulting system would be silently wrong rather than obviously broken.
        The generated set of the surface element generator satisfies this by construction; a
        hand-written set need not, which is why it is checked rather than assumed.

        Parameters
        ----------
        model
            The model tree.
        elementSetName
            The name of the element set holding this side's contact elements.
        side
            'Non-mortar' or 'Mortar', for the error messages.

        Returns
        -------
        list
            The contact elements of that side.
        """

        if elementSetName not in model.elementSets:
            raise KeyError(
                f"{side} surface '{elementSetName}' is not an element set of this model. It has to name "
                "a set of contact elements, e.g. the '<name>_facets' set of a surfaceElementGenerator "
                "run with facets=wholeFace."
            )

        facets = list(model.elementSets[elementSetName])
        if not facets:
            raise ValueError(f"{side} surface '{elementSetName}' contains no elements.")

        offending = sorted({el.elType for el in facets if not el.elType.upper().startswith("CON")})
        if offending:
            raise ValueError(
                f"{side} surface '{elementSetName}' must contain contact elements only (element types "
                f"starting with 'CON'), but also holds: {', '.join(offending)}."
            )

        return facets

    def _checkFacetWinding(self, facets, surface_name: str):
        """Whether the facets of one surface are wound consistently.

        Purely topological, no geometry involved. Two facets sharing an edge must
        traverse it in OPPOSITE directions; if they traverse it in the same
        direction, one of them is flipped and its normal points the other way.

        3D  the directed corner edges of a facet. A directed edge seen twice means
            two facets walk it the same way.
        2D  a line facet is the directed segment node0 -> node1. An interior node
            is the end of one facet and the start of the next, so a node appearing
            twice as a start (or twice as an end) marks a flipped facet.

        This catches a locally inconsistent surface. It cannot catch a surface that
        is consistently wound but globally inside-out - that is what
        _checkSurfacesFaceEachOther is for.
        """
        dim = self.model.domainSize
        culprits = []

        if dim == 3:
            seen = {}
            for el in facets:
                nodes = el.nodes
                nCorner = 3 if len(nodes) in (3, 6) else 4
                corners = nodes[:nCorner]
                for i in range(nCorner):
                    edge = (corners[i].label, corners[(i + 1) % nCorner].label)
                    if edge in seen:
                        culprits.append((seen[edge], el.elNumber))
                    seen[edge] = el.elNumber
        else:
            starts, ends = {}, {}
            for el in facets:
                a, b = el.nodes[0].label, el.nodes[1].label
                if a in starts:
                    culprits.append((starts[a], el.elNumber))
                if b in ends:
                    culprits.append((ends[b], el.elNumber))
                starts[a] = el.elNumber
                ends[b] = el.elNumber

        if culprits:
            pairs = ", ".join(f"({a}, {b})" for a, b in culprits[:5])
            self._warnOnce(
                f"winding_{surface_name}",
                f"surface '{surface_name}' is not consistently wound: {len(culprits)} facet "
                f"pair(s) traverse a shared edge in the SAME direction, first {pairs}. One facet "
                f"of each pair has its normal reversed, so the weighted gap and the contact "
                f"pressure change sign there and the contact transmits tension instead of "
                f"compression. Fix the node ordering of the contact overlay elements.",
            )

    def _checkSurfacesFaceEachOther(self):
        """Whether the slave normals point towards the master surface at all.

        A surface can be wound perfectly consistently and still be inside-out as a
        whole; the winding check cannot see that, but the relative position of the
        two surfaces can. The contact normal must point away from the slave body,
        i.e. roughly towards the master.

        Deliberately a weak test with a wide margin: it is meant to catch a
        surface that is flipped outright, not to police curved or partially
        overlapping interfaces.

        The margin is what makes it usable. A bare sign test on the normal
        separation between the two centroids also fires on an interface whose two
        sides INTERPENETRATE in the initial configuration, which is an ordinary way
        to set a contact problem up and says nothing about orientation. The two
        cases differ by orders of magnitude rather than by sign: a flipped surface
        puts the master a body thickness behind the slave, an initial overlap a
        fraction of an element. Comparing against a tenth of the interface's own
        diameter separates them with room to spare in both directions.
        """
        if not self.nNonMortarNodes or not self.nMortarNodes:
            return

        slave_centroid = np.mean(self._X[: self.nNonMortarNodes], axis=0)
        master_centroid = np.mean(self._X[self.nNonMortarNodes :], axis=0)
        towards_master = master_centroid - slave_centroid
        separation = np.linalg.norm(towards_master)
        if separation < 1e-14:
            return  # coincident centroids: no information, and not our problem

        mean_normal = np.mean(self.undeformedNormals, axis=0)
        if np.linalg.norm(mean_normal) < 1e-8:
            return  # normals cancel out (e.g. a closed surface) - nothing to say

        mean_normal = mean_normal / np.linalg.norm(mean_normal)
        normalSeparation = float(np.dot(mean_normal, towards_master))

        interfaceDiameter = self._interfaceDiameter
        if interfaceDiameter <= 0.0:
            return

        if normalSeparation < -0.1 * interfaceDiameter:
            self._warnOnce(
                "surfaces_face_away",
                f"the averaged non-mortar normal points AWAY from the mortar surface: the mortar "
                f"centroid lies {-normalSeparation:.4g} BEHIND the non-mortar one, against an "
                f"interface diameter of {interfaceDiameter:.4g}. The contact normal must point out "
                f"of the non-mortar body, so this indicates that its facets are oriented inwards - "
                f"or that the two surfaces are swapped. Gap and pressure then carry the wrong sign "
                f"and the contact bonds the surfaces instead of separating them.",
            )

    def _adjacentYoungsModuli(self) -> dict:
        """Young's moduli of the materials whose elements touch the contact surfaces.

        Returned as {section/material: E} so that a caller can report what it chose
        from. Only the ADJACENT materials are considered - a soft material far from
        the interface says nothing about the contacting pair.

        Marmot materials carry their parameters as a flat array whose first entry is
        the Young's modulus (LINEARELASTIC, GCDP, ...). That is a convention, not a
        guarantee, which is why every caller reports the resolved value.
        """
        contact_nodes = set(self._nodes)
        candidates = {}
        for name, section in self.model.sections.items():
            material = getattr(section, "material", None)
            if material is None:
                continue
            touches = False
            for elSet in getattr(section, "elSets", []):
                for el in elSet:
                    if contact_nodes.intersection(el.nodes):
                        touches = True
                        break
                if touches:
                    break
            if not touches:
                continue
            if isinstance(material, dict):
                # Marmot: flat parameter array, first entry is E by convention
                props = material.get("properties")
                E = float(props[0]) if props is not None and len(props) else None
                mat_name = material.get("name", "?")
            else:
                # EdelweissFE-native material classes keep it as _E (see e.g.
                # materials/linearelastic); materialProperties is the fallback.
                mat_name = type(material).__name__
                E = getattr(material, "_E", None)
                if E is None:
                    props = getattr(material, "materialProperties", None)
                    E = float(props[0]) if props is not None and len(props) else None
                E = float(E) if E is not None else None
            if E is not None and E > 0.0:
                candidates[f"{name}/{mat_name}"] = E

        return candidates

    def _smallestAdjacentYoungsModulus(self):
        """The smallest of :meth:`_adjacentYoungsModuli`, or None if there is none."""

        candidates = self._adjacentYoungsModuli()
        return min(candidates.values()) if candidates else None

    def _resolveCn(self):
        """Set c_n to the smallest initial Young's modulus adjacent to the interface.

        Why a Young's modulus at all: the lower end of the admissible band, below which
        the active set does not converge, scales linearly with the stiffness of the
        contacting pair, and above that bound the influence of the value is negligible
        until, far above it, the set begins to chatter between two states. The modulus
        therefore fixes the order of magnitude and nothing finer.

        On the sources, precisely, because neither covers this case as it stands. That
        the parameter is purely algorithmic is Gitterle, Popp, Gee & Wall (2010): "the
        algorithmic parameters c_n and c_t do not influence the accuracy of the solution
        but can influence the convergence behaviour". The lower bound and its stiffness
        dependence are Hueber & Wohlmuth (2005), Section 7: "for c > c_0, the influence
        of c on N_l is negligible ... the lower bound c_0 depends on the material
        parameters ... it seems to be that c_0 depends linearly on E" - their wording,
        an observation rather than a result. But that whole section concerns their
        INEXACT strategy, which updates the active set after each multigrid sweep, and
        they say plainly that c "does not have an influence, if we solve the linear
        problems exactly". EdelweissFE does solve them exactly, so their argument does
        not reach this code directly; what makes the value matter here is that the set
        is re-decided per Newton iteration on a geometry that changes between
        increments, which their quasi-linear setting does not have. The band is real -
        it is measured in 06_active_set_pdass - but it is measured here, not inherited.

        Taking the modulus at the start of the computation is
        well defined even for a damaging material (GCDP), because no damage has
        accumulated yet - and since c_n is purely algorithmic, its later evolution
        does not matter.

        Why only the ADJACENT materials: the global minimum over the model would be
        the wrong direction. A soft material somewhere far from the interface would
        push c_n BELOW the lower bound of the contacting pair, which is exactly the
        regime in which the active set fails to converge.

        Marmot materials carry their parameters as a flat array whose first entry is
        the Young's modulus (LINEARELASTIC, GCDP, ...). That is a convention, not a
        guarantee, which is why the resolved value is always reported.
        """
        candidates = self._adjacentYoungsModuli()

        if not candidates:
            self.c_n = 1.0e6
            self._warnOnce(
                "cn_not_derivable",
                "no Young's modulus could be determined for the materials adjacent to the "
                f"contact surfaces, so c_n falls back to {self.c_n:.3e}. That value carries a "
                "unit and is only meaningful for a model in MPa. Set 'cn' explicitly - it must "
                "be at the order of the Young's modulus of the softer contacting body, and "
                "larger is the safe direction.",
            )
            return

        chosen = min(candidates, key=candidates.get)
        self.c_n = candidates[chosen]
        listed = ", ".join(f"{k} = {v:.4g}" for k, v in sorted(candidates.items(), key=lambda kv: kv[1]))
        self._warnOnce(
            "cn_derived",
            f"c_n was not given and has been derived as {self.c_n:.4g}, the smallest initial "
            f"Young's modulus adjacent to the interface ({chosen}). Considered: {listed}. This "
            f"assumes the first material constant is the Young's modulus, which is the Marmot "
            f"convention but not guaranteed - check it, or set 'cn' explicitly. Larger is the "
            f"safe direction.",
        )

    def _resolveEpsN(self):
        """Derive the penalty modulus eps_N of p_I = eps_N * penetration_I.

        THIS RULE IS NOT AN ESTABLISHED ONE. The penalty form itself prescribes no way
        of choosing its parameter. What follows is the engineering rule of this
        implementation, reported once so it stays checkable, and overridable with the
        `penaltyStiffness` keyword.

        The rule. eps_N is a POINTWISE contact stiffness per unit area -- the classical
        parameter of a penalty form, and the same quantity as the ``penalty`` of the two
        node/segment penalty constraints, so a value can be carried between them.
        Taking eps_N = E/h would give the interface the stiffness of one adjacent
        element layer, which is the natural scale but too soft in practice: the
        error reduction of the Uzawa loop per augmentation is governed by the
        stiffness ratio, roughly k_structure/(k_structure + k_contact), so a
        contact as stiff as the structure halves the violation per round. Measured
        on the two-block reference (Control_Tests 03, hex8, E = 1000): eps_N = E/h
        reduces it by a factor 0.51 per augmentation and needs about 25 rounds for
        a relative 1e-6, while 100*E/h needs FOUR (penetration 4.4e-5 -> 3.5e-7 ->
        3.4e-9 -> 3.5e-11). The factor is therefore set to

            eps_N = C * E / h ,   C = 100

        with E the smallest initial Young's modulus adjacent to the interface (same
        argument as for c_n) and h the characteristic facet size, taken from the mortar
        weights themselves: sum_I D_II = |gamma| is the covered interface measure
        (row-sum identity), so h = (|gamma|/nFacets)^(1/(dim-1)).

        NO NODAL WEIGHT ENTERS HERE any more. It used to, as a mean over the interface,
        because the stiffness was a single kappa multiplying the weighted gap; the weight
        now sits per node in :meth:`_nodalPenaltyStiffness`, which is the only place it
        belongs -- a MEAN weight cannot stand in for weights that differ by a factor of
        four across the interface, and using one is exactly what made the patch test fail.

        With the augmented Lagrangian enabled the converged solution does not depend
        on kappa at all (the outer loop drives the constraint violation
        to the augmentation tolerance whatever kappa is); it only sets how fast the
        outer loop converges. The rule therefore has to be reasonable, not right.
        """
        # EXPLICIT DYNAMICS takes a different rule, because there the stiffness is bounded from
        # above by something the implicit rule knows nothing about: the stability limit of the
        # time integrator. The rule below chooses the largest stiffness that the current time
        # increment can carry, which is the same idea LS-DYNA's SOFT=1 contact uses -- a stiffness
        # from the nodal masses and the time step rather than from the material -- except that
        # here it follows from the eigenvalue bound of Kwon et al. (2024), Eqs. (20)-(22), rather
        # than from a rule of thumb:
        #
        #     omega^2 = kappa * R <= lambda_max = 4/dt^2   =>   kappa <= 4/(dt^2 R)
        #
        # Re-derived at EVERY geometry rebuild rather than once: R carries the segmentation, which
        # changes as the contact evolves, and a stiffness that was stable for the first overlap is
        # not thereby stable for a later one.
        if self._explicit and self._explicitLumpedMass is not None:
            rows = self._stabilityRowSums(self._explicitLumpedMass)
            factor = float(np.max(rows[0])) if rows is not None else 0.0
            dt = self._explicitTimeIncrement
            if factor > 0.0 and dt > 0.0:
                # Half the limit. The bound holds for the geometry it was computed on, and the
                # geometry is frozen only until the next connectivity tick -- between two ticks the
                # overlap moves, and R with it. A factor of two in the stiffness is a factor of
                # sqrt(2) in the frequency, which is the margin that buys.
                #
                # With friction the tangential stick spring shares the budget, row by row:
                # max_a (eps_N r_N,a + eps_T r_T,a) <= 2/dt^2. With eps_T following eps_N that is a
                # single bound on eps_N; with a fixed eps_T, eps_N gets what eps_T leaves.
                budget = 2.0 / (dt * dt)
                previous = getattr(self, "epsN", None)
                if not self.friction:
                    self.epsN = budget / factor
                elif self._epsTGiven <= 0.0:
                    factor = float(np.max(rows[0] + rows[1]))
                    self.epsN = budget / factor
                else:
                    loaded = rows[0] > 0.0
                    self.epsN = float(np.min((budget - self._epsTGiven * rows[1][loaded]) / rows[0][loaded]))
                    if self.epsN <= 0.0:
                        raise ValueError(
                            f"MortarContact '{self._name}': tangentPenalty = {self._epsTGiven:.4g} alone exceeds "
                            f"the stability limit of the time increment {dt:.4g}, so no normal penalty "
                            f"stiffness is left. Lower tangentPenalty or leave it at 0 to follow eps_N."
                        )
                if previous is None or not np.isclose(previous, self.epsN, rtol=0.25):
                    self._warnOnce(
                        "epsN_derived_explicit",
                        f"'penaltyStiffness' was not given and this is an explicit analysis, so "
                        f"eps_N has been derived from the STABILITY limit rather than from the "
                        f"material: eps_N = {self.epsN:.4g} = 0.5 * 4/(dt^2 * R) with "
                        f"dt = {dt:.4g} and R = {factor:.4g}"
                        f"{' (R including the tangential friction spring)' if self.friction and self._epsTGiven <= 0.0 else ''}. "
                        f"A larger value would need a smaller "
                        f"time increment; set 'penaltyStiffness' explicitly to choose that "
                        f"trade-off yourself, and watch the reported penetration.",
                    )
                return
            self._warnOnce(
                "epsN_not_derivable_explicit",
                "the stability limit needed to derive the penalty parameter for this explicit "
                "analysis is not available (no covered interface, or no lumped mass on the "
                "contact nodes), so the implicit rule below is used instead. That rule knows "
                "nothing about the stable time increment -- check the result for instability.",
            )

        self._derive_epsN = False

        weights = self.currentNodalWeights
        total = float(np.sum(weights))
        nFacets = len(self.nonMortarFacets)
        dim = self.model.domainSize

        if total <= 0.0 or nFacets == 0 or self.nNonMortarNodes == 0:
            self.epsN = 1.0e6
            self._warnOnce(
                "epsN_not_derivable",
                f"the interface measure needed to derive the penalty parameter is not "
                f"available (covered measure {total:.3e} over {nFacets} facet(s)), so eps_N "
                f"falls back to {self.epsN:.3e}. That value carries a unit and is almost "
                f"certainly wrong for this model - set 'penaltyStiffness' explicitly.",
            )
            return

        # dim-1 is the dimension of the interface: a length in 2D, an area in 3D.
        h = (total / nFacets) ** (1.0 / (dim - 1))
        # Stiffness ratio contact/structure, see the docstring. Not a literature
        # value - it buys the convergence rate of the outer loop, and with the
        # augmentation enabled it does not influence the converged result.
        stiffness_ratio = 100.0

        E = self._smallestAdjacentYoungsModulus()
        if E is None:
            self.epsN = 1.0e6
            self._warnOnce(
                "epsN_not_derivable_E",
                f"no Young's modulus could be determined for the materials adjacent to the "
                f"contact surfaces, so eps_N falls back to {self.epsN:.3e}. That value "
                f"carries a unit and is only meaningful for a model in MPa - set "
                f"'penaltyStiffness' explicitly.",
            )
            return

        self.epsN = stiffness_ratio * E / h
        self._warnOnce(
            "epsN_derived",
            f"'penaltyStiffness' was not given and eps_N has been derived as "
            f"{self.epsN:.4g} = {stiffness_ratio:g}*E/h with E = {E:.4g} and h = {h:.4g}. "
            f"It is a pointwise contact stiffness per unit area, i.e. the same quantity as the "
            f"'penalty' of nodeToDeformableSurfacePenalty. This is an engineering rule of this "
            f"implementation rather than an established value. With augmentedLagrange enabled "
            f"the converged result does not depend on it.",
        )

    def _checkConvergedActiveSet(self, U_ref: np.ndarray):
        """Re-evaluate the NCP indicator on the CONVERGED state of the last increment.

        The active set is frozen once it has settled for two consecutive iterations
        (or on a detected cycle, or at the iteration cap). The Newton iteration then
        continues on that fixed set, so the state it finally converges to was
        decided at an intermediate iterate. Usually that is exactly right - the set
        settled because it had converged. It is not guaranteed, though, and nothing
        in the increment itself notices.

        This is the missing check, run at the start of the next increment when the
        previous one has converged: with the geometry that was in force and the
        displacements that came out, would the indicator still produce the same set?
        If not, the increment enforced the wrong branch at those nodes and its
        solution does not satisfy the Signorini conditions.
        """
        if not self.useActiveSet:
            return

        # Only the Lagrange branch freezes its set. The penalty branch re-derives it
        # from the pressure in every single iteration, so there is no decision taken
        # at an intermediate iterate that could need re-checking - and no multiplier
        # in U_ref to read, since it carries no multiplier degrees of freedom.
        if self.formulation == "penalty":
            return

        dim = self.model.domainSize
        sf = self.sizeField
        nNodes = len(self._nodes)
        nSlave = self.nNonMortarNodes

        disp = U_ref[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
        coords = self._X + disp
        x_slave, x_master = coords[:nSlave], coords[nSlave:]
        idx_LM_0 = sf * nNodes

        would_be = np.zeros(nSlave, dtype=bool)
        for I in range(nSlave):  # noqa: E741 - I is the non-mortar node index of the formulation
            n_I = self.currentNormals[I]
            nzD, nzC = self.currentDNonzero[I], self.currentCNonzero[I]
            g_weak = 0.0
            if len(nzD):
                g_weak -= self.currentDRow[I] @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_weak += self.currentCRow[I] @ (x_master[nzC] @ n_I)
            D_II = self.currentNodalWeights[I]
            inv_D = 1.0 / D_II if abs(D_II) > self.currentWeightTolerance else 0.0
            p_n = U_ref[idx_LM_0 + I] * np.sign(D_II)
            would_be[I] = bool(p_n - self.c_n * g_weak * inv_D > 0.0)

        flipped = np.flatnonzero(would_be != self.activeSet)
        if len(flipped):
            self._warnOnce(
                "active_set_unstable_at_convergence",
                f"the active set of a converged increment does not reproduce itself: "
                f"{len(flipped)} slave node(s) (first local index {flipped[0]}) would switch "
                f"branch if the indicator were re-evaluated on the converged state. That "
                f"increment enforced g_weak = 0 on a node that wants to open, or lambda = 0 on "
                f"one that wants to close, so its solution does not satisfy the Signorini "
                f"conditions. Reduce the increment size, or verify the result by checking the "
                f"nodal pressures and openings of the converged state against each other.",
            )

    @classmethod
    def fromConstraintDefinition(cls, name: str, definition: dict, model: FEModel, journal: Journal) -> "Constraint":
        """Build this constraint from a parsed ``*constraint`` definition. See
        :class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase` for why this is
        separate from ``__init__``.

        Overridden only to pass the journal through: the base class accepts one and the input-file
        route supplies one, but its default implementation drops it, which would leave this
        constraint's diagnostics with nowhere to go. The options themselves stay in their raw form,
        since ``__init__`` resolves them against the schema itself.
        """
        return cls(name, model, journal, **definition)

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        return self._nDof

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return self.nMultipliers

    def computeNormals(self, U_np: np.ndarray = None) -> np.ndarray:
        """Averaged outward-pointing unit normal at every non-mortar (slave) node.

        The averaged nodal normal: the element normal
        x,xi cross x,eta is evaluated at the node's OWN natural coordinate in each
        adjacent facet, each contribution is normalized to unit length, the unit
        normals are summed, and the sum is normalized again. Normalizing at the end
        is what makes the field continuous across facet boundaries.

        Two properties this rests on, both measured in 01_node_normals:

        1. Evaluated per NODE, not per facet. A single facet-constant normal (the
           diagonal cross product of the corner nodes in 3D, the corner chord in 2D)
           assigns the same direction to a corner and to a mid-side node of the SAME
           quadratic facet, so the curvature of that facet never reaches the normal
           and the field converges only linearly: on the cylinder patch, CONQUAD8
           and CONQUAD9 then produce exactly the CONQUAD4 error (7.16 deg -> 3.58
           deg, rate 1), i.e. the quadratic facet types buy nothing at all.

        2. The contributions are summed UNWEIGHTED, as in both references. Summing
           the un-normalized cross products instead would weight each facet by its
           local surface Jacobian, and that is measurably worse on a graded mesh:
           the larger facet averages over a wider arc and is therefore the poorer
           estimate of the normal, so giving it more weight pulls the nodal normal
           away from the exact one (cylinder patch, neighbour size ratio 2, nodes
           with unequal neighbours: 3.82 deg weighted against 1.91 deg unweighted
           for CONQUAD4, 0.0211 against 0.0147 deg for CONQUAD8/9). It
           would also make the weight depend on the element type rather than on the
           geometry, because the surface Jacobian carries the size of the reference
           element - a triangle and a quadrilateral of EQUAL area contribute in the
           ratio 8:1.

        All these variants coincide exactly wherever the contributions at a node are
        parallel (a FLAT slave surface) or of equal length and symmetric about the
        node (a uniform mesh on a curved one): any positive weighting then normalizes
        to the same direction. It is the equal size of the adjacent facets that does
        this, not mesh uniformity as such - on a curved surface meshed uniformly in
        the PARAMETER but not in facet size, the variants differ again.

        If U_np is provided, the normal vectors are calculated in the deformed
        configuration. Otherwise, they are calculated in the undeformed one.
        """
        dim = self.model.domainSize
        normals = np.zeros((self.nNonMortarNodes, dim))

        node_to_idx = self.nonMortarNodeToIndex

        # Iterate over all non-mortar facets
        for el in self.nonMortarFacets:
            facet_nodes = el.nodes

            # Retrieve coordinates
            coords = []
            for node in facet_nodes:
                X = node.coordinates
                if U_np is not None:
                    # Retrieve displacement from local solution slice
                    node_idx = self.nodeToGlobalIndex[node]
                    u = U_np[self.sizeField * node_idx : self.sizeField * node_idx + dim]
                    coords.append(X + u)
                else:
                    coords.append(X)

            coords = np.array(coords)

            # Element normal evaluated AT each node's own natural coordinate,
            # normalized, and summed over the adjacent facets - the averaged nodal
            # normal of Popp, Gitterle, Gee & Wall (2010), written out as Farah
            # (2018), Eq. (4.41). Both sum UNIT element normals; see the docstring
            # for what each of the two steps is worth, measured.
            #
            # A contribution of zero length is a degenerate facet at that node. It is
            # dropped rather than divided by: the sum below then rests on the
            # remaining facets, and if there are none the node ends up with a zero
            # normal, which is reported right after this loop.
            dN_at_nodes = el.getShapeFunctionDerivativesAtNodes()
            for local, node in enumerate(facet_nodes):
                if node not in node_to_idx:
                    continue
                t = dN_at_nodes[local] @ coords
                if dim == 3:
                    n_node = np.cross(t[0], t[1])
                else:
                    n_node = np.array([t[0][1], -t[0][0]])
                len_n = np.linalg.norm(n_node)
                if len_n <= 1e-14:
                    continue
                normals[node_to_idx[node]] += n_node / len_n

        # Normalize the normal vectors
        degenerate = []
        for i in range(self.nNonMortarNodes):
            norm = np.linalg.norm(normals[i])
            if norm > 1e-14:
                normals[i] /= norm
            else:
                normals[i] = np.zeros(dim)
                degenerate.append(i)

        if degenerate:
            # A zero normal is not a harmless special case: g_weak and p_n are both
            # built by contracting with n_I, so both collapse to zero, the
            # indicator s_n is zero, and the node can NEVER become active. It
            # silently drops out of the contact. Causes are a degenerate facet
            # (zero area, skipped in the loop above) or facets whose unit normals
            # cancel, which is what a node on a sharp fold between two opposing
            # facets does - and note that with unit contributions the cancellation
            # no longer needs the two facets to be of equal size.
            self._warnOnce(
                "degenerate_nodal_normal",
                f"{len(degenerate)} slave node(s) with a vanishing averaged nodal normal "
                f"(first: local index {degenerate[0]}). Such nodes carry no contact at all - "
                f"the weighted gap and the pressure both contract with n_I and vanish with it, "
                f"so the active-set indicator can never turn them on. Usual causes: a "
                f"degenerate (zero-area) contact facet, or adjacent facets whose normals "
                f"cancel because they fold back onto each other.",
            )

        return normals

    def _nonMortarFacetNeighbours(self) -> dict:
        """Per non-mortar facet number, the non-mortar facets sharing at least one node with it.

        The facets a Gauss point can be re-attributed to on the non-mortar side (see
        _containingFacet): the chord of a curved edge deviates from the edge by a fraction of the
        facet size, so the point always lies in a facet of the one-ring. Built once; the topology
        of the contact surfaces does not change.
        """
        if not hasattr(self, "_nonMortarNeighbourCache"):
            byNode = {}
            for f in self.nonMortarFacets:
                for nd in f.nodes:
                    byNode.setdefault(nd, []).append(f)
            self._nonMortarNeighbourCache = {
                f.elNumber: list(
                    {g.elNumber: g for nd in f.nodes for g in byNode[nd] if g.elNumber != f.elNumber}.values()
                )
                for f in self.nonMortarFacets
            }
        return self._nonMortarNeighbourCache

    @staticmethod
    def _containingFacet(el, coords_2d, point_2d, alternatives, inPlane):
        """The facet that contains a Gauss point of the segmentation, and its natural coordinate there.

        The segmentation clips the straight sub-cells of the facets (see _subCellMap), and the
        point is first mapped back into the facet whose sub-cell produced it. On a facet with
        straight edges that facet contains the point. On a quadratic facet whose edges are CURVED
        in the surface -- an O-grid, a circular mesh, or any quadratic mesh that has been
        distorted -- it need not: the sliver between the chord and the true edge belongs to the
        neighbouring facet, and the back-mapping into the wrong facet returns a natural
        coordinate outside the reference domain, i.e. shape functions extrapolated beyond the
        element. Measured on the crossed-interface patch test of the contact study (quadratic
        hexahedra, 60 degrees): 6.9 % of the mortar-side Gauss points, up to |xi| = 1.17, and a
        mortar-side consistency error max|C^T 1 - int N_m| / max|int N_m| of 1.0e-2 that no
        quadrature order removes, since it is the attribution and not the integration that is
        wrong. Re-attributed, the error drops to 2.2e-3, and the remainder is quadrature of the
        triangles that straddle the curved edge (6.5e-5 with a 16-fold subdivided rule). On
        full-integration C3D20 the patch-test pressure scatter goes from 5.6e-3 to 1.9e-3. (With
        C3D20R the scatter stays at about 3 % either way: the reduced-integrated serendipity
        element with curved edges does not pass the patch test by itself -- its reactions to a
        uniform stress miss the consistent face loads by 3.1 %, with no contact in the model.)
        Such a point is re-attributed to the facet among ``alternatives`` that contains it.

        Returns ``(facet, natural coordinate, converged, moved)``, or ``(None, None, False,
        False)`` if the point provably lies outside every facet -- which happens only beyond the
        curved boundary of a surface, where there is nothing to integrate. A back-mapping that
        did not converge proves nothing; such a point stays with its original facet and is
        reported through the projection-failure diagnostic, exactly as before.
        """
        local, ok = mapPlaneToNatural(el, coords_2d, point_2d)
        if ok and isInsideReferenceElement(el, local):
            return el, local, ok, False
        for other in alternatives:
            if other is el:
                continue
            local_o, ok_o = mapPlaneToNatural(other, inPlane(other), point_2d)
            if ok_o and isInsideReferenceElement(other, local_o):
                return other, local_o, ok_o, True
        if not ok:
            return el, local, ok, False
        return None, None, False, False

    def computeMortarCouplingMatrices(self, U_np: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        """Compute the global mortar coupling matrices D (slave-slave) and C (slave-master).
        Supports both 2D and 3D contact elements.
        """
        dim = self.model.domainSize
        if dim not in (2, 3):
            raise NotImplementedError("Mortar coupling matrix integration is only implemented for 2D and 3D.")

        n_slave = self.nNonMortarNodes
        n_master = self.nMortarNodes

        # Precompute current deformed coordinates for all nodes
        current_coords = {}
        for node in self._nodes:
            X = node.coordinates
            if U_np is not None:
                node_idx = self.nodeToGlobalIndex[node]
                u = U_np[self.sizeField * node_idx : self.sizeField * node_idx + dim]
                current_coords[node] = X + u
            else:
                current_coords[node] = X

        # Build AABB bounding boxes for all master facets to construct the BVH tree
        facets_with_bounds = []
        for m_el in self.mortarFacets:
            m_coords = np.array([current_coords[n] for n in m_el.nodes])
            centroid = np.mean(m_coords, axis=0)

            aabb_min = np.min(m_coords, axis=0)
            aabb_max = np.max(m_coords, axis=0)
            margin = max(0.15 * np.max(aabb_max - aabb_min), 0.1)
            aabb_min -= margin
            aabb_max += margin

            facets_with_bounds.append((m_el, centroid, aabb_min, aabb_max))

        bvh_root = buildBoundingVolumeHierarchy(facets_with_bounds)

        # ------------------------------------------------------------------
        # PASS 1: Segmentation - collect all integration point records per
        # slave facet: (N_s, master elNumber, N_m, dGamma)
        # ------------------------------------------------------------------
        seg_records = {}  # slave elNumber -> list of records
        seg_masters = {}  # slave elNumber -> {master elNumber: m_el}
        slave_els = {}  # slave elNumber -> (s_el, s_idx)

        # Gauss point back-mapping failures (see mapPlaneToNatural): a non-converged
        # natural coordinate is indistinguishable from a converged one and goes
        # straight into N_s / N_m and thus into D and C, so it has to be counted here
        # - nothing downstream can notice it.
        n_proj_failed = 0
        first_proj_failure = None

        # Gauss points re-attributed to the facet that actually contains them, and Gauss points
        # that no facet contains -- see _containingFacet. Diagnostic; kept for the tests.
        self.lastReattributedGaussPoints = 0
        self.lastDroppedGaussPoints = 0
        slave_neighbours = self._nonMortarFacetNeighbours()

        for s_el in self.nonMortarFacets:
            s_nodes = s_el.nodes

            s_coords = np.array([current_coords[nd] for nd in s_nodes])
            s_idx = np.array([self.nonMortarNodeToIndex[nd] for nd in s_nodes])

            # Query the BVH tree once per slave facet to find nearby Master candidates
            s_aabb_min = np.min(s_coords, axis=0)
            s_aabb_max = np.max(s_coords, axis=0)
            s_margin = max(0.15 * np.max(s_aabb_max - s_aabb_min), 0.1)

            candidates = []
            queryBoundingVolumeHierarchy(bvh_root, s_aabb_min - s_margin, s_aabb_max + s_margin, candidates)
            if not candidates:
                continue

            # Loop over the linear sub-cells of the slave facet
            for s_sub in getSubCells(s_el):
                sc_coords = s_coords[s_sub]

                if dim == 3:
                    # 3D surface Mortar integration (Auxiliary plane projection & Sutherland-Hodgman clipping)
                    n_vec = facetNormal(sc_coords)
                    n_norm = np.linalg.norm(n_vec)
                    if n_norm < 1e-14:
                        continue  # degenerate sub-cell
                    normal = n_vec / n_norm
                    p0 = np.mean(sc_coords, axis=0)
                    t1, t2 = tangentBasis(normal)

                    s_sub_2d = toPlaneCoordinates(sc_coords, p0, t1, t2)
                    s_full_2d = toPlaneCoordinates(s_coords, p0, t1, t2)

                    # Facets projected into THIS auxiliary plane, computed only when a Gauss
                    # point has to be re-attributed (see _containingFacet).
                    plane_cache = {}

                    def inPlane(el, _cache=plane_cache, _p0=p0, _t1=t1, _t2=t2):
                        if el.elNumber not in _cache:
                            _cache[el.elNumber] = toPlaneCoordinates(
                                np.array([current_coords[nd] for nd in el.nodes]), _p0, _t1, _t2
                            )
                        return _cache[el.elNumber]

                    # The slave sub-cell is the CLIP polygon, so it must be convex
                    # (see isConvexPolygon). For CONQUAD8 this requires the
                    # mid-side node to stay on its own side of the element centre,
                    # for CONQUAD9 the centre node to stay clear of the corners -
                    # margins no usable volume element gets anywhere near, which is
                    # why this is a warning and not a raise.
                    if not isConvexPolygon(s_sub_2d):
                        self._warnOnce(
                            "nonconvex_subcell",
                            f"non-convex sub-cell on slave facet {s_el.elNumber} (element type "
                            f"{s_el.elType}). The segmentation clips against the half-plane of "
                            f"each sub-cell edge, so contact area is lost without further notice. "
                            f"Check the mid-side/centre node positions of the quadratic contact "
                            f"facets; the element is severely distorted.",
                        )

                    for m_el in candidates:
                        m_nodes = m_el.nodes
                        m_coords = np.array([current_coords[nd] for nd in m_nodes])
                        m_full_2d = toPlaneCoordinates(m_coords, p0, t1, t2)

                        for m_sub in getSubCells(m_el):
                            m_sub_2d = m_full_2d[m_sub]

                            # The master sub-cell is the SUBJECT of the clip, which
                            # Sutherland-Hodgman accepts non-convex - but the overlap
                            # inherits its reflex vertices and the fan triangulation
                            # below then over-counts the area (see isConvexPolygon).
                            if not isConvexPolygon(m_sub_2d):
                                self._warnOnce(
                                    "nonconvex_master_subcell",
                                    f"non-convex sub-cell on master facet {m_el.elNumber} (element "
                                    f"type {m_el.elType}), projected into the plane of slave facet "
                                    f"{s_el.elNumber}. The overlap polygon inherits the reflex "
                                    f"vertex and the fan triangulation then integrates over area "
                                    f"outside the facet. Check the mid-side/centre node positions "
                                    f"of the quadratic contact facets; the element is severely "
                                    f"distorted.",
                                )

                            overlap_2d = sutherlandHodgmanClip(m_sub_2d, s_sub_2d)
                            if len(overlap_2d) < 3:
                                continue

                            for tri in triangulatePolygon(overlap_2d):
                                v0, v1, v2 = tri[0], tri[1], tri[2]
                                area_jac = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
                                if area_jac < 1e-14:
                                    continue

                                for gp, w in zip(_triangleGaussPoints, _triangleGaussWeights):
                                    L1, L2 = gp[0], gp[1]
                                    x_gp_2d = (1.0 - L1 - L2) * v0 + L1 * v1 + L2 * v2

                                    # The clipping ran on straight sub-cells, so the point
                                    # need not lie in the two facets whose sub-cells produced
                                    # it (see isInsideReferenceElement); _containingFacet finds
                                    # the facets that do contain it.
                                    rec_s, local_s, ok_s, moved_s = self._containingFacet(
                                        s_el, s_full_2d, x_gp_2d, slave_neighbours[s_el.elNumber], inPlane
                                    )
                                    rec_m, local_m, ok_m, moved_m = self._containingFacet(
                                        m_el, m_full_2d, x_gp_2d, candidates, inPlane
                                    )
                                    if rec_s is None or rec_m is None:
                                        # On neither side a facet contains the point: it lies
                                        # beyond the curved boundary of a surface, where there
                                        # is no contact area to integrate.
                                        self.lastDroppedGaussPoints += 1
                                        continue
                                    if moved_s or moved_m:
                                        self.lastReattributedGaussPoints += 1
                                    if not ok_s:
                                        n_proj_failed += 1
                                        if first_proj_failure is None:
                                            first_proj_failure = ("slave", rec_s.elNumber, rec_s.elType)
                                    if not ok_m:
                                        n_proj_failed += 1
                                        if first_proj_failure is None:
                                            first_proj_failure = ("master", rec_m.elNumber, rec_m.elType)

                                    N_s = rec_s.getShapeFunctions(local_s)
                                    N_m = rec_m.getShapeFunctions(local_m)

                                    s_num = rec_s.elNumber
                                    if s_num not in seg_records:
                                        seg_records[s_num] = []
                                        seg_masters[s_num] = {}
                                        slave_els[s_num] = (
                                            rec_s,
                                            np.array([self.nonMortarNodeToIndex[nd] for nd in rec_s.nodes]),
                                        )
                                    seg_records[s_num].append((N_s, rec_m.elNumber, N_m, area_jac * w))
                                    seg_masters[s_num][rec_m.elNumber] = rec_m

                else:
                    # 2D line segment Mortar integration
                    for m_el in candidates:
                        m_nodes = m_el.nodes
                        m_coords = np.array([current_coords[nd] for nd in m_nodes])
                        for m_sub in getSubCells(m_el):
                            mc_coords = m_coords[m_sub]
                            s_start, s_end, t_vec = clipLineSegments(sc_coords, mc_coords)
                            if s_end - s_start < 1e-12:
                                continue

                            half_len = 0.5 * (s_end - s_start)
                            mid_s = 0.5 * (s_start + s_end)
                            for gp_1d, w_1d in zip(_lineGaussPoints, _lineGaussWeights):
                                s_gp = mid_s + half_len * gp_1d[0]
                                x_gp = sc_coords[0] + s_gp * t_vec
                                dG = half_len * w_1d

                                local_s, ok_s = mapPlaneToNatural(s_el, s_coords, x_gp)
                                local_m, ok_m = mapPlaneToNatural(m_el, m_coords, x_gp)
                                if not ok_s:
                                    n_proj_failed += 1
                                    if first_proj_failure is None:
                                        first_proj_failure = ("slave", s_el.elNumber, s_el.elType)
                                if not ok_m:
                                    n_proj_failed += 1
                                    if first_proj_failure is None:
                                        first_proj_failure = ("master", m_el.elNumber, m_el.elType)

                                N_s = s_el.getShapeFunctions(local_s)
                                N_m = m_el.getShapeFunctions(local_m)

                                s_num = s_el.elNumber
                                if s_num not in seg_records:
                                    seg_records[s_num] = []
                                    seg_masters[s_num] = {}
                                    slave_els[s_num] = (s_el, s_idx)
                                seg_records[s_num].append((N_s, m_el.elNumber, N_m, dG))
                                seg_masters[s_num][m_el.elNumber] = m_el

        if n_proj_failed:
            side, el_num, el_type = first_proj_failure
            self._warnOnce(
                "gp_projection_failed",
                f"the Gauss point back-mapping did not converge for {n_proj_failed} integration "
                f"point(s) (first on the {side} facet {el_num}, element type {el_type}). For those "
                f"points the last Newton iterate was used as the natural coordinate, so the shape "
                f"function values entering D and C are wrong by an unknown amount - a "
                f"non-converged coordinate is indistinguishable from a converged one. The closest "
                f"point projection is not solvable for arbitrarily distorted or strongly curved "
                f"facets (Konyukhov & Schweizerhof 2008); check the facets around the one named "
                f"above.",
            )

        return self._assembleCouplingMatricesFromSegments(
            seg_records, seg_masters, slave_els, current_coords, n_slave, n_master
        )

    def requiresCorrectionBeforeConvergence(self) -> bool:
        """True on the first assembly of an increment in the penalty branch.

        There the contact pressure is a FUNCTION of the current state,
        p = z + kappa*g_A, and not an unknown with an equation of its own. The
        forces assembled on the extrapolated state that opens an increment have
        therefore never been equilibrated - and with the augmented Lagrangian the
        estimate z is warm-started from the previous increment on top of that, so
        they are not even the forces that produced the incoming displacements. The
        state has to be corrected before it can be tested, exactly as after an
        augmentation.

        Without this, an increment can be accepted at iteration 0 on the
        extrapolated state: no correction exists yet, so the field-correction
        criterion is satisfied by an ABSENT correction rather than a small one, and
        the flux criterion carries an absolute floor that the nodal forces of a
        small model fall below anyway. Measured on the scale-invariance case
        (11_scale_invariance, k = 1e-3): a residual of 2.5e-03 against a flux
        measure of 5.4e-04 was accepted, and the contact pressure stayed at half its
        correct value with no warning at all. The saddle-point branch cannot reach
        that state, because its multiplier rows are checked as scalar variables and
        that criterion has no floor.
        """

        return self.active and self.formulation == "penalty" and self.currentIteration == 0

    def augmentConstraint(self) -> bool:
        """One augmented-Lagrangian (Uzawa) outer iteration on the converged state.

        The augmented (Uzawa) update,

            p^(k+1)_A = p^k_A + kappa_A * g_A ,

        which is Eq. (18) of Puso, Laursen & Solberg (2008), advanced - as they require -
        only "once convergence of the Newton-Raphson loop is achieved", after which
        equilibrium is re-established with the updated estimate. The projection onto
        p >= 0 is this implementation's own release condition, where they instead let
        the contact-status step decide; a node whose augmented pressure would turn
        tensile is simply inactive.

        Termination monitors the change of the multiplier estimate from one
        augmentation to the next, as a
        RELATIVE change. The alternative they mention first - checking
        non-penetration directly - would have to bound the weighted gap, which
        carries length x area and is therefore not scale-invariant; the same trap
        the absolute solver tolerance on the multiplier row falls into (see the
        documentation section on the limit of the length scale).

        Returns
        -------
        bool
            True if the estimate moved and the increment has to be re-equilibrated.

        """

        # Everything except an augmented penalty constraint is enforced inside the
        # Newton loop and is done at this point.
        if not self.active or self.formulation != "penalty" or not self.useAugmentedLagrange:
            return False

        # Nothing assembled yet (a constraint that never saw an increment).
        if not hasattr(self, "currentNodalWeights"):
            return False

        sgn_D = np.sign(self.currentNodalWeights)
        g_pen = -self.currentWeakGap * sgn_D
        # The same noise floor the activation uses, applied here as well - see currentGapTolerance.
        # This update runs past the active set rather than through it, by design: a node that is
        # currently released must still be able to build pressure again. That makes the floor
        # indispensable here, because without it a load-free interface feeds kappa times the
        # rounding of its own geometry into the estimate. The estimate then never stops moving,
        # the convergence test below divides one noise by another, and the loop exhausts its cap
        # on an interface that has nothing to correct.
        g_pen = np.where(np.abs(self.currentWeakGap) > self.currentGapTolerance, g_pen, 0.0)
        # Node-wise, for the same reason the assembly is: the Uzawa update has to advance the
        # estimate by the same pressure increment the penalty law would have produced, and that
        # is eps_N times the POINTWISE opening at this node, not kappa times its weighted one.
        z_new = np.maximum(0.0, self.augmentedMultipliers + self._nodalPenaltyStiffness() * g_pen)

        change = float(np.max(np.abs(z_new - self.augmentedMultipliers))) if len(z_new) else 0.0
        scale = float(np.max(np.abs(z_new))) if len(z_new) else 0.0
        self.augmentedMultipliers = z_new
        self._augmentation_counter += 1

        # scale == 0 means no node carries pressure - the interface is open and
        # there is nothing to augment.
        converged = change <= self.augmentationTolerance * scale or scale == 0.0

        if converged:
            self.augmentedMultipliersConverged[:] = self.augmentedMultipliers
            return False

        if self._augmentation_counter >= self.maxAugmentations:
            self.augmentedMultipliersConverged[:] = self.augmentedMultipliers
            self._warnOnce(
                "augmentation_cap",
                f"the augmented Lagrangian hit its cap of {self.maxAugmentations} outer "
                f"iterations without reaching the tolerance "
                f"{self.augmentationTolerance:.1e} (last relative change "
                f"{change / scale:.2e}). The increment therefore converged on a pressure "
                f"estimate that had not settled, so the remaining penetration is larger "
                f"than asked for. Raise 'maxAugmentations', raise 'penaltyStiffness' (it "
                f"sets the rate of the outer loop), or reduce the increment size.",
            )
            return False

        return True

    def applyConstraint(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        if not self.active:
            return

        # c_n belongs to the semi-smooth activation test of the multiplier branch and is read
        # nowhere else. Deriving it under formulation=penalty would announce a value that does not
        # enter a single equation, which is worse than saying nothing: a reported number invites the
        # reader to check it against a result it cannot have influenced.
        if self._derive_c_n and self.formulation == "lagrange":
            self._derive_c_n = False
            self._resolveCn()

        dim = self.model.domainSize
        sf = self.sizeField
        nNodes = len(self._nodes)
        nSlave = self.nNonMortarNodes

        # Detect new increment to track iterations and reset currentIteration.
        # Normals and coupling matrices are frozen within each increment
        # (staggered geometry update), so the assembled stiffness is the exact
        # Jacobian of the residual equations within the increment.
        # A RE-ATTEMPT of an increment after a solver cutback carries the SAME
        # increment number but a smaller time increment, and it restarts the Newton
        # iteration from the last converged state. It must therefore be treated
        # exactly like a new increment: the staggered geometry (normals, D, C) has
        # to be re-evaluated at the state the attempt starts from - otherwise the
        # frozen data would stem from the diverged iterate of the failed attempt -
        # and the semi-smooth active-set iteration has to restart as well.
        # Keying the reset on the increment number alone would miss this.
        #
        # REFERENCE CONFIGURATION of the frozen geometry: the LAST CONVERGED state,
        # U_n = U_np - dU, not the state handed in as U_np.
        #
        # The reference formulations do not freeze at all: they re-evaluate D, M and the nodal
        # normals in every Newton iteration and carry their linearizations in K (Popp, Gitterle,
        # Gee & Wall 2010, Section 4.3 and Appendix A; Gitterle et al. 2010 does the same for the
        # frictional case). That is the target state (see the "The tangent" section of the mortar
        # theory page), not what is done here.
        #
        # Note what this is NOT. Popp et al. measure an "incomplete linearization" in their Table I
        # -- a LIVE geometry carried with a tangent that omits the linearizations of the nodal
        # normal and of D and M -- and it costs them 52 Newton steps against 8. Here the geometry
        # is FROZEN, so the assembled tangent is exact for the residual that is actually assembled
        # (measured: max|K - K_FD| / max|K| = 5.3e-10). The Newton rate is therefore not what
        # suffers; the cost is a consistency error of the increment itself, first order in its
        # size. Their number must not be quoted against this scheme.
        #
        # Within a staggered scheme, however, the reference must be the converged
        # state. EdelweissFE's solvers extrapolate the previous increment before the
        # first assembly (`extrapolation`, DEFAULT "linear": U_np = U_n + dU_extrap
        # already at iteration 0). Freezing at that predictor would make the
        # CONVERGED contact solution depend on a solver switch that must not
        # influence it - and it would be inconsistent within itself, since a cutback
        # re-attempt resets dU to zero and would then use a different kind of
        # reference than a regular increment. It would also void the error statement
        # of the staggered scheme, which is first order in the increment size only
        # with respect to an equilibrated reference configuration.
        step_key = (timeStep.number, timeStep.timeIncrement, timeStep.totalTime)
        self._refreshFrozenGeometry(step_key, U_np, dU, timeStep)

        normals = self.currentNormals

        # Current coordinates of all constraint nodes in the deformed configuration
        disp = U_np[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
        coords = self._X + disp
        x_slave = coords[:nSlave]
        x_master = coords[nSlave:]

        # ----------------------------------------------------------------------
        # SADDLE-POINT MODE: Lagrange multipliers lambda_I are explicit unknowns
        # in U_np, solved for jointly with the displacements. This is the only
        # currently supported/validated formulation; the module docstring above says
        # why a condensed (multiplier-free) variant was removed.
        # ----------------------------------------------------------------------
        idx_LM_0 = sf * nNodes
        penalty = self.formulation == "penalty"
        # One stiffness per non-mortar node, not one for the interface -- see
        # _nodalPenaltyStiffness. Formed once per assembly rather than per node: it depends
        # only on the frozen weights and on eps_N, neither of which moves inside this loop.
        nodalPenalty = self._nodalPenaltyStiffness() if penalty else None
        friction = penalty and self.friction
        _dim_offsets = np.arange(dim)

        for I in range(nSlave):  # noqa: E741 - I is the non-mortar node index of the formulation
            idx_LM_I = idx_LM_0 + I
            lambda_I = 0.0 if penalty else U_np[idx_LM_I]
            n_I = normals[I]
            nzD = self.currentDNonzero[I]
            nzC = self.currentCNonzero[I]
            dRow = self.currentDRow[I]
            cRow = self.currentCRow[I]

            g_I_weak = 0.0
            if len(nzD):
                g_I_weak -= dRow @ (x_slave[nzD] @ n_I)
            if len(nzC):
                g_I_weak += cRow @ (x_master[nzC] @ n_I)

            # Sign-consistent contact measures, valid for BOTH signs of the nodal
            # weight D_II = int(Phi_I). D_II is positive by construction for
            # CONQUAD4/8, CONTRI3/6 and CONLINE2/3, but NOT guaranteed for the
            # full-Lagrangian CONQUAD9 under partial coverage: its shape functions
            # are not pointwise non-negative, so the integral positivity required by
            # Popp, Wohlmuth, Gee & Wall (2012) can be violated there.
            #
            #   pressure  lambda_I is the multiplier in the LITERATURE convention
            #             lambda_n >= 0 in compression, i.e. the
            #             NEGATIVE slave traction. The nodal contact force along n_I
            #             is therefore -lambda_I * D_II, which points against n_I -
            #             into the slave body - for lambda_I > 0 and D_II > 0. With a
            #             negative weight the roles flip, so the physical pressure is
            #             p_n = lambda_I * sgn(D_II) >= 0 in compression for either
            #             sign of D_II.
            #   opening   Translating the master by a * n_I changes the weak gap by
            #             D_II * a (row-sum identity sum_K D_IK = sum_J C_IJ). The
            #             physical nodal opening is therefore g_weak / D_II. Dividing
            #             by the SIGNED D_II already restores exactly the property
            #             the formulation demands - "a positive weighted
            #             gap if the physical gap is positive". Multiplying by
            #             sgn(D_II) on top of that flips the sign back and makes an
            #             open node look like a penetrating one; that is a bug this
            #             code carried until it was caught by the negative-weight
            #             regression test in 06_active_set_pdass. Note the asymmetry:
            #             the pressure carries sgn(D_II), the opening does not.
            D_II = self.currentNodalWeights[I]
            sgn_D = np.sign(D_II)
            inv_D = 1.0 / D_II if abs(D_II) > self.currentWeightTolerance else 0.0
            p_n = lambda_I * sgn_D  # physical normal pressure (>= 0 in contact)
            g_sep = g_I_weak * inv_D  # physical opening (>0 open, <0 penetrating)

            if penalty:
                # ==============================================================
                # PENALTY / AUGMENTED LAGRANGIAN
                #
                # The penalty pressure is t_A = kappa * g_A, and the augmentation advances it
                # as p^(k+1) = p^k + kappa * g_A.
                #
                # g_A is the WEIGHTED gap, i.e. our g_weak up to the sign convention: the
                # non-penetration condition is usually written g_A <= 0 with the gap measured
                # from the non-mortar side towards the mortar one, while here g_sep > 0 means
                # open. g_pen below is that g_A: positive under penetration, and weighted
                # (length x area), NOT the physical opening.
                #
                # THE NODAL WEIGHT IS IN THE STIFFNESS, NOT IN THE GAP. The law enforced
                # is the pointwise one, p = eps_N * penetration, but it is evaluated on
                # the weighted gap with the node-wise stiffness kappa_I = eps_N/|D_II| --
                # the two are identical, since g^pen_I = |D_II| * penetration. Keeping the
                # WEIGHTED gap as the variable is what preserves the robustness the older
                # un-normalised form was written for: D_II is not guaranteed positive (a
                # partially covered CONQUAD9, a CONQUAD8 corner at alpha = 1/3, or the
                # sliver fallback all produce negative or near-zero weights, see the
                # negative-weight diagnostic above), and an uncovered node here simply
                # receives kappa_I = 0 from the floor in _nodalPenaltyStiffness instead of
                # an unbounded pressure from a division inside the gap.
                #
                # A SINGLE kappa FOR ALL NODES IS WRONG, and was the formulation until this
                # branch: it makes the pointwise stiffness eps_I = kappa*D_II, i.e. it gives
                # every node a stiffness proportional to its own tributary area. A corner
                # node of a plain conforming interface is then four times softer than an
                # interior one, a uniform pressure cannot produce a uniform penetration, and
                # the contact patch test fails at every finite stiffness. See
                # _nodalPenaltyStiffness for the measurement and the reference.
                #
                # The sign of the gap measure does depend on sgn(D_II) - translating
                # the master by a*n changes g_weak by D_II*a - so g_pen carries it.
                # The pressure is physical (>= 0), the assembled multiplier is
                # lambda_I = p_n * sgn(D_II), the same convention the Lagrange branch
                # uses, so that the force lambda_I * D_II * n is compressive for
                # either sign of the weight.
                #
                # ACTIVE SET. p_trial > 0 IS the Signorini branch test here; no
                # complementarity parameter and no primal-dual iteration are
                # involved. With augmentedMultipliers = 0 it reduces to "penetrating" and the law
                # to the pure penalty form; with the augmentation it is the release
                # condition of the outer loop. Note that
                # this indicator has the same structure as the semi-smooth NCP
                # indicator s_n = p_n - c_n*g_sep of the Lagrange branch, with the
                # augmented multiplier in place of the multiplier unknown - the
                # augmented Lagrangian and the semi-smooth reformulation are the
                # same object seen from two sides.
                # Consequently there is no active-set freezing and no anti-cycling
                # bookkeeping in this branch: the set is not iterated, it follows
                # the pressure.
                # ==============================================================
                self.currentWeakGap[I] = g_I_weak
                g_pen = -g_I_weak * sgn_D
                kappa_I = nodalPenalty[I]
                p_trial = self.augmentedMultipliers[I] + kappa_I * g_pen
                # useActiveSet = False pins the branch from outside - the same
                # switch the Lagrange branch honours. It is what lets a consistency
                # check perturb the state without the branch flipping underneath it,
                # and what a deliberately tied (bilateral) contact would use.
                if self.useActiveSet:
                    # A CLOSED node is an active node. The branch test is against the noise floor
                    # of the gap the pressure is made from (see currentGapTolerance), and the band
                    # |g_pen| <= currentGapTolerance -- a gap that is zero up to the rounding of the
                    # geometry -- counts as closed, not as open.
                    #
                    # The penalty law p = max(0, p_trial) has a kink at p_trial = 0, and a
                    # semi-smooth Newton method may use ANY element of its generalized derivative
                    # there: Ito & Kunisch (2003), Proposition 1.1, give the generalized derivative
                    # of max(0, y) as 1 for y > 0, 0 for y < 0 and an ARBITRARY delta at y = 0,
                    # proved in Hintermueller, Ito & Kunisch (2002). The choice is free for convergence
                    # theory and anything but free in practice. Taking 0 -- which is what the
                    # test `p_trial > kappa*tol` did -- removes every closed node from the
                    # tangent, and a body carried by the contact alone then has a singular
                    # tangent the moment its gap is exactly closed: at a cold start from a
                    # touching configuration, and at the start of every increment after a
                    # load-free one, which resolves an initial interference to EXACTLY zero.
                    # Measured on the two-block self-weight case of the contact study (C3D20R,
                    # conforming): all 65 nodes inactive at the start of increment 1, a first
                    # Newton correction of 2e+10, and although the iteration recovers, the
                    # cancellation leaves rounding of order 1e-5 in the near-zero-energy modes of
                    # the reduced-integrated elements -- a 5-7 % pressure scatter, a non-planar
                    # interface and lateral displacements, on a problem whose exact solution is
                    # a uniform pressure. With the kink assigned to the active side the tangent
                    # keeps the contact spring and the same run is exact to round-off.
                    #
                    # The pressure itself stays max(0, p_trial): a closed node carries the
                    # stiffness but no tension. Inside the band that makes the tangent differ
                    # from the residual by at most kappa*currentGapTolerance, which is the noise
                    # floor itself.
                    # kappa_I > 0: a node without coverage has no spring at all (see
                    # _nodalPenaltyStiffness) and is not closed but absent.
                    self.activeSet[I] = bool(kappa_I > 0.0 and p_trial >= -kappa_I * self.currentGapTolerance)

                if not self.activeSet[I]:
                    self.nodalMultipliers[I] = 0.0
                    if friction:
                        # An open node carries no traction, and its history goes with it.
                        self._friction.markOpen(I)
                    continue

                lambda_I = max(p_trial, 0.0) * sgn_D
                self.nodalMultipliers[I] = lambda_I

                # The whole nodal contribution is rank one. With the weights
                #   w_a = +D_IK on the slave nodes, w_a = -C_IJ on the master nodes,
                # the weighted gap is g_weak = -sum_a w_a (x_a . n_I), so
                #   dg_weak/dx_a = -w_a n_I,    dlambda_I/dx_a = +kappa_I w_a n_I
                # (the sgn(D_II) cancels), and with v_a = w_a n_I:
                #   PExt_a -= lambda_I * v_a,     K_ab += kappa_I * v_a v_b.
                # K is therefore symmetric positive semi-definite - this is the
                # first term of the linearised penalty force; the second and
                # third terms are the linearizations of the nodal normal and of the
                # mortar weights, which the frozen geometry of this implementation
                # deliberately omits (see the note on the reference configuration
                # above and the documentation chapter on consistent linearization).
                # astype: both index arrays can be empty (a slave node that lost
                # all coverage while still carrying an augmented pressure), and
                # concatenating two empty arrays does not reliably keep an integer
                # dtype - which np.ix_ below would reject.
                node_ids = np.concatenate((nzD, nSlave + nzC)).astype(np.intp)
                w = np.concatenate((dRow, -cRow))
                idcs = (sf * node_ids[:, None] + _dim_offsets).ravel()
                v = (w[:, None] * n_I[None, :]).ravel()
                PExt[idcs] -= lambda_I * v
                K[np.ix_(idcs, idcs)] += kappa_I * np.outer(v, v)
                if friction:
                    self._assembleFriction(
                        I, w, v, idcs, n_I, max(p_trial, 0.0), kappa_I * sgn_D, abs(D_II), dU, PExt, K
                    )
                continue

            self.nodalMultipliers[I] = lambda_I
            self.currentWeakGap[I] = g_I_weak

            if self.useActiveSet and not self.activeSetFrozen:
                # Semi-smooth normal complementarity: the Signorini conditions p_n >= 0,
                # g_sep >= 0,
                # p_n*g_sep = 0 are written as the single non-smooth function
                #   C_n = p_n - max(0, p_n - c_n*g_sep) = 0,
                # whose two branches are
                #   active   (s_n > 0):  constraint  g_weak = 0,
                #   inactive (s_n <= 0): constraint  lambda = 0,
                # with the augmented indicator s_n = p_n - c_n*g_sep. It is
                # re-evaluated EVERY Newton iteration - this is the literature-
                # standard PDASS = semi-smooth-Newton formulation with local
                # superlinear convergence (Hintermueller, Ito & Kunisch 2002, who
                # identify the primal-dual active set strategy AS a semi-smooth
                # Newton method; De Luca, Facchinei & Kanzow 1996 for the
                # complementarity function itself; Hueber & Wohlmuth 2005, Eq. (3.9)
                # and Gitterle et al. 2010, Eq. (55) for this contact form of it).
                # c_n is purely algorithmic: at convergence g_sep -> 0, so the
                # converged result is c_n-independent and identical to any admissible
                # active-set rule.
                #
                # Which gap measure enters the indicator is a choice, and the one made here
                # is the POINTWISE gap, i.e. a length.
                # g_sep = g_weak/D_II is exactly that: the physical nodal opening.
                # This is a DEPARTURE from the sources just named: Gitterle et al.
                # Eq. (55) reads C_nj = z_nj - max(0, z_nj - c_n*g~_j) with g~_j the
                # mortar-WEIGHTED gap of their Eq. (36), which carries length x area.
                # Both are admissible - the
                # indicator only decides the branch, and at convergence either gap
                # vanishes - but only the length-valued form makes c_n*g_sep a
                # pressure comparable to p_n, and only for it is the recommendation
                # c_n ~ O(E) dimensionally meaningful.
                #
                # The branch at the kink s_n = 0 is a choice as well, and the same one the
                # penalty branch makes (see there): a CLOSED node with no pressure yet --
                # lambda = 0 and a gap that is zero up to the rounding of the geometry -- is
                # put on the ACTIVE side. Any element of the generalized derivative is
                # admissible for the semi-smooth Newton method at the kink (Ito & Kunisch 2003,
                # Proposition 1.1: an arbitrary delta where the argument of max vanishes), but only this one
                # keeps a body that is carried by the contact alone from having a singular
                # tangent when it starts exactly touching: with `s_n > 0` every node of such a
                # start is inactive, its multiplier row decouples, and the first Newton
                # correction of the single-block self-weight case of the contact study was
                # 1.4e+10. The iteration recovers, but the cancellation leaves a 1.7-2.4 %
                # pressure scatter on C3D20R where the exact pressure is uniform. The band is
                # the noise floor of the weighted gap, carried over to the opening g_sep.
                s_n = p_n - self.c_n * g_sep
                self.activeSet[I] = bool(s_n > -self.c_n * self.currentGapTolerance * abs(inv_D))

            # Assembly in the literature sign convention (lambda_n >= 0 in
            # compression). Throughout, K = -dPExt/dU, which is what makes the
            # multiplier block symmetric:
            #   slave force   -lambda_I * D_IK * n_I     -> K[x_s, lambda] = +D_IK*n
            #   master force  +lambda_I * C_IJ * n_I     -> K[x_m, lambda] = -C_IJ*n
            #   lambda row    +g_weak (active)           -> K[lambda, x_s] = +D_IK*n,
            #                                               K[lambda, x_m] = -C_IJ*n
            # The active lambda row therefore carries the OPPOSITE overall sign to
            # the one it had while lambda was the slave traction; that is a row
            # scaling by -1 and is exactly what keeps K[x, lambda] = K[lambda, x].
            # The inactive row (lambda = 0) has no coupling and is unaffected.
            if self.activeSet[I]:
                PExt[idx_LM_I] += g_I_weak

                for K_nd, D_IK in zip(nzD, dRow):
                    s_dofs = slice(sf * K_nd, sf * K_nd + dim)
                    D_IK_n = D_IK * n_I
                    PExt[s_dofs] -= lambda_I * D_IK_n
                    K[s_dofs, idx_LM_I] += D_IK_n
                    K[idx_LM_I, s_dofs] += D_IK_n

                for J, C_IJ in zip(nzC, cRow):
                    m_global = nSlave + J
                    m_dofs = slice(sf * m_global, sf * m_global + dim)
                    C_IJ_n = C_IJ * n_I
                    PExt[m_dofs] += lambda_I * C_IJ_n
                    K[m_dofs, idx_LM_I] -= C_IJ_n
                    K[idx_LM_I, m_dofs] -= C_IJ_n
            else:
                PExt[idx_LM_I] -= lambda_I
                K[idx_LM_I, idx_LM_I] += 1.0

        # Termination of the semi-smooth active-set iteration. The discrete state
        # is the normal active set, updated every Newton iteration. Two
        # literature-grounded freeze triggers:
        #  (1) the active set has settled,
        #      i.e. is unchanged for two consecutive iterations past a warm-up.
        #  (2) anti-cycling: the state CHANGED this iteration but
        #      revisits a state already seen earlier this increment -> a proven
        #      limit cycle (only finitely many states, so a revisit-after-change
        #      is a cycle). Freezing breaks it; the remaining Newton iterations
        #      are a linear solve on the fixed set.
        # A hard iteration cap remains as a final safeguard.
        self._freezeActiveSetIfSettled(penalty, timeStep)
        if friction:
            self._freezeSlipStateIfChattering()
        self._reportThinlyCoveredNodes(timeStep)

    def _freezeActiveSetIfSettled(self, penalty: bool, timeStep: TimeStep):
        """Decide whether the active set may stop being re-evaluated for the rest of this increment.

        Three triggers, in the order they are tested: the set has reproduced itself for two
        iterations (the PDASS criterion, which is the one that normally fires), a discrete state
        already seen in this increment has recurred after a change (a proven limit cycle, since
        there are only finitely many states), or the iteration cap is reached. Only the last is
        reported, because only it leaves the set unconfirmed.

        Parameters
        ----------
        penalty
            Whether the penalty branch is active, which has no active set to freeze.
        timeStep
            The current time step, for the increment number in the diagnostic.
        """
        if self.useActiveSet and not self.activeSetFrozen and not penalty:
            state = self.activeSet.tobytes()
            changed = state != self._last_state
            if changed:
                self.activeSetStableCount = 0
            else:
                self.activeSetStableCount += 1
            cycle = changed and (state in self._seen_states)
            self._seen_states.add(state)
            self._last_state = state
            settled = self.activeSetStableCount >= 2 and self.currentIteration >= 2
            hit_cap = self.currentIteration >= 20
            if settled or cycle or hit_cap:
                if hit_cap and not settled and not cycle:
                    # Measured over the Control_Tests and patch-test suite: this
                    # never fires - the set freezes 111 times out of 111 through the
                    # PDASS criterion, at most 11 iterations. Where it DOES fire the
                    # set was still moving, so the increment converges onto a set
                    # that was never confirmed and the converged solution may
                    # violate the Signorini conditions. Nothing checks that
                    # automatically; a converged model has to be checked by comparing
                    # its nodal pressures and openings against each other.
                    self._warnOnce(
                        "active_set_iteration_cap",
                        f"the active set was frozen by the iteration cap (20), not by the PDASS "
                        f"convergence criterion, first in increment {timeStep.number}. The set had "
                        f"not settled, so the converged solution of such increments is not "
                        f"guaranteed to satisfy the Signorini conditions - verify it by checking "
                        f"the nodal pressures and openings of the converged state against each other.",
                    )
                self.activeSetFrozen = True

    def _reportThinlyCoveredNodes(self, timeStep: TimeStep):
        """Report non-mortar nodes in contact whose own facet area is almost entirely unopposed.

        Parameters
        ----------
        timeStep
            The current time step, for the increment number in the diagnostic.
        """
        # A node that carries a constraint while almost none of its own facet area is opposed by the
        # other surface is worth saying out loud. The constraint is exact there and the weighting of
        # the boundary rows keeps its multiplier an ordinary pressure, so nothing is broken - but the
        # force that crosses at such a node is set by the surrounding mechanics rather than by the
        # sliver it stands on, and on a surface that ends while still under full pressure that force
        # is not small. Putting the SMALLER of the two surfaces on the non-mortar side avoids the
        # situation altogether, since every one of its facets is then fully covered.
        #
        # Checked here rather than at the start of an increment, and against the active set of THIS
        # assembly: the condition can develop within the last increment of a run, where an
        # increment-boundary check would never see it. Restricted to nodes actually in contact,
        # because a thinly covered node that carries no multiplier transmits nothing - a curved
        # indenter has such nodes just outside its contact zone in every increment, and reporting
        # those would bury the case that matters.
        #
        # The bound decides when to speak and nothing else. No quantity entering the solution is
        # compared against it.
        if hasattr(self, "currentCoverage"):
            thinlyCovered = np.flatnonzero(
                (self.currentCoverage < 1e-2) & (self.currentCoverage > 0.0) & self.activeSet
            )
            if len(thinlyCovered):
                self._warnOnce(
                    "thinly_covered_nodes",
                    f"{len(thinlyCovered)} non-mortar node(s) in contact are opposed over less than "
                    f"1 % of their own facet area (thinnest: "
                    f"{float(np.min(self.currentCoverage[thinlyCovered])):.2e}), first in increment "
                    f"{timeStep.number}. Such a node is constrained exactly like a fully covered one "
                    f"- coverage sets how the contact force is distributed, never whether a node is "
                    f"tied - so it can transmit a force out of proportion to the area backing it. "
                    f"This arises where the mortar surface ends while still transmitting pressure. "
                    f"Putting the smaller of the two surfaces on the non-mortar side removes it, "
                    f"because every non-mortar facet is then fully covered.",
                )

    def _nodalPenaltyStiffness(self, modulus: float | None = None) -> np.ndarray:
        """The stiffness each non-mortar node's penalty spring actually carries.

        ``modulus`` is the pointwise penalty modulus to normalise, eps_N when omitted; the
        friction law passes eps_T through the same normalisation, for the same reasons.

        THE ONE PLACE THE NORMALISATION OF THE WEIGHTED GAP LIVES. The penalty law of this
        constraint is a pressure per unit POINTWISE opening,

            p_I = eps_N * (-g_I / D_II) ,

        and since the assembly works with the weighted gap ``g_I`` throughout -- which is
        the right variable, for the robustness reason below -- the division is folded into
        a per-node stiffness instead:

            kappa_I = eps_N / |D_II| ,     so that   kappa_I * g^pen_I = eps_N * penetration.

        WHY THIS IS NOT A COSMETIC CHOICE. Without it the whole constraint runs on a single
        kappa multiplying the weighted gap, which gives every node the pointwise stiffness
        ``eps_I = kappa * D_II`` -- proportional to its own nodal weight. On a plain 4x4
        conforming interface that is a factor of FOUR between a corner node and an interior
        one, so a uniform pressure cannot produce a uniform penetration and the interface
        does not stay plane: the contact patch test fails at every finite stiffness, on a
        conforming mesh, for purely discrete reasons. Measured on the 03_TwoBlocksPressure
        patch test of the contact study, the relation ``penetration_I = p_I/(kappa D_II)``
        held to 2e-16 -- it is the formulation, not a numerical artefact.

        The scale factor is Yang, Laursen & Meng (2005), Eqs. (36)-(37): their zeta_A is
        ``1/sum_D n_AD = 1/D_AA``, introduced, in their words, "to cause the gap function
        g_A to have the proper units of length, which is of crucial importance when
        implementing penalty methods in particular". Their Section 8.1 passes the patch
        test on non-conforming meshes to machine precision with a penalty.

        THE ROBUSTNESS OBJECTION IS REAL AND IS HANDLED HERE RATHER THAN AVOIDED. D_II is
        not guaranteed positive -- a partially covered CONQUAD9, a CONQUAD8 corner at
        alpha = 1/3 and the sliver fallback all produce negative or near-zero weights -- so
        putting it in a denominator needs a floor. It gets the same relative floor the
        physical opening ``g_sep`` already uses (:attr:`currentWeightTolerance`), and a node
        beneath it carries NO penalty spring at all rather than an enormous one. That is
        also the physically right answer: a node with no coverage transmits no force, and
        in the weighted form it received ``kappa * g_I`` with ``g_I -> 0`` anyway.

        Only the MAGNITUDE of the weight enters. The sign is already carried by ``g^pen``
        and by the assembled multiplier, exactly as before; this changes how hard the
        spring is, not which way it pushes.
        """

        weights = np.asarray(self.currentNodalWeights, dtype=float)
        magnitude = np.abs(weights)
        covered = magnitude > self.currentWeightTolerance
        stiffness = np.zeros_like(magnitude)
        stiffness[covered] = (self.epsN if modulus is None else modulus) / magnitude[covered]
        return stiffness

    def _tangentPenaltyStiffness(self) -> float:
        """The pointwise tangential penalty modulus eps_T: the given one, or eps_N as resolved."""

        return self._epsTGiven if self._epsTGiven > 0.0 else self.epsN

    # The frictional state lives in the shared law (self._friction); these keep the names under which
    # the mortar constraint has always exposed it, for output, tests and diagnostics.

    @property
    def frictionTractionConverged(self) -> np.ndarray:
        """The accepted tangential traction per non-mortar node, the history of the stick predictor."""
        return self._friction.tractionConverged

    @property
    def frictionTractionCurrent(self) -> np.ndarray:
        """The tangential traction per non-mortar node at the current iterate."""
        return self._friction.tractionCurrent

    @property
    def frictionPressureConverged(self) -> np.ndarray:
        """The pressure that went with the accepted traction."""
        return self._friction.pressureConverged

    @property
    def frictionActiveConverged(self) -> np.ndarray:
        """Whether each non-mortar node was in contact at the accepted state."""
        return self._friction.activeConverged

    @property
    def slipState(self) -> np.ndarray:
        """The stick/slip state per non-mortar node: 0 open, 1 stick, 2 slip."""
        return self._friction.slipState

    @property
    def slipStateFrozen(self) -> bool:
        """Whether the stick/slip state of the current increment attempt is frozen."""
        return self._friction.slipStateFrozen

    @slipStateFrozen.setter
    def slipStateFrozen(self, frozen: bool):
        self._friction.slipStateFrozen = frozen

    def _freezeDirections(self, nodes: np.ndarray):
        """Hold the given slipping nodes in slip along their current traction direction."""
        self._friction.freezeDirections(nodes)

    @property
    def slipStateFreezeCount(self) -> int:
        """How many increments needed the stick/slip state frozen; a diagnostic only."""
        return self._friction.slipStateFreezeCount

    @property
    def _frozenSlip(self) -> np.ndarray:
        return self._friction.frozenSlip

    @property
    def _frozenDirection(self) -> np.ndarray:
        return self._friction.frozenDirection

    @property
    def _elasticSlip(self) -> float | None:
        return self._friction.elasticSlip

    def _assembleFriction(
        self,
        I: int,  # noqa: E741 - the non-mortar node index of the formulation
        w: np.ndarray,
        v: np.ndarray,
        idcs: np.ndarray,
        n_I: np.ndarray,
        p: float,
        dp_dv: float,
        nodalWeight: float,
        dU: np.ndarray,
        PExt: np.ndarray,
        K: np.ndarray,
    ):
        """Penalty-regularised Coulomb friction at one active non-mortar node.

        The law is the one shared by all contact formulations
        (:class:`~edelweissfe.utils.contactfriction.CoulombFriction`): that of Yang, Laursen & Meng
        (2005), Eqs. (39), (47)-(49), in the 3D form of Puso, Laursen & Solberg (2008), Eqs.
        (16)-(21), with the return map of Wriggers (2006), Box 14. What is mortar-specific is only how
        a non-mortar node maps onto it. With the weights ``w_a`` the weighted gap is built from (+D_IK
        on the slave, -C_IJ on the master nodes), the node is a contact point with

            c = w / |D_II|,     A = |D_II|,     n = n_I,     p,  dp/dU = dp_dv * v ,

        so that its pointwise slip increment ``P_T (c (x) I) dU`` is the mortar-weighted slip
        ``P_T sum_a w_a dU_a`` over |D_II| -- eps_T times it is a traction, exactly as the normal
        pressure is eps_N times the pointwise penetration -- and its nodal forces ``A c (x) t =
        w (x) t`` are the gradient of the convex stick energy, so the sign of D_II needs no special
        treatment: only its magnitude enters.

        WITHIN AN INCREMENT THE SLIP IS THE FROZEN-GEOMETRY ONE, ``D dU_s - C dU_m`` with D, C and n
        held at the last converged state, so that the assembled tangent is exact for the residual
        that is assembled, as for the normal part. It is not objective on its own: a rigid rotation
        theta of a penetrating pair turns the weighted gap vector g and reads as a slip
        ``P_T (R - I) g``, i.e. a spurious traction of about ``(eps_T/eps_N) theta p``. The objective
        slip of Yang et al. (2005), Eq. (47), ``-(D_n+1 - D_n) x_s + (C_n+1 - C_n) x_m``, differs
        from it by exactly the change of the gap vector, and the coupling matrices of the end of the
        increment are what the next rebuild computes -- so :meth:`_carryFrictionHistory` removes that
        difference from the accepted history there. The error is then that of the frozen normal,
        first order in one increment and not accumulated over many.

        "No pressure" is the noise floor of the gap the pressure is made from (currentGapTolerance),
        the same band in which the normal law counts a node as closed but unloaded.

        Parameters
        ----------
        I
            The non-mortar node.
        w, v, idcs
            The weight row, ``w (x) n_I``, and the local degrees of freedom of this node's coupling.
        n_I
            The frozen nodal normal.
        p
            The physical contact pressure, >= 0.
        dp_dv
            The derivative of p along v, i.e. ``dp/dU = dp_dv * v``.
        nodalWeight
            |D_II|, which turns the pointwise modulus into the nodal stiffness.
        dU, PExt, K
            As in :meth:`applyConstraint`.
        """

        # idcs holds exactly the dim displacement components of every coupled node, in node order.
        forces, tangent = self._friction.evaluatePoint(
            I,
            w / nodalWeight,
            nodalWeight,
            n_I,
            p,
            dp_dv * v,
            dU[idcs],
            self.epsN,
            pressureTolerance=abs(dp_dv) * self.currentGapTolerance,
        )
        PExt[idcs] += forces
        K[np.ix_(idcs, idcs)] += tangent

    def _freezeSlipStateIfChattering(self):
        """Freeze the stick/slip state of this increment once it chatters, by the rule of
        :meth:`~edelweissfe.utils.contactfriction.CoulombFriction.freezeIfChattering`, counted in
        this constraint's own Newton iterations. The friction decks of the test suite settle within
        five iterations, so this only fires on an increment that is genuinely struggling."""

        self._friction.freezeIfChattering(self.currentIteration)

    def _carryFrictionHistory(self, U_ref: np.ndarray, previousNormals: np.ndarray | None):
        """Carry the accepted tangential tractions into a rebuilt geometry, with the objective slip.

        Within an increment the slip is measured on the frozen geometry (see
        :meth:`_assembleFriction`). Between two rebuilds that measure sums to

            sum s_frozen = D_old (x_new - x_old)_s - C_old (x_new - x_old)_m ,

        while the objective slip of Yang, Laursen & Meng (2005), Eq. (47), integrated over the same
        interval, is ``-(D_new - D_old) x_new,s + (C_new - C_old) x_new,m``. Their difference is
        exactly the change of the weighted gap VECTOR ``g = D x_s - C x_m`` from one rebuild to the
        next, which both sides of it can compute. The accepted traction is therefore corrected by

            t  <-  P_T,new ( t + eps_T (g_new/|D_II,new| - g_old/|D_II,old|) ) ,

        and returned onto the cone of the accepted pressure if that takes it outside. The pointwise
        normalisation by |D_II| is the one the traction carries throughout (kappa_T = eps_T/|D_II|).

        What it removes is the spurious slip of a rotation: a rigidly rotating, penetrating pair
        keeps D and C and turns g, and the frozen measure reads the turn as a slip of the penalty
        gap. Under a pure slide of a flat pair g does not change and the correction is zero, so a
        genuine slip is left as it was measured.

        Two nodes are not carried. A node whose normal turned by more than 30 degrees across the
        rebuild -- a crease of the mortar side -- loses its history, since projecting it would drop
        the part that now points along the new normal and leave a predictor unrelated to the
        frictional state; it enters like a fresh contact, which is harmless where it matters (a
        sticking node does not cross a crease, a slipping one is returned onto the cone within the
        increment). A node whose gap vector is not defined on both sides of the rebuild -- no weight,
        or not in contact at the accepted state -- is left as it is.

        Repeating a rebuild at the same configuration, as a cutback re-attempt may, changes nothing,
        since the reference gap vector is updated here as well.

        Parameters
        ----------
        U_ref
            The configuration the geometry was just rebuilt at, the last accepted one.
        previousNormals
            The nodal normals of the previous rebuild, ``None`` at the first.
        """

        nSlave = self.nNonMortarNodes
        dim = self.model.domainSize
        sf = self.sizeField
        nNodes = len(self._nodes)
        coords = self._X + U_ref[: sf * nNodes].reshape(nNodes, sf)[:, :dim]
        x_slave, x_master = coords[:nSlave], coords[nSlave:]

        weights = np.abs(np.asarray(self.currentNodalWeights, dtype=float))
        gap = np.full((nSlave, dim), np.nan)
        for I in range(nSlave):  # noqa: E741 - I is the non-mortar node index of the formulation
            if weights[I] <= self.currentWeightTolerance:
                continue
            nzD, nzC = self.currentDNonzero[I], self.currentCNonzero[I]
            g = np.zeros(dim)
            if len(nzD):
                g += self.currentDRow[I] @ x_slave[nzD]
            if len(nzC):
                g -= self.currentCRow[I] @ x_master[nzC]
            gap[I] = g / weights[I]

        previousGap = self.frictionGapReference.copy()
        self.frictionGapReference[:] = gap

        # NaN where the gap vector is not defined on both sides: such a node is left as it is.
        self._friction.carry(self.currentNormals, previousNormals, deltaGap=gap - previousGap, normalPenalty=self.epsN)

    def acceptLastState(self):
        """Promote the tangential tractions of the converged iterate to the frictional history.

        Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` once an increment is
        accepted -- after the augmented outer loop, if any, has finished -- and under explicit
        dynamics after every time step. A cutback never reaches it, so the history of a failed
        attempt cannot leak into the next one.

        If the stick/slip state was frozen in this increment, the accepted state is also checked for
        what the freeze may have cost: a node held in slip whose predictor lies inside the cone
        should have stuck, and carries mu*p instead of the smaller stick traction. That is reported
        once, with the worst ratio, rather than refused -- the traction is still on the cone.
        """

        if not self.friction:
            return
        insideCone, turnedAway = self._friction.accept()
        if insideCone or turnedAway:
            self._warnOnce(
                "slip_state_frozen_inadmissible",
                f"the stick/slip state chattered and was frozen, and some nodes were accepted in slip although "
                f"their stick predictor lies inside the friction cone ({insideCone}) or points more than 30 "
                f"degrees away from the frozen slip direction ({turnedAway}). Those nodes carry the full "
                f"friction mu*p in the frozen direction; the traction never exceeds the cone. Smaller "
                f"increments, or a larger slipTolerance, remove it.",
            )

    def getNormalPressures(self) -> np.ndarray:
        """The contact pressure of each non-mortar node, >= 0 in compression, as of the last assembly.

        Ordered like the node set ``<name>_nonMortarNodes`` (see :meth:`_publishNonMortarNodeSet`).
        In the penalty branch it is eps_N times the pointwise penetration (plus the augmented
        estimate); in the multiplier branch it is the multiplier, with the sign of the nodal weight
        taken out.
        """

        weights = getattr(self, "currentNodalWeights", None)
        sign = np.sign(weights) if weights is not None else np.ones(self.nNonMortarNodes)
        return self.nodalMultipliers * np.where(sign == 0.0, 1.0, sign)

    def getTangentialTractions(self) -> np.ndarray:
        """The frictional traction of each non-mortar node, a pressure-valued vector in global
        components, as of the last assembly; zero without friction.

        It is the traction the MORTAR side exerts on the non-mortar side, i.e. it points along the
        motion of the mortar body relative to the non-mortar one. Ordered like
        ``<name>_nonMortarNodes``.
        """

        return self.frictionTractionCurrent.copy()

    def getSlipStates(self) -> np.ndarray:
        """The stick/slip state of each non-mortar node as of the last assembly, as a float for output:
        0 open, 1 stick, 2 slip. Ordered like ``<name>_nonMortarNodes``. Always 0 without friction."""

        return self.slipState.astype(float)

    def getGaps(self) -> np.ndarray:
        """The pointwise normal opening of each non-mortar node, ``g_I / D_II``: positive open,
        negative penetrating, 0 where the node has no weight. Ordered like ``<name>_nonMortarNodes``."""

        weights = getattr(self, "currentNodalWeights", None)
        if weights is None:
            return np.zeros(self.nNonMortarNodes)
        covered = np.abs(weights) > self.currentWeightTolerance
        return np.divide(self.currentWeakGap, weights, out=np.zeros(self.nNonMortarNodes), where=covered)

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """The converged frictional history, the augmented pressure estimate and, under
        formulation=lagrange, the multipliers -- each per non-mortar node, together with the node
        labels they belong to.

        Everything else -- the frozen geometry, the active set, the derived stiffnesses -- is rebuilt
        from the restored solution at the first assembly after a restart. The reference gap vector
        comes along so that the objective correction of the first rebuild after the restart has
        something to compare against. The multipliers are also scalar variables of the model, but
        those are restored by name, i.e. by index, which after a refinement no longer identifies a
        node; the labels do.
        """

        data = {
            **self._friction.restartData(),
            "frictionGapReference": self.frictionGapReference,
            "augmentedMultipliersConverged": self.augmentedMultipliersConverged,
            "nonMortarNodeLabels": np.array([node.label for node in self.nonMortarNodes], dtype=int),
        }
        if self.formulation == "lagrange":
            data["lagrangeMultipliers"] = np.array([variable.value for variable in self.scalarVariables])
        return data

    def setRestartData(self, data: dict[str, np.ndarray]):
        """Restore the state of :meth:`getRestartData`.

        Called by :meth:`~edelweissfe.models.femodel.FEModel.readRestart` after the topology replay
        of an AMR run has re-created the refined mesh, but before any solver has refreshed the mesh
        dependents -- so the constraint first catches up with the refined surfaces itself, and then
        matches the checkpoint to its non-mortar nodes by label. A checkpoint without labels is taken
        to be in the current node order; one written before the objective slip correction existed
        holds only the traction, and the correction then starts at the first rebuild after the
        restart instead.
        """

        self.refreshIfMeshChanged(self.model)

        if "nonMortarNodeLabels" in data:
            current, stored = matchRestartKeys(
                [int(label) for label in data["nonMortarNodeLabels"]], [node.label for node in self.nonMortarNodes]
            )
        else:
            current = stored = np.arange(self.nNonMortarNodes)

        self._friction.setRestartData(data, current, stored)
        if "frictionGapReference" in data:
            self.frictionGapReference[current] = np.asarray(data["frictionGapReference"])[stored]
        augmented = np.asarray(data["augmentedMultipliersConverged"])[stored]
        self.augmentedMultipliersConverged[current] = augmented
        self.augmentedMultipliers[current] = augmented
        if "lagrangeMultipliers" in data and self.formulation == "lagrange":
            multipliers = np.asarray(data["lagrangeMultipliers"])
            for i, j in zip(current, stored):
                self.scalarVariables[i].value = multipliers[j]

    def _stabilityFactor(self, lumpedMass: np.ndarray) -> float:
        """The stiffness-independent part R of this constraint's stability bound.

        The penalty contribution of one slave node is RANK ONE, ``K_I = kappa_I * v_I v_I^T``
        with ``v_I = w_I (x) n_I`` (see applyConstraint) and the NODE-WISE stiffness
        ``kappa_I = eps_N/|D_II|`` of :meth:`_nodalPenaltyStiffness`, so the contact stiffness
        assembles as ``K = eps_N * sum_I (1/|D_II|) v_I v_I^T``. Gershgorin's theorem bounds the
        largest eigenvalue of ``M^-1 K`` by the largest scaled absolute row sum, which for that
        form is

            omega^2 <= eps_N * max_a (1/m_a) * sum_I (1/|D_II|) |v_I,a| * ||v_I||_1  =:  eps_N * R

        The ``1/|D_II|`` sits INSIDE the sum over the nodes and cannot be pulled out of it: the
        weights differ by a factor of four across an ordinary interface, and a bound taken with a
        mean weight would be wrong by that factor at the stiffest node -- which is precisely the
        node that sets the stable increment.

        R is returned. It carries the geometry and the inertia and not the stiffness, which is
        what makes it usable in both directions: with eps_N known it gives the stable time step
        ``dt <= 2/sqrt(eps_N*R)``, and with the time step known it gives the largest stable
        stiffness ``eps_N <= 4/(dt^2 R)``. Both follow from the central-difference stability
        limit ``dt_cr = 2/sqrt(lambda_max)`` of Kwon, Kim, Cho & González (2024), Eqs. (20)-(21),
        whose stability-assured condition (their Eq. 22) is exactly ``omega^2 <= lambda_max``.

        Gershgorin OVERestimates the eigenvalue, so the bound is conservative: it may ask for a
        slightly smaller time step, or a slightly softer stiffness, than strictly necessary. That
        is the right direction to err in, and the alternative -- the exact rank-one eigenvalue
        ``kappa * v_I^T M^-1 v_I`` -- is exact for one node in isolation but UNDERestimates as
        soon as two nodes share a degree of freedom, which is the normal case.

        Every non-mortar node is counted, not only the currently active ones: the active set
        changes from one increment to the next, and a bound that only held for the set that
        happened to be active when it was computed would not be a bound.
        """

        rows = self._stabilityRowSums(lumpedMass)
        return float(np.max(rows[0])) if rows is not None else 0.0

    def _stabilityRowSums(self, lumpedMass: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """The mass-scaled Gershgorin row sums of the normal and of the tangential contact stiffness.

        Returned per degree of freedom and free of the moduli, as ``(r_N, r_T)``, so that the bound
        of the combined stiffness is ``omega^2 <= max_a (eps_N r_N,a + eps_T r_T,a)``. The normal part
        is the one :meth:`_stabilityFactor` describes; the tangential part is the stick stiffness of
        the friction law, ``(eps_T/|D_II|) (w w^T) (x) P_T``, bounded the same way with ``|P_T|`` in
        place of ``|n n^T|``. Splitting the two is what keeps a stiff tangential spring from being
        hidden behind a normal one that loads other components: on an axis-aligned interface they
        act on disjoint rows, and eps_T = eps_N then costs no time step at all. ``r_T`` is zero
        without friction. ``None`` while no geometry exists or a touched row has no mass.
        """

        if not hasattr(self, "currentNormals"):
            return None

        nSlave = self.nNonMortarNodes
        sf = self.sizeField
        dim = self.model.domainSize
        offsets = np.arange(dim)
        rowSums = np.zeros(len(lumpedMass))
        rowSumsT = np.zeros(len(lumpedMass))

        # The stiffness-free part of kappa_I = eps_N/|D_II|. Taken directly rather than through
        # _nodalPenaltyStiffness, because this method is also called BEFORE eps_N is known --
        # that is the direction that derives it -- and must not depend on it.
        weights = np.abs(np.asarray(self.currentNodalWeights, dtype=float))
        covered = weights > self.currentWeightTolerance

        for I in range(nSlave):  # noqa: E741 - the non-mortar node index of the formulation
            nzD, nzC = self.currentDNonzero[I], self.currentCNonzero[I]
            if (len(nzD) == 0 and len(nzC) == 0) or not covered[I]:
                continue
            nodeIds = np.concatenate((nzD, nSlave + np.asarray(nzC))).astype(np.intp)
            w = np.concatenate((self.currentDRow[I], -self.currentCRow[I]))
            idcs = (sf * nodeIds[:, None] + offsets).ravel()
            v = np.abs((w[:, None] * self.currentNormals[I][None, :]).ravel())
            rowSums[idcs] += v * float(np.sum(v)) / weights[I]
            if self.friction:
                n_I = self.currentNormals[I]
                absW = np.abs(w)
                projectorRowSums = np.abs(np.eye(dim) - np.outer(n_I, n_I)).sum(axis=1)
                rowSumsT[idcs] += np.kron(absW * float(np.sum(absW)), projectorRowSums) / weights[I]

        mass = np.asarray(lumpedMass, dtype=float)
        touched = (rowSums > 0.0) | (rowSumsT > 0.0)
        if not np.any(touched) or np.any(mass[touched] <= 0.0):
            return None
        scaledN = np.zeros_like(rowSums)
        scaledT = np.zeros_like(rowSumsT)
        scaledN[touched] = rowSums[touched] / mass[touched]
        scaledT[touched] = rowSumsT[touched] / mass[touched]
        return scaledN, scaledT

    def _omegaSquaredBound(self, lumpedMass: np.ndarray) -> float:
        """Gershgorin bound of the largest contact eigenvalue of ``M^-1 K`` at the current moduli."""

        rows = self._stabilityRowSums(lumpedMass)
        if rows is None:
            return 0.0
        epsT = self._tangentPenaltyStiffness() if self.friction else 0.0
        return float(np.max(self.epsN * rows[0] + epsT * rows[1]))

    def computeCriticalTimeStepForExplicitDynamics(self, lumpedMass: np.ndarray) -> float:
        """The stable time increment this constraint's penalty stiffness permits.

        Also the point at which the lumped mass reaches this constraint: the solver assembles it
        before it asks for the time step, and nothing else hands it over. It is kept for the
        stiffness derivation in :meth:`_resolveEpsN`, which needs the same inertia.

        Returns ``inf`` until the geometry exists -- on the call before the increment loop it does
        not yet, since the segmentation needs a solution vector this hook is not given. The real
        bound is applied from the first assembly onwards, where the stiffness is checked against
        it (a user-given value) or derived from it (the default).
        """

        self._explicitLumpedMass = np.asarray(lumpedMass, dtype=float).copy()

        if self.formulation != "penalty" or not hasattr(self, "currentNormals"):
            return np.inf

        omegaSquared = self._omegaSquaredBound(self._explicitLumpedMass)
        if omegaSquared <= 0.0:
            return np.inf
        return 2.0 / np.sqrt(omegaSquared)

    def applyConstraintExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        PExt: np.ndarray,
        timeStep: TimeStep,
    ):
        """Refuse the variants an explicit solver cannot honour, then evaluate normally.

        This method is called by explicit solvers and by nothing else, which makes it the
        one place where "am I being integrated explicitly?" can be answered without the
        constraint knowing about solvers.

        ``formulation=lagrange`` is already refused by the explicit dynamic solver itself,
        whose ``validateModelCapabilities`` rejects any constraint carrying its own scalar
        variables: such degrees of freedom have no inertia, so the explicit update never
        moves them and the constraint would appear active while enforcing nothing. The
        check is repeated here because that refusal belongs to one solver, while this
        entry point is common to all of them.

        ``augmentedLagrange`` needs refusing for a different and more dangerous reason. The
        augmented update runs from :meth:`augmentConstraint`, which only the implicit
        solvers call. Under an explicit solver it is simply never invoked -- so the
        constraint would not fail, it would silently degrade to the pure penalty form and
        report nothing. A run would look like an augmented-Lagrangian result and be a
        penalty one, which is precisely the kind of quiet wrong answer that is worth an
        exception.

        PENALTY STIFFNESS IS BOUNDED FROM ABOVE HERE, in a way it is not implicitly. An explicit
        solver derives its critical time step from the MESH; the contact spring is not part of that
        estimate, so an ``penaltyStiffness`` whose own frequency exceeds the stability limit simply
        makes the integration diverge. The bound is not left to the user: it is checked against
        eps_N at every geometry rebuild and reported, and derived FROM the limit when no stiffness
        was given (see _resolveEpsN). Measured on ``testfiles/edelweiss-only/NEDMortarContact`` with
        everything else held fixed, the run is stable at eps_N = 8e4, warned about at 1.2e5 while
        still stable, and blown up at 1.6e5. Do not carry a value over from an implicit analysis --
        there the limit is conditioning, which is far weaker.

        The tangent is NOT skipped here, unlike the node-to-surface constraint's override:
        measured on this constraint, residual and tangent together are 2 to 7 % of its
        cost while the frozen geometry is the remaining 93 to 98 %, so building and
        discarding the tangent buys almost nothing and reworking the assembly to run
        without it would put the far more valuable part at risk for that.
        """

        if self.formulation == "lagrange":
            raise NotImplementedError(
                f"Constraint '{self._name}': formulation=lagrange cannot be integrated explicitly. "
                "Its multipliers are unknowns of the global system, and an explicit increment "
                "solves no system: the multiplier degrees of freedom carry no inertia and would "
                "never be updated. Use formulation=penalty."
            )
        if self.useAugmentedLagrange:
            raise NotImplementedError(
                f"Constraint '{self._name}': augmentedLagrange cannot be integrated explicitly. "
                "The augmented update runs on a converged equilibrium, which an explicit "
                "increment does not produce, so it would never run and the constraint would "
                "quietly reduce to the pure penalty form. Set augmentedLagrange=False and accept "
                "the penalty error, or run implicitly."
            )

        self._friction.checkExplicitIntegrability()

        # No implicit solver calls this method, so it is also where the constraint learns that
        # it is being integrated explicitly -- which changes how the penalty stiffness is chosen.
        self._explicit = True
        self._explicitTimeIncrement = timeStep.timeIncrement

        super().applyConstraintExplicit(U_np, dU, PExt, timeStep)

    def updateConnectivity(self, model: FEModel) -> bool:
        """Mark the frozen geometry for rebuild; the rebuild itself happens on next use.

        This is the hook through which the SOLVER, not the constraint, decides how often
        the segmentation runs -- and the two solver families need very different answers.
        An implicit solver calls this once per increment, which reproduces exactly the
        behaviour this constraint had when it detected a new increment itself. An explicit
        dynamic solver calls it only every ``contact-update-frequency`` increments, because
        a segmentation per time step is not affordable: it is 93-98 % of this constraint's
        cost, some 26 ms per rebuild on a small linear interface and ten times that on a
        quadratic one, against time steps counted in millions.

        The rebuild is deferred rather than done here because it needs the solution vector
        restricted to this constraint's degrees of freedom, which is what
        :meth:`applyConstraint` is handed and this method is not. Deferring also means a
        tick costs nothing if the constraint is never assembled afterwards.

        Returns ``False``: this constraint's DOF footprint never changes, so the solver
        need not rebuild the equation system on its account. The segmentation changes which
        nodes are COUPLED, but they are all in :attr:`nodes` from the outset either way.
        """

        self._geometryStale = True
        return False

    def _refreshFrozenGeometry(self, step_key: tuple, U_np: np.ndarray, dU: np.ndarray, timeStep: TimeStep):
        """Re-freeze the geometry at the start of an increment, or count an iteration within one.

        Everything geometric -- the nodal normals, the segmentation, and the two coupling matrices --
        is evaluated once per increment from the LAST CONVERGED configuration and then held fixed
        for that increment's Newton iterations. This is the method that decides which of the two
        has happened and, on a new increment, does the evaluating: it also resets the active-set
        bookkeeping, restores the augmented multiplier estimate, and derives the relative
        tolerances that the per-node loop compares against.

        Parameters
        ----------
        step_key
            Identifies the increment AND the attempt, so that a cutback re-freezes rather than
            reusing the geometry of the attempt that failed.
        U_np
            The current solution vector.
        dU
            The increment of it, so that ``U_np - dU`` is the last converged state.
        timeStep
            The current time step, for the increment number in the diagnostics.
        """
        nSlave = self.nNonMortarNodes

        # TWO conditions, deliberately separate. The per-increment bookkeeping below
        # follows the increment, as it always did. The GEOMETRY follows the solver's
        # connectivity tick instead, so that an explicit solver can throttle the
        # segmentation without also freezing the active-set bookkeeping, and an
        # implicit solver -- which ticks once per increment -- sees the two coincide
        # exactly as before. ``not hasattr`` is the safety net for a constraint that
        # is assembled before any solver ticked it.
        newIncrement = step_key != self.lastTimeStepKey
        rebuildGeometry = self._geometryStale or not hasattr(self, "currentNormals")

        if newIncrement:
            # The previous increment has converged and U_np - dU is its result.
            # Before anything is reset, ask whether its active set reproduces
            # itself on that result - the one moment where that is checkable.
            if hasattr(self, "currentNormals") and self.activeSetFrozen:
                self._checkConvergedActiveSet(U_np - dU)

            self.lastTimeStepKey = step_key
            self.currentIteration = 0
            # The active set (Signorini) is re-evaluated in EVERY Newton iteration
            # and the outer semi-smooth loop is converged once the set no longer
            # changes - the convergence criterion of the primal-dual active set
            # strategy, i.e. a semi-smooth Newton method. There is deliberately NO
            # fixed iteration-count cutoff (that would be an ad-hoc heuristic and
            # could freeze a not-yet-settled set); a safeguard cap only guards
            # against pathological non-settling. Reset per increment:
            self.activeSetFrozen = False
            self.activeSetStableCount = 0
            # Anti-cycling bookkeeping: the discrete states already
            # visited this increment, and the immediately previous one.
            self._seen_states = set()
            self._last_state = None
            # Restart the augmented Lagrangian for this increment ATTEMPT from the
            # last CONVERGED pressure estimate. Warm starting is what makes the outer
            # loop cheap after the first increment; resetting on a cutback re-attempt
            # is required for the same reason the frozen geometry is reset - the
            # estimate of a failed attempt belongs to a diverged iterate.
            self._augmentation_counter = 0
            self.augmentedMultipliers[:] = self.augmentedMultipliersConverged
            # The stick/slip state is free again in every increment attempt, like the active set.
            self._friction.startIncrementAttempt()
        else:
            self.currentIteration += 1

        if not rebuildGeometry:
            return

        # Everything below is the frozen geometry and the quantities derived from it.
        # It is the expensive part -- 93 to 98 % of this constraint's cost -- and runs
        # only when the solver's connectivity tick has marked it stale.
        self._geometryStale = False
        # Last converged state - see the reference-configuration note above.
        U_ref = U_np - dU
        previousNormals = getattr(self, "currentNormals", None)
        self.currentNormals = self.computeNormals(U_ref)
        D_full, C_full = self.computeMortarCouplingMatrices(U_ref)
        # The FULL (element-locally sparse) D matrix is used for forces,
        # stiffness and weak gap. With the basis transformation T_e the
        # biorthogonality holds w.r.t. N_tilde, so D is not diagonal for
        # quadratic elements - lumping it would destroy the consistency of
        # the contact force distribution (a constant pressure could not be
        # transmitted exactly, i.e. the patch test would fail). This is
        # algebraically equivalent to the transformed formulation of Popp,
        # Wohlmuth, Gee & Wall (2012) of the condensed system.
        # Translational invariance of the weak gap is guaranteed by the
        # row-sum identity sum_K D_IK = sum_J C_IJ (same-domain integration).
        self.currentD = D_full
        self.currentC = C_full
        # Positive by construction: sum_K D_IK = int(Phi_I) = int(N_tilde_I) > 0
        self.currentNodalWeights = np.asarray(D_full.sum(axis=1)).ravel()
        # The sign-consistent contact measures below carry a negative D_II
        # correctly, but the weighted gap loses its reading as a mean opening
        # there - so it is worth saying out loud that it happened.
        # Only for weights that are meaningfully negative, though: a node far
        # outside the covered region integrates to a value that is zero up to
        # round-off, and whether that lands at +1e-19 or -1e-19 says nothing.
        # Measured range of the real cases: -0.031 (CONQUAD9 at 70 % coverage,
        # 06_active_set_pdass) down to -0.60 (sliver fallback,
        # hertz_hex20_medium), so a relative threshold of 1e-6 separates them
        # from the dust by orders of magnitude.
        weights = self.currentNodalWeights
        weight_scale = np.max(np.abs(weights)) if len(weights) else 0.0
        # Threshold below which a nodal weight counts as "no weight at all" and
        # the division 1/D_II is suppressed. RELATIVE, for the same reason as in
        # _nonzeroRows: D_II = int(Phi_I) carries the unit of an area, so an
        # absolute bound only ever fits one system of units. It also has to be
        # generous rather than tiny: at D_II = 1e-25 an absolute bound of 1e-30
        # would still divide, g_sep = g_weak/D_II would explode by 25 orders, and
        # the sign of that garbage would decide the branch of the NCP indicator -
        # a node can be switched ACTIVE by pure round-off that way, with a
        # constraint row that means nothing. Relative to the largest weight of the
        # interface, 1e-12 is far below any weight a covered node can have (the
        # smallest measured over the test suite is ~1e-3 of the largest) and far
        # above the dust of an uncovered one.
        self.currentWeightTolerance = 1e-12 * weight_scale
        # Noise floor of the WEIGHTED GAP, by the same reasoning and with the same relative
        # factor. The weighted gap is an integral of a length against a nodal basis function, so
        # its own scale is the interface diameter times the largest nodal weight, and the
        # rounding of the coordinate differences and the segment quadrature that produce it
        # lands a few hundred times below that product. Anything at or beneath this level is not
        # a small gap, it is the absence of one.
        #
        # The penalty branch needs this where the multiplier branch does not. There the contact
        # pressure is an unknown of the system and a node that touches nothing simply solves to
        # zero; here the pressure is MANUFACTURED from the gap as kappa*g, so at kappa = 1e6 a
        # gap of 5e-16 becomes a pressure of 5e-10 and a bare `p > 0` test reads it as contact.
        # Measured on MortarContactPenaltyHexa20: without the floor, 2 to 9 of 21 nodes are
        # declared active in the load-free increment 0, the set flickers from one augmentation
        # to the next, and the outer loop compares noise against noise until it exhausts its
        # iteration cap. The separation is not marginal - the noise sits at 5e-16 against a
        # threshold of 1e-13, while the first genuinely loaded increment carries 5e-06.
        self.currentGapTolerance = 1e-12 * weight_scale * self._interfaceDiameter
        if weight_scale > 0.0:
            significant = np.flatnonzero(weights < -1e-6 * weight_scale)
            if len(significant):
                worst = np.min(weights) / weight_scale
                self._warnOnce(
                    "negative_nodal_weight",
                    f"{len(significant)} slave node(s) with a negative nodal mortar weight "
                    f"D_II (worst: {worst:.3e} of the largest weight), first in increment "
                    f"{timeStep.number}. The active-set indicator handles the sign, but the "
                    f"weighted gap is no longer a mean opening at those nodes. Usual causes: a "
                    f"partially covered CONQUAD9, or a sliver overlap that triggered the "
                    f"reference-element fallback.",
                )
        # Precompute the sparsity patterns AND the row values once per increment.
        # The geometry is frozen for the increment, so both are constant while
        # applyConstraint runs once per Newton iteration.
        self.currentDNonzero, self.currentDRow = _nonzeroRows(D_full, nSlave)
        self.currentCNonzero, self.currentCRow = _nonzeroRows(C_full, nSlave)

        # The penalty parameter needs the nodal weights, so it is derived here
        # rather than at the top of the assembly like c_n.
        if self.formulation == "penalty" and self._derive_epsN:
            self._resolveEpsN()
        elif self.formulation == "penalty" and self._explicit and self._explicitLumpedMass is not None:
            # A stiffness the user chose is honoured, but it is still measured against the
            # stability limit of the integrator -- silently integrating past that limit produces
            # a run that completes and means nothing. Reported rather than refused, because the
            # Gershgorin bound behind it is conservative and would otherwise block models that
            # are in fact stable.
            omegaSquared = self._omegaSquaredBound(self._explicitLumpedMass)
            dt = getattr(self, "_explicitTimeIncrement", 0.0)
            if omegaSquared > 0.0 and dt > 0.0:
                dtStable = 2.0 / np.sqrt(omegaSquared)
                if dt > dtStable:
                    self._warnOnce(
                        "epsN_above_stability_limit",
                        f"penaltyStiffness = {self.epsN:.4g}"
                        f"{f' with tangentPenalty = {self._tangentPenaltyStiffness():.4g}' if self.friction else ''}"
                        f" needs a time increment of at most "
                        f"{dtStable:.4g} to stay stable, and this analysis runs at {dt:.4g} -- a "
                        f"factor of {dt / dtStable:.2f} above it. Either lower the stiffness by a factor of at "
                        f"least {(dt / dtStable) ** 2:.4g}, lower 'courant-number' by the same "
                        f"factor, or leave 'penaltyStiffness' unset and let it be derived from the "
                        f"stability limit. The bound is a conservative (Gershgorin) estimate, so a "
                        f"small exceedance may still run -- but it will not be trustworthy.",
                    )

        if self.friction and self.slipTolerance > 0.0 and self._elasticSlip is None:
            # h as in _resolveEpsN: the nodal weights sum to the non-mortar surface measure.
            h = characteristicFacetSize(
                float(np.sum(np.abs(self.currentNodalWeights))), len(self.nonMortarFacets), self.model.domainSize
            )
            self._friction.setCharacteristicLength(h)
            if self.journal is not None:
                self.journal.message(
                    f"contact '{self._name}': friction with slipTolerance = {self.slipTolerance:.4g}: elastic "
                    f"slip gamma_crit = {self._elasticSlip:.4g} (h = {h:.4g}), eps_T = mu * p / gamma_crit.",
                    self.identification,
                    level=1,
                )

        if self.friction:
            self._carryFrictionHistory(U_ref, previousNormals)

    def _assembleCouplingMatricesFromSegments(
        self, seg_records, seg_masters, slave_els, current_coords, n_slave, n_master
    ) -> tuple[csr_matrix, csr_matrix]:
        """Turn the segment quadrature of one increment into the two mortar coupling matrices.

        The second of the two passes: the first clips the two surfaces against each other and
        records an integration point per clipped cell, this one builds the dual basis those points
        imply and assembles D and C from it. The dual coefficients are rebuilt per facet from the
        covered area rather than taken from a reference element, and the rows of a partly covered
        facet are weighted back to their whole-facet integral; both steps are explained where they
        happen below.

        Parameters
        ----------
        seg_records
            Per non-mortar facet number, the integration point records ``(N_s, mortar element
            number, N_m, dGamma)`` collected by the segmentation.
        seg_masters
            Per non-mortar facet number, the mortar elements it was found to overlap.
        slave_els
            Per non-mortar facet number, the element and its node indices.
        current_coords
            The deformed coordinates of every node of both surfaces.
        n_slave
            Number of non-mortar nodes.
        n_master
            Number of mortar nodes.

        Returns
        -------
        tuple[csr_matrix, csr_matrix]
            The non-mortar matrix D and the mortar matrix C.
        """
        # D and C are assembled sparsely from the element-local blocks. The sparsity
        # is a property of the dual formulation, not an implementation detail: the
        # point of the dual space of Wohlmuth (2000) is a nodal basis function that
        # satisfies the interface constraint and "at the same time has local
        # support", so that the mortar map "can be represented by a diagonal matrix"
        # instead of a linear system. Popp, Wohlmuth, Gee & Wall (2012) carry this to
        # second order, where the basis transformation splits D into two trivially
        # invertible factors and the biorthogonality reduces the inverse "either to a
        # diagonal matrix (see section 4.2) or to at least a sparse matrix (see
        # section 4.4)" - the quadratic case here. Dense storage would therefore cost
        # O(n_slave^2) for a structurally sparse object.
        D_rows, D_cols, D_vals = [], [], []
        C_rows, C_cols, C_vals = [], [], []

        # ------------------------------------------------------------------
        # PASS 2: Dual coefficients from the actual segment quadrature
        # and assembly
        # ------------------------------------------------------------------
        # Numerator and denominator of the per-node coverage fraction, filled facet by facet below.
        coverageCovered = np.zeros(n_slave)
        coverageFull = np.zeros(n_slave)

        for s_num, records in seg_records.items():
            s_el, s_idx = slave_els[s_num]
            n_s = len(s_idx)

            # Biorthogonality system on the true integration domain:
            # M_t[a,b] = int_seg(N_tilde_a * N_tilde_b), D_t[a] = int_seg(N_tilde_a)
            #
            # Two things decide that these integrals run over the SEGMENTS rather than over a
            # reference element. First, Popp, Gitterle, Gee & Wall (2010) after their Eq. (22):
            # "the biorthogonality condition has to be fulfilled in the physical space and not in
            # the finite element parameter space" -- it makes no difference only where the element
            # Jacobian is constant. Second, and this is the first of the two remedies of Cichosz &
            # Bischoff (2011) for partly covered facets, their Eq. (41): "the underlying integrals
            # now have to be evaluated on the contact area instead of the entire slave element".
            # The extra factor T_e is the basis transformation of Popp et al. (2012), which Cichosz
            # & Bischoff (2D, linear) do not need; composing the two is this implementation's own
            # step, not something either paper states.
            T_e = s_el.getBasisTransformation()
            M_t = np.zeros((n_s, n_s))
            D_t = np.zeros(n_s)
            for N_s, _, _, dG in records:
                N_tilde = T_e @ N_s
                M_t += np.outer(N_tilde, N_tilde) * dG
                D_t += N_tilde * dG

            # For sliver overlaps M_t becomes (near-)singular; fall back to the
            # reference-element coefficients in that case.
            cond_M_t = np.linalg.cond(M_t)
            if cond_M_t < 1e12:
                A_e = np.diag(D_t) @ np.linalg.inv(M_t) @ T_e
            else:
                # Measured over the Control_Tests and patch-test suite: 4 of 3989
                # slave-element evaluations take this path, all of them in the
                # curved Hertz models with CONQUAD8. It is not a dead branch, and
                # it has a consequence worth naming.
                #
                # D_II = int_overlap(N_tilde) would be non-negative for ANY
                # sub-region if N_tilde were pointwise non-negative. It is not,
                # for CONQUAD8 at alpha = 1/3: the transformed corner function
                # dips to -1/324 over exactly 12.5 % of the reference square (the
                # negative region is the triangle (1-xi) + (1-eta) < 1, of area 1/2
                # out of 4), and
                # pointwise non-negativity would need alpha >= 3/8 (CONLINE3 gets
                # it at alpha = 1/3 because its threshold is 1/4). So a partial
                # overlap can produce D_II < 0 on the CONSISTENT path already -
                # see the dual-basis tests in
                # edelweissfe/elements/contactelement/test_element.py.
                #
                # This branch is worse by orders of magnitude: the reference-
                # element dual functions are used instead, and those change sign
                # over half the element, so the integral over a sub-region takes
                # any sign AND any magnitude. Measured: below ~0.1 % coverage
                # D_II flips sign, at -1.0 of the largest weight.
                self._warnOnce(
                    "sliver_fallback",
                    f"degenerate overlap on slave facet {s_num}: cond(M_t) = {cond_M_t:.2e} >= 1e12, "
                    f"falling back to reference-element dual coefficients. Biorthogonality then "
                    f"holds on the full element only, and the nodal weights D_II can take any sign "
                    f"and magnitude there - for CONQUAD8 as well.",
                )
                # Evaluated only HERE, not for every slave facet up front: the
                # reference-element coefficients cost an element mass matrix plus its
                # inversion, and this branch is taken for 4 of 3989 slave-element
                # evaluations over the test suite. These are the element's own
                # reference-element coefficients, for this one element and the
                # same (deformed) coordinates.
                A_e = s_el.computeLocalMassMatrices(np.array([current_coords[nd] for nd in s_el.nodes]))[2]

            # WEIGHTING OF THE BOUNDARY ROWS -- the second remedy of Cichosz & Bischoff (2011),
            # their Eqs. (43) to (45), whose diagnosis is exactly the one below: the nodal weight
            # of a boundary element decreases "quadratically with the contact area" while the
            # contact force decreases "approximately linearly", so "the corresponding nodal value
            # of the Lagrange multiplier is increasing by 1/a".
            #
            # A non-mortar facet that is only partly covered by the mortar surface integrates its
            # own basis functions over the covered part alone, so its nodal weight decreases
            # QUADRATICALLY with the covered area while the force that has to cross there decreases
            # only LINEARLY. The multiplier, being the ratio of the two, grows without bound as the
            # coverage shrinks, and those values are assembled into the stiffness matrix: its
            # condition number degrades with them, and with it the Newton convergence - measured on
            # a facet covered to 8e-06 of its area, a multiplier of 2.1e+05 against a contact
            # pressure of 10 elsewhere on the same interface.
            #
            # The remedy is to weight each row by the reciprocal of its own covered integral, so
            # that the weighted nodal weight is the integral over the WHOLE facet again. Both D and
            # C rows carry the same factor, which leaves the row-sum identity intact and with it the
            # NORMALISED gap g_weak/D_II, since numerator and denominator carry the same factor. The
            # multiplier is rescaled inversely and becomes a pressure of ordinary magnitude again.
            #
            # For a FIXED active set this is a rescaling of the constraint and the displacement
            # solution is identical. Across the active-set decision it is not: the indicator
            # compares p_n against c_n*g_sep, the normalised gap is invariant under the rescaling
            # and p_n is not, so a node near the activation threshold can switch differently.
            # That is a change for the better rather than a side effect to tolerate - before the
            # rescaling, p_n at a thinly covered node was a force divided by a vanishing area and
            # was compared against a genuine pressure; afterwards both sides of the comparison are
            # pressures. Measured on the curved indenter of the archived Hertz series: three nodes
            # at the edge of the contact zone move, by at most 3.4 % of the peak pressure, and the
            # agreement with the analytical solution is unchanged at 1.0 %.
            #
            # No case distinction is needed. On a fully covered facet the covered and the whole
            # integral coincide and the factor is exactly one, which is why an inner facet and a
            # boundary facet can be treated by the same expression. That is Cichosz & Bischoff's
            # own point: with the weighting in place "there is no difference between slave mortar
            # integrals in inner and boundary elements and the switch introduced in Eq. (42) can be
            # skipped".
            #
            # Their weighting factor is defined parametrically, kappa_k = [int_covered N_k dxi]^-1,
            # which for their straight two-node elements returns the row to the whole-element value
            # l_e/2. The ratio of the two PHYSICAL integrals used here lands on the same value and
            # generalises it to a curved facet, whose Jacobian does not cancel out of a parametric
            # expression.
            #
            # NOTE what this costs: D_II is no longer the covered area of a node. It is the area of
            # its facets, covered or not. The coverage itself is kept separately below, because the
            # diagnostics need it and it can no longer be read off the weights.
            D_full = np.diag(s_el.computeLocalMassMatrices(np.array([current_coords[nd] for nd in s_el.nodes]))[1])
            # Only where the covered integral is POSITIVE. The weighting is derived for a covered
            # integral that is a measure, and a quadratic facet covered in the region where its
            # transformed corner function is negative does not provide one - there the factor would
            # flip the sign of the row, which is an equivalent constraint but silently redefines
            # what the sign of the multiplier means, and the active-set indicator reads that sign.
            # Those rows are left alone, so the sign handling they were written for keeps applying
            # to them unchanged.
            covered = D_t > 0.0
            rowScale = np.ones_like(D_t)
            rowScale[covered] = D_full[covered] / D_t[covered]
            A_e = A_e * rowScale[:, None]

            # Coverage bookkeeping, assembled per node over all its facets: how much of the area
            # this node's basis function spans is actually opposed by the other surface. Purely
            # diagnostic - nothing in the formulation reads it - but it is the quantity that tells a
            # partially covered node from a fully covered one now that the weights no longer do.
            for local, node in enumerate(s_el.nodes):
                idx = self.nonMortarNodeToIndex[node]
                coverageCovered[idx] += D_t[local]
                coverageFull[idx] += D_full[local]

            # Assemble the D block and per-master C blocks of this slave element
            D_blk = np.zeros((n_s, n_s))
            C_blks = {}
            for N_s, m_num, N_m, dG in records:
                M_bar = A_e @ N_s
                D_blk += np.outer(M_bar, N_s) * dG
                if m_num not in C_blks:
                    C_blks[m_num] = np.zeros((n_s, len(seg_masters[s_num][m_num].nodes)))
                C_blks[m_num] += np.outer(M_bar, N_m) * dG

            rr, cc = np.meshgrid(s_idx, s_idx, indexing="ij")
            D_rows.append(rr.ravel())
            D_cols.append(cc.ravel())
            D_vals.append(D_blk.ravel())
            for m_num, C_blk in C_blks.items():
                m_idx = np.array([self.mortarNodeToIndex[nd] for nd in seg_masters[s_num][m_num].nodes])
                rr, cc = np.meshgrid(s_idx, m_idx, indexing="ij")
                C_rows.append(rr.ravel())
                C_cols.append(cc.ravel())
                C_vals.append(C_blk.ravel())

        # coo -> csr sums duplicate entries, which is exactly what the dense "+="
        # accumulation did. The summation ORDER differs, so the result agrees with the
        # dense one to round-off rather than bit for bit.
        D = _cooToCsr(D_vals, D_rows, D_cols, (n_slave, n_slave))
        C = _cooToCsr(C_vals, C_rows, C_cols, (n_slave, n_master))

        # Fraction of each node's own facet area that the other surface actually opposes. Zero for a
        # node with nothing opposite it, one for a fully covered one. Diagnostic only.
        self.currentCoverage = np.zeros(n_slave)
        opposed = coverageFull > 0.0
        self.currentCoverage[opposed] = coverageCovered[opposed] / coverageFull[opposed]

        return D, C
