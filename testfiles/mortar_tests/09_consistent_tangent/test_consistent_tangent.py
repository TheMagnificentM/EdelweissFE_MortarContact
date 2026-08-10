#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test 9: Konsistenz der Kontakt-Tangente (Finite Differenzen)
============================================================

Dieser Test belegt numerisch, dass die vom Mortar-Constraint assemblierte
Steifigkeit die exakte Jacobi-Matrix des assemblierten Residuums ist -- und zwar
GENAU in dem Umfang, in dem der Code sie beansprucht: bei innerhalb eines
Increments EINGEFRORENER Geometrie.

Vorzeichenkonvention
--------------------
Aus den beiden Zweigen der Normal-NCP in ``applyConstraint`` folgt

    inaktiv:   PExt[lambda] -= lambda          K[lambda, lambda] += 1
    aktiv:     PExt[lambda] -= g_weak          K[lambda, x_s]    -= D_IK * n_I

also durchgaengig

    K = - d PExt / d U .

Diese Konvention wird vom Test mitgeprueft: ein Vorzeichenfehler in der
Assemblierung wuerde als Faktor -1 zwischen K und der Differenzenquotienten-
Matrix auffallen.

Teil 1 -- eingefrorene Geometrie (Pass/Fail)
--------------------------------------------
Normalen ``n_I`` und Koppelmatrizen ``D``, ``C`` werden einmal zu Increment-
Beginn berechnet und fuer alle Newton-Iterationen des Increments festgehalten
(staggered geometry update). Fuer dieses eingefrorene Residuum ist die
assemblierte Tangente exakt. Der Test differenziert das Residuum zentral nach
ALLEN Freiheitsgraden -- Verschiebungen UND Multiplikatoren -- und vergleicht:

    max|K - K_FD| / max|K|  <  TOL_FD .

Zwei Details sind fuer die Aussagekraft entscheidend:

1. Der aktive Zweig wird fuer alle FD-Auswertungen FIXIERT
   (``use_active_set = False``). Die NCP-Funktion enthaelt ein ``max(...)`` und
   ist an der Umschaltgrenze nur semismooth; wuerde eine Stoerung den Zweig
   umschalten, differenzierte man ueber den Knick und der Differenzenquotient
   waere bedeutungslos (nicht falsch assembliert -- nur nicht definiert).
2. Alle Auswertungen verwenden DENSELBEN ``timeStep``-Schluessel, damit die
   Geometrie eingefroren bleibt und wirklich das Residuum differenziert wird,
   dessen Jacobi-Matrix K sein soll.

Teil 2 -- volle Geometrie (nur Messung, kein Pass/Fail)
-------------------------------------------------------
Dieselbe FD-Auswertung, aber mit bei JEDER Stoerung neu berechneten ``n_I``,
``D``, ``C``. Die Differenz zu K ist damit ein direktes Mass fuer die im Code
bewusst weggelassenen Geometrieterme d D/d x, d C/d x und d n/d x. Dieser Teil
hat absichtlich kein Bestehenskriterium: er dokumentiert eine bekannte
Einschraenkung der Formulierung (keine konsistente Linearisierung im Sinne von
Popp et al. 2010 / Farah 2018, Kap. 4), er prueft sie nicht.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from edelweissfe.config.elementlibrary import getElementClass
from edelweissfe.constraints.mortarcontact import Constraint as MortarContact
from edelweissfe.models.femodel import FEModel
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.fieldvariable import FieldVariable

# Relative Toleranz des Differenzenquotienten. Zentrale Differenzen mit H_FD
# erreichen etwa H_FD^2 / eps_mach ~ 1e-10; 1e-8 laesst gut eine Dekade Luft.
TOL_FD = 1e-8
H_FD = 1.0e-6


def build_contact(el_type, slave_pts, master_pts, dim=3, cn=1000.0):
    """Zwei gegenueberliegende Kontaktfacetten als minimales Constraint-Modell."""
    model = FEModel(dimension=dim)
    ConClass = getElementClass(el_type, "edelweiss")

    slave_nodes = [Node(1 + i, np.array(p, dtype=float)) for i, p in enumerate(slave_pts)]
    master_nodes = [Node(100 + i, np.array(p, dtype=float)) for i, p in enumerate(master_pts)]
    for nd in slave_nodes + master_nodes:
        model.nodes[nd.label] = nd

    s_con = ConClass(el_type, 1)
    s_con.setNodes(slave_nodes)
    model.elements[1] = s_con

    m_con = ConClass(el_type, 2)
    m_con.setNodes(master_nodes)
    model.elements[2] = m_con

    model.surfaces = {"slave": {1: [s_con]}, "master": {1: [m_con]}}
    for nd in model.nodes.values():
        nd.fields["displacement"] = FieldVariable(nd, "displacement")

    return MortarContact(
        "c", model, nonMortarSurface="slave", mortarSurface="master", field="displacement", cn=cn
    )


def evaluate(constraint, U, timeStep):
    """PExt und K des Constraints fuer den Zustand U auswerten."""
    n = constraint.nDof
    PExt = np.zeros(n)
    K = np.zeros((n, n))
    constraint.applyConstraint(U, np.zeros(n), PExt, K, timeStep)
    return PExt, K


def make_state(constraint, normal_offsets, lambdas):
    """Zustandsvektor: Slave-Knoten entlang ihrer Normale verschieben, lambda setzen.

    normal_offsets: je Slave-Knoten die Verschiebung entlang seiner undeformierten
    Knotennormale (positiv = in den Master hinein, erzeugt also Durchdringung).
    """
    dim = constraint.model.domainSize
    sf = constraint.sizeField
    U = np.zeros(constraint.nDof)
    for local, nd in enumerate(constraint.non_mortar_nodes):
        idx = constraint.node_to_global_idx[nd]
        n_I = constraint.undeformed_normals[local]
        U[sf * idx : sf * idx + dim] = normal_offsets[local] * n_I
    U[sf * len(constraint.nodes) :] = lambdas
    return U


def fd_jacobian(constraint, U, timeStep_of):
    """Zentrale Differenzen von -PExt nach allen Freiheitsgraden.

    timeStep_of(j, sign) liefert den TimeStep der Einzelauswertung: derselbe
    Schluessel haelt die Geometrie eingefroren, ein neuer erzwingt ihre
    Neuberechnung.
    """
    n = constraint.nDof
    K_fd = np.zeros((n, n))
    for j in range(n):
        Up = U.copy()
        Up[j] += H_FD
        Um = U.copy()
        Um[j] -= H_FD
        P_plus = evaluate(constraint, Up, timeStep_of(j, +1))[0]
        P_minus = evaluate(constraint, Um, timeStep_of(j, -1))[0]
        K_fd[:, j] = -(P_plus - P_minus) / (2.0 * H_FD)
    return K_fd


def run_case(name, el_type, slave_pts, master_pts, normal_offsets, lambdas, dim=3):
    print(f"\n* {name}")
    constraint = build_contact(el_type, slave_pts, master_pts, dim=dim)
    U = make_state(constraint, normal_offsets, lambdas)

    # Increment 1: baut und friert Normalen, D und C; liefert die zu pruefende Tangente.
    frozen_step = TimeStep(1, 0.0, 0.0, 0.0, 0.0, 0.0)
    _, K = evaluate(constraint, U, frozen_step)

    active = constraint.active_set.copy()
    nActive = int(active.sum())
    print(f"  Aktive Knoten: {nActive} von {constraint.nNonMortarNodes}   (Active Set: {active.astype(int)})")
    if nActive == 0:
        print("  [FAIL] Kein aktiver Knoten - der Fall prueft den aktiven Zweig nicht!")
        return False

    # Zweig fuer alle FD-Auswertungen fixieren (siehe Docstring, Punkt 1).
    constraint.use_active_set = False
    constraint.active_set[:] = active

    scale = np.max(np.abs(K))
    if scale <= 0.0:
        print("  [FAIL] Assemblierte Tangente ist identisch null!")
        return False

    # --- Teil 1: eingefrorene Geometrie -> exakte Jacobi-Matrix erwartet ---
    K_fd = fd_jacobian(constraint, U, lambda j, sign: frozen_step)
    err = np.max(np.abs(K - K_fd)) / scale
    print(f"  max|K|                      = {scale:.6g}")
    print(f"  max|K - K_FD| / max|K|      = {err:.3e}   (Toleranz {TOL_FD:.0e})")

    # Ein Vorzeichenfehler wuerde als K ~ -K_FD auffallen; explizit ausweisen.
    err_flipped = np.max(np.abs(K + K_fd)) / scale
    if err_flipped < err:
        print(f"  [FAIL] Vorzeichen: max|K + K_FD|/max|K| = {err_flipped:.3e} < max|K - K_FD|/max|K|")
        return False

    if err > TOL_FD:
        print("  [FAIL] Assemblierte Tangente ist NICHT die Jacobi-Matrix des eingefrorenen Residuums!")
        idx = np.unravel_index(np.argmax(np.abs(K - K_fd)), K.shape)
        print(f"         groesste Abweichung bei K[{idx[0]}, {idx[1]}]: {K[idx]:.6g} vs. {K_fd[idx]:.6g}")
        return False

    # --- Teil 2: volle Geometrie -> Groesse der weggelassenen Terme (nur Messung) ---
    # Ein neuer, je Auswertung eindeutiger TimeStep-Schluessel erzwingt die
    # Neuberechnung von n_I, D und C an der gestoerten Konfiguration.
    counter = [1]

    def fresh_step(j, sign):
        counter[0] += 1
        return TimeStep(counter[0], 0.0, float(j), 0.0, 0.0, 0.0)

    K_fd_full = fd_jacobian(constraint, U, fresh_step)
    err_full = np.max(np.abs(K - K_fd_full)) / scale
    print(f"  max|K - K_FD,voll| / max|K| = {err_full:.3e}   (Messung: weggelassene Terme dD/dx, dC/dx, dn/dx)")

    print(f"  [PASS] {name}")
    return True


# ---------------------------------------------------------------------------
# Geometrien
# ---------------------------------------------------------------------------

QUAD4_SLAVE = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
# Master gegen den Slave versetzt -> nicht-passendes Interface, D und C sind
# voll besetzt und die Zeilensummen-Identitaet wird nicht trivial erfuellt.
QUAD4_MASTER = [[0.15, 0.10, 0.01], [1.15, 0.10, 0.01], [1.15, 1.10, 0.01], [0.15, 1.10, 0.01]]


def rotate_y(points, degrees):
    """Punkte um die y-Achse drehen (kippt das Interface schraeg in den Raum)."""
    a = np.radians(degrees)
    R = np.array([[np.cos(a), 0.0, -np.sin(a)], [0.0, 1.0, 0.0], [np.sin(a), 0.0, np.cos(a)]])
    return [(R @ np.array(p, dtype=float)).tolist() for p in points]


# Gekruemmte quadratische Facetten: die Mittelknoten liegen bewusst NICHT in der
# Ebene der Eckknoten, damit die Newton-Rueckabbildung und die Sub-Cell-Zerlegung
# im Differenzenquotienten mit erfasst werden.
QUAD8_SLAVE = QUAD4_SLAVE + [
    [0.5, 0.0, -0.12],
    [1.0, 0.5, 0.06],
    [0.5, 1.0, -0.12],
    [0.0, 0.5, 0.06],
]
QUAD8_MASTER = QUAD4_MASTER + [
    [0.65, 0.10, -0.05],
    [1.15, 0.60, 0.04],
    [0.65, 1.10, -0.05],
    [0.15, 0.60, 0.04],
]

# 2D: quadratische Linienelemente, Knotenreihenfolge [Ende, Ende, Mitte].
# Die Slave-Normale ist n = (t_y, -t_x) mit t = x_1 - x_0 = +e_x, zeigt also nach
# -y. Der Master liegt deshalb DARUNTER, damit die Flaechen aufeinander zu zeigen
# und ``normal_offsets > 0`` (Verschiebung entlang +n) wie in den 3D-Faellen eine
# Durchdringung erzeugt.
LINE3_SLAVE = [[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]
LINE3_MASTER = [[0.2, -0.01], [1.2, -0.01], [0.7, -0.01]]


def test_consistent_tangent():
    results = []

    results.append(
        run_case(
            "CONQUAD4, flaches nicht-passendes Interface",
            "CONQUAD4",
            QUAD4_SLAVE,
            QUAD4_MASTER,
            normal_offsets=[0.02] * 4,
            lambdas=[-1.0] * 4,
        )
    )

    results.append(
        run_case(
            "CONQUAD4, um 25 Grad geneigtes Interface",
            "CONQUAD4",
            rotate_y(QUAD4_SLAVE, 25.0),
            rotate_y(QUAD4_MASTER, 25.0),
            normal_offsets=[0.02] * 4,
            lambdas=[-1.0] * 4,
        )
    )

    results.append(
        run_case(
            "CONQUAD8, gekruemmt und nicht-passend",
            "CONQUAD8",
            QUAD8_SLAVE,
            QUAD8_MASTER,
            normal_offsets=[0.05] * 8,
            lambdas=[-1.0] * 8,
        )
    )

    results.append(
        run_case(
            "CONLINE3 in 2D, nicht-passend",
            "CONLINE3",
            LINE3_SLAVE,
            LINE3_MASTER,
            normal_offsets=[0.03] * 3,
            lambdas=[-1.0] * 3,
            dim=2,
        )
    )

    # Gemischter Zustand: nur ein Teil der Knoten durchdringt, der Rest bleibt
    # offen. Damit enthaelt EIN K beide Zweige der NCP gleichzeitig - die
    # aktiven Kopplungsbloecke und die Einheitszeilen der inaktiven Knoten.
    results.append(
        run_case(
            "CONQUAD4, gemischt aktiv/inaktiv (beide NCP-Zweige in einem K)",
            "CONQUAD4",
            QUAD4_SLAVE,
            QUAD4_MASTER,
            normal_offsets=[0.02, 0.02, -0.05, -0.05],
            lambdas=[-1.0, -1.0, 0.0, 0.0],
        )
    )

    n_fail = results.count(False)
    assert n_fail == 0, f"{n_fail} von {len(results)} Tangenten-Tests fehlgeschlagen"


if __name__ == "__main__":
    print("====================================================")
    print("MORTAR KONTAKT: KONSISTENTE TANGENTE (FINITE DIFFERENZEN)")
    print("====================================================")

    test_consistent_tangent()

    print("\n====================================================")
    print("ALLE TANGENTEN-TESTS ERFOLGREICH PASSIERT!")
    print("====================================================")
