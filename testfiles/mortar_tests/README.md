# Mortar-Kontakt – Testreihe

Verifikation des reibungsfreien Mortar-Normalkontakts (`edelweissfe/constraints/mortarcontact.py`,
`mortar_geom_utils.py`, `elements/contactelement/element.py`). Die Theorie und die Begründung
jedes einzelnen Bausteins stehen in `Doku/mortar_kontakt_frictionless.pdf`; dieses README sagt nur,
was hier liegt und wie man es ausführt.

## Ausführen

Conda-Umgebung `next_v26.11`. Alles auf einmal, aus diesem Verzeichnis:

```bash
pytest .                       # Tests 01–11
```

Einzeln, mit voller Ausgabe (jede Datei ist auch als Skript lauffähig):

```bash
python 08_patch_test_hex20/test_patch_test_hex20.py
pytest 10_signorini_check -s   # -s zeigt die Messwerte statt sie zu schlucken
```

Die Zwei-Block-Referenzreihe hat einen eigenen Report (Laufzeit deutlich länger, rechnet 22
Varianten inklusive Hertz):

```bash
cd Control_Tests && python final_report.py
```

> **Namenskonvention.** Die Einstiegsdateien heißen `test_<name>.py` und ihre Prüfungen liegen in
> `test_*`-Funktionen mit `assert`. Das ist keine Kosmetik: vorher hießen sie
> `<name>.py` mit den Prüfungen unter `if __name__ == "__main__"`, und ein `pytest .` sammelte
> **null** Tests, ohne das zu melden. Wer eine neue Prüfung ergänzt, hält sich bitte daran und
> kontrolliert mit `pytest --collect-only .`, dass sie auch eingesammelt wird.

## Was welcher Test prüft

| Verzeichnis | Ebene | Prüft |
|---|---|---|
| `01_node_normals` | Baustein | Knotennormale gegen die **analytische** Normale eines Zylinderausschnitts (mit Konvergenz), Orientierung, und die exakte 2D-Sehnenformel auf gekrümmten CONLINE3-Kanten |
| `02_polygon_clipping` | Baustein | Sutherland–Hodgman und Fächer-Triangulierung gegen analytische Überlappungen; Flächen**verlust** bei nicht-konvexem Clip-Fenster und Flächen**gewinn** bei nicht-konvexem Subject |
| `03_coupling_matrices` | Baustein | Aufbau von `D`/`C`, Zeilensummen-Identität, Überlappungsfläche, Positivität der Knotengewichte |
| `04_dual_biorthogonality` | Baustein | Biorthogonalität der dualen Basis, Positivität der dualen Gewichte |
| `05_quadratic_segmentation` | Baustein | Sub-Cell-Zerlegung + Segmentquadratur für CONQUAD8/9 und CONTRI6; **Schwellen**, ab denen Sub-Zellen nicht-konvex werden, und was das kostet |
| `06_active_set_pdass` | Baustein | NCP-Indikator, Neubestimmung je Newton-Iteration, Freeze/Anti-Cycling, Vorzeichenbehandlung bei negativem Knotengewicht, BVH-Suche |
| `07_patch_test_2d` | Löser | 2D-Patch-Test CPE4/CONLINE2 und CPE8/CONLINE3, konform und nichtkonform; dazu die Bausteine einzeln und der Eingabeschutz gegen überlappende Flächen |
| `08_patch_test_hex20` | Löser | 3D-Patch-Test, fünf Netz-/Ordnungskombinationen inkl. verzerrtem Interface |
| `09_consistent_tangent` | Baustein | `K` gegen zentrale Differenzen des Residuums, alle Freiheitsgrade, beide NCP-Zweige; misst zusätzlich die weggelassenen Geometrieterme |
| `10_signorini_check` | Löser | Die **konvergierte** Lösung erfüllt `p_n ≥ 0`, `g_sep ≥ 0`, `p_n·g_sep = 0` |
| `11_scale_invariance` | Löser | Dieselbe Rechnung über sechs Zehnerpotenzen der Längenskala, 2D **und** 3D; hält zusätzlich fest, ab welcher Modellausdehnung die absolute Konvergenzschranke des Lösers für die Multiplikatorzeile unerreichbar wird |
| `Control_Tests` | Löser | Zwei-Block-Referenzreihe: Eigengewicht, Druck, Wegsteuerung, Abheben, Gleiten, schiefe Fläche, Steifigkeitskontrast, Hertz. Eigenes README |
| `POT_Dejori` | Anwendung | Ausziehversuch eines Kopfbolzendübels (kein Test, sondern das Zielproblem) |
| `Doku` | — | LaTeX-Quelle und PDF der Dokumentation |

## Laufzeitdiagnosen

Der Constraint meldet fünf Zustände über `warnings.warn`, je Constraint und Ursache einmal:
nicht-konvexe Slave-Sub-Zelle, nicht-konvexe Master-Sub-Zelle, Sliver-Rückfall
(`cond(M_t) ≥ 1e12`), negatives Knotengewicht `D_II`, und ein über die Iterationsobergrenze
eingefrorenes Active Set. Über die Testreihe hinweg sprechen sie nur dort an, wo sie sollen —
auf den Hertz-Modellen mit CONQUAD8 (Sliver + negatives Gewicht) und im Regressionsfall von
`06_active_set_pdass`. Was sie jeweils bedeuten, steht in der Doku, Abschnitt
„Eingabeprüfungen und Laufzeitdiagnosen".
