# Mortar-Kontakt – Testreihe

Verifikation des reibungsfreien Mortar-Normalkontakts (`edelweissfe/constraints/mortarcontact.py`,
`mortar_geom_utils.py`, `elements/contactelement/element.py`). Die Theorie und die Begründung
jedes einzelnen Bausteins stehen in `Doku/mortar_kontakt_doku.pdf`; dieses README sagt nur,
was hier liegt und wie man es ausführt.

## Ausführen

Conda-Umgebung `next_v26.11`. Alles auf einmal, aus diesem Verzeichnis:

```bash
pytest .                       # Tests 01–12
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
| `01_node_normals` | Baustein | Knotennormale gegen die **analytische** Normale eines Zylinderausschnitts, mit geforderter Mindest-Konvergenz**ordnung** (≥1,8 linear, ≥3,5 quadratisch); Orientierung; und `N_a(ξ_b) = δ_ab` für alle sieben Elementtypen, also die Übereinstimmung der Knoten-Naturkoordinaten mit der Knotenreihenfolge der Formfunktionen |
| `02_polygon_clipping` | Baustein | Sutherland–Hodgman und Fächer-Triangulierung gegen analytische Überlappungen; Flächen**verlust** bei nicht-konvexem Clip-Fenster und Flächen**gewinn** bei nicht-konvexem Subject |
| `03_coupling_matrices` | Baustein | Aufbau von `D`/`C`, Zeilensummen-Identität, Überlappungsfläche, Positivität der Knotengewichte |
| `04_dual_biorthogonality` | Baustein | Biorthogonalität gegen **unabhängige** Referenzen: geschlossene duale Basis des CONQUAD4, Nachrechnung mit höhergradiger Quadratur, und Diagonalität von `D @ T_e.T` aus der echten Segmentquadratur (Produktionspfad, auch bei Teilüberdeckung); Positivität der dualen Gewichte |
| `05_quadratic_segmentation` | Baustein | Sub-Cell-Zerlegung + Segmentquadratur für CONQUAD8/9 und CONTRI6; **Schwellen**, ab denen Sub-Zellen nicht-konvex werden, und was das kostet; **Übergang** zum Sliver-Rückfallpfad (`cond(M_t) ≥ 1e12`) samt Vorzeichenwechsel von `D_II`; und die Grenze der Basistransformation — bei α = 1/3 ist die CONQUAD8-Eckfunktion nicht punktweise nicht-negativ (dafür bräuchte es α ≥ 3/8), also kippt `D_II` schon auf dem **konsistenten** Pfad, sobald die Überlappung im Negativgebiet liegt (gegen die geschlossene Form gerechnet) |
| `06_active_set_pdass` | Baustein | NCP-Indikator, Neubestimmung je Newton-Iteration, Freeze/Anti-Cycling, Vorzeichenbehandlung bei negativem Knotengewicht, BVH-Suche |
| `07_patch_test_2d` | Löser | 2D-Patch-Test CPE4/CONLINE2 und CPE8/CONLINE3, konform und nichtkonform; dazu die Bausteine einzeln und der Eingabeschutz gegen überlappende Flächen |
| `08_patch_test_hex20` | Löser | 3D-Patch-Test, fünf Netz-/Ordnungskombinationen inkl. verzerrtem Interface |
| `09_consistent_tangent` | Baustein | `K` gegen zentrale Differenzen des Residuums, alle Freiheitsgrade, beide NCP-Zweige; misst zusätzlich die weggelassenen Geometrieterme |
| `10_signorini_check` | Löser | Die **konvergierte** Lösung erfüllt `p_n ≥ 0`, `g_sep ≥ 0`, `p_n·g_sep = 0` — in 2D **und** 3D; zusätzlich der *physikalische* Spalt an **neu gerechneter** Geometrie, was der eingefrorenen Auswertung prinzipiell entgeht |
| `11_scale_invariance` | Löser | Dieselbe Rechnung über sechs Zehnerpotenzen der Längenskala, 2D **und** 3D; hält zusätzlich fest, ab welcher Modellausdehnung die absolute Konvergenzschranke des Lösers für die Multiplikatorzeile unerreichbar wird |
| `12_increment_size` | Löser | Der Fehler der eingefrorenen Geometrie über die Schrittweite: gekrümmter Indenter mit N = 1…16 Increments gegen einen feinen Referenzlauf, mit Schätzung der Konvergenzordnung |
| `Control_Tests` | Löser | Zwei-Block-Referenzreihe: Eigengewicht, Druck, Wegsteuerung, Abheben, Gleiten, schiefe Fläche, Steifigkeitskontrast, Hertz. Eigenes README |
| `POT_Dejori` | Anwendung | Ausziehversuch eines Kopfbolzendübels (kein Test, sondern das Zielproblem) |
| `Doku` | — | LaTeX-Quelle und PDF der Dokumentation |

## Laufzeitdiagnosen

Der Constraint meldet zehn Zustände über `warnings.warn`, je Constraint und Ursache einmal:
nicht-konvexe Slave-Sub-Zelle, nicht-konvexe Master-Sub-Zelle, Sliver-Rückfall
(`cond(M_t) ≥ 1e12`), **nicht konvergierte Gauß-Punkt-Rückabbildung**, negatives Knotengewicht
`D_II`, verschwindende Knotennormale, ein über die Iterationsobergrenze eingefrorenes Active Set,
ein Active Set, das sich am konvergierten Zustand nicht reproduziert, sowie die beiden
Orientierungsprüfungen (uneinheitlicher Umlaufsinn einer Fläche; Slave-Normalen zeigen von der
Masterfläche weg).

Dazu zwei **informative** Meldungen, die keine Voraussetzung verletzen: der aus den angrenzenden
Materialien abgeleitete Wert von `c_n`, bzw. der Hinweis, dass er sich nicht ableiten ließ.

Über die Testreihe hinweg sprechen vier davon an, jeder dort, wo er soll: Sliver-Rückfall und
negatives Knotengewicht gemeinsam auf den beiden hex20-Hertz-Modellen **und** im Lastfall
`teilkontakt` von `10_signorini_check` (CONLINE2/3 am Rand der Überdeckung); negatives
Knotengewicht und Active-Set-Selbstprüfung in den beiden Regressionsfällen von
`06_active_set_pdass`; nicht-konvexe Slave-Sub-Zelle und Sliver-Rückfall in
`05_quadratic_segmentation`, das genau diese Schwellen vermisst. Die übrigen sechs treten nicht
auf. Auf dem Ausziehversuch (`POT_Dejori`) meldet sich in der hex8-Variante (CONQUAD4) als
einzige Diagnose die Active-Set-Selbstprüfung, auf allen sechs Kontaktpaaren an je einem bis zwei
Knoten; die hex20-Variante (CONQUAD8) meldet zusätzlich **negative Knotengewichte** auf einem der
sechs Paare — ohne Sliver-Rückfall, also der oben genannte Fall der nur integral gesicherten
Basistransformation.
Was sie jeweils bedeuten, steht in der Doku, Abschnitt „Eingabeprüfungen und Laufzeitdiagnosen".

## Zwei Fallstricke außerhalb des Kontakts

- **`*updateConfiguration` wirkte prozessweit — behoben.** `loadConfiguration` reichte die Dicts aus
  `config/phenomena.py` per Referenz durch, und `updateConfiguration` schrieb hinein; eine in einer
  Eingabedatei gelockerte Toleranz galt danach für jeden weiteren Job im selben Python-Prozess. Ein
  Verifikationslauf nach einem Produktionsmodell wurde also mit dessen gelockerten Schranken
  bewertet, ohne dass das irgendwo sichtbar war. `loadConfiguration` kopiert die Dicts jetzt
  (`configurator.py`). Die manuelle Wiederherstellung in `11_scale_invariance` ist damit redundant,
  aber harmlos, und bleibt als zusätzliche Absicherung stehen.
- **`planeRectQuad` versetzt die Knotenlabels nicht — offen.** `boxGen` erhöht `currentNodeLabel` um
  `max(model.nodes)`, wenn das Modell schon Knoten hat (`boxgen.py:145–147`); `planeRectQuad` tut
  das nicht und beginnt immer bei 1. Zwei Aufrufe im selben Modell überschreiben sich damit still.
  In der Testreihe wird das umgangen, im Kern ist es nicht behoben.

## Wenn du einen neuen Test schreibst

**Vorzeichen des Multiplikators.** `lambda` folgt der Konvention der Kontaktliteratur (Popp et al.
2009/2012, Gitterle et al. 2010, Farah 2018): **bei Druck positiv**. An einem voll überdeckten
Knoten (`D_II > 0`) ist `lambda` damit unmittelbar der Kontaktdruck — ein Patch-Test mit
aufgebrachtem Druck `p` liefert `lambda = +p`. Die allgemeine Form, die auch negative Knotengewichte
trägt, ist `p_n = lambda * sgn(D_II)`; die Knotenkraft entlang `n_I` ist `-lambda * D_II`. Prüfe
deshalb den **vorzeichenrichtigen** Wert und nicht `abs(lambda)` — sonst geht ein Vorzeichenfehler
in der Assemblierung unbemerkt durch. Herleitung: Doku, Abschnitt „Vorzeichenkonvention des
Multiplikators".

`compute_mortar_coupling_matrices` liefert `D` und `C` als **scipy-sparse** (CSR), ebenso
`current_D` / `current_C`. Wer sie dicht braucht, ruft `.toarray()` an der Verbrauchsstelle — so
machen es die bestehenden Tests. `current_D_rowsum` ist unverändert ein `ndarray`.
