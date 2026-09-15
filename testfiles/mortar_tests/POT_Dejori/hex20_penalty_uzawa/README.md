# POT Dejori – Ausziehversuch mit Penalty-Kontakt (C3D20/CONQUAD8)

Gegenstück zu `../hex20/`, das den Versuch mit **Lagrange-Multiplikatoren** rechnet.
Alles außer der Kontaktformulierung ist identisch: Netz, Material (GCDP-Beton,
linear-elastischer Stahl), Randbedingungen, elastische Bettung, Laststeuerung,
Solver. Nur so sind die Läufe gegeneinander aussagekräftig.

## Rechnen

```bash
cd POT_Dejori/hex20_penalty
PYTHON_GIL=0 edelweissfe POT_Dejori_penalty_uzawa.inp     # empfohlen
PYTHON_GIL=0 edelweissfe POT_Dejori_penalty_pure.inp      # reines Penalty
python plot_results.py                                    # liest disp.csv/force.csv
```

## Die beiden Eingabedateien

| Datei | Formulierung | Quelle |
|---|---|---|
| `POT_Dejori_penalty_uzawa.inp` | Penalty im augmentierten Lagrange-Verfahren | Puso, Laursen & Solberg (2008), Gl. (7) und (18) |
| `POT_Dejori_penalty_pure.inp` | reines Penalty, $t_A = \kappa g_A$ | Puso & Laursen (2004), Gl. (24) |

**Welche wofür.** Die augmentierte Variante erfüllt die Zwangsbedingung bis zur
Augmentierungstoleranz; ihr Ergebnis hängt praktisch nicht von $\kappa$ ab.
Das reine Penalty lässt eine Durchdringung der Ordnung $1/\kappa$ stehen, und
$\kappa$ bestimmt das Ergebnis mit – dafür braucht es keine äußere Schleife und
es ist der Weg, der später für eine **explizite** Rechnung in Frage kommt.

## Was zu beachten ist

* **Das Netz wird nicht kopiert**, sondern über `*include, input=../hex20/Mesh_POTDejori_edelweissfe.inp`
  eingebunden (18 MB). Wird `../hex20/` verschoben, laufen die Dateien hier ins Leere.
* **$\kappa$ ist nicht gesetzt** und wird als $100\,E/(h\,\bar D)$ abgeleitet; der
  Wert wird beim Start einmal gemeldet. Das ist eine Regel dieser Implementierung,
  **kein Literaturwert** – Puso gibt keine an. Zum Festsetzen: `penaltyStiffness=<wert>`
  im jeweiligen Kontaktblock.
* **`cn` wurde entfernt.** Der Komplementaritätsparameter gehört zur semi-smoothen
  NCP des Sattelpunkts und ist im Penalty-Zweig wirkungslos.
* Die beiden Varianten schreiben getrennte Ausgaben (`ensight_penalty_uzawa` bzw.
  `ensight_penalty_pure`); die reine Variante exportiert nach
  `disp_pure.csv`/`force_pure.csv`, damit `plot_results.py` unverändert auf der
  augmentierten Variante arbeitet.

## Offener Punkt

Penalty und Lagrange laufen dort auseinander, wo die Kontaktfläche nur **teilweise
überdeckt** ist – Penalty erzwingt den *gewichteten* Spalt $g_A$, nicht die
physikalische Öffnung $g_A/D_{II}$. In den Control_Tests betrifft das die
Gleitfälle (1–2 % Abweichung). Beim Ausziehversuch mit wandernder Kontaktzone ist
mit demselben Effekt zu rechnen; welche Formulierung dort richtig liegt, ist nicht
geklärt. Siehe `../../FORMULATION_REPORT.txt` und
`../../Control_Tests/CONTROL_TESTS_MATRIX.txt`.
