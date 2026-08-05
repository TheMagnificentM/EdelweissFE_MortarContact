# Control_Tests – kontrollierte Überprüfung des Mortar-Kontakts

Diese Testreihe prüft den Mortar-Kontakt (`MortarContact3D`) an der einfachsten denkbaren
Situation: zwei aufeinanderliegende Einheitswürfel, die über den Kontakt miteinander verbunden
sind. Der Aufbau ist bewusst so einfach gewählt, dass man die exakte Lösung von Hand kennt und
jeden berechneten Wert dagegen halten kann. Dadurch lässt sich sauber trennen, welcher Anteil
eines Fehlers vom Element kommt und welcher vom Kontakt.

---

## 1. Aufbau

Zwei linear-elastische Einheitswürfel liegen aufeinander. Der untere Block A liegt im Bereich
y ∈ [0, 1], der obere Block B im Bereich y ∈ [1, 2]; zwischen beiden wirkt an der Ebene y = 1
der Mortar-Kontakt.

```
        y=2  ┌───────────┐   ← Oberseite von Block B (hier greift die Last an)
             │  Block B  │      Master,  y ∈ [1,2]
        y=1  ├───────────┤   ← Interface: Mortar-Kontakt (con_slave ↔ con_master)
             │  Block A  │      Slave,   y ∈ [0,1]
        y=0  └───────────┘   ← Unterseite von Block A ist festgehalten (u_y = 0)
```

Das Material ist linear-elastisch mit E = 1000 und ν = 0. Die Querkontraktionszahl ν = 0 ist
Absicht: Sie erzeugt einen rein einachsigen Spannungszustand, für den die exakte Lösung eine
einfache geschlossene Formel ist.

Jeder Test wird für drei Elementtypen gerechnet – `C3D8` (hex8, linear), `C3D20` (hex20,
quadratisch, voll integriert) und `C3D20R` (hex20R, quadratisch, reduziert integriert) – und
jeweils für zwei Netze am Interface: ein zueinander passendes Netz (beide Blöcke 2×2) und ein
nicht passendes Netz (unterer Block 2×2, oberer Block 3×3). Die Kontaktelemente (`CONQUAD4`
bzw. `CONQUAD8`) werden automatisch passend zum Elementtyp erzeugt.

---

## 2. Die Tests und ihre exakten Lösungen

Die ersten drei Tests belasten den Kontakt auf Druck (unterschiedlich aufgebracht); die weiteren
prüfen zusätzliche Kontakt-Eigenschaften. Tests 01–07 laufen auf den zwei Einheitswürfeln (drei
Elementtypen × zwei Netze); Test 08 (Hertz) hat eine eigene, gekrümmte Geometrie.

**01 Eigengewicht** (`01_selfweight`): Die Blöcke werden nur durch ihr eigenes Gewicht belastet
(Volumenkraft b = 10 nach unten). Die Spannung wächst linear: σ_yy(y) = −b·(2 − y), also 0 oben,
−10 am Interface, −20 unten. Verschiebung u_y(y) = (5y² − 20y)/1000.

**02 Druck von oben** (`02_pressure`): Auf die Oberseite wirkt ein konstanter Druck p = 10
(kraftgesteuert). σ_yy = −10 überall, u_y(y) = −0.01·y.

**03 Verschiebung von oben** (`03_dispcontrol`): Die Oberseite wird um u_y = −0.02 nach unten
geschoben (weggesteuert). Gleiche Lösung wie der Drucktest: σ_yy = −10, u_y(y) = −0.01·y.

**04 Abheben unter Zug** (`04_separation`): Die Oberseite wird um +0.02 nach **oben** gezogen. Ein
einseitiger (unilateraler) Kontakt muss sich **trennen**: Kontaktdruck → 0, Spannung → 0, ein
Spalt öffnet sich, der untere Block bleibt in Ruhe. Prüft, ob der Kontakt wirklich ein Kontakt
ist (der unter Zug loslässt) und keine feste Verklebung.

**05 Tangentiales Gleiten** (`05_sliding`): Die Oberseite wird gleichzeitig gedrückt (u_y = −0.02)
und seitlich geschoben (u_x = +0.01). Reibungsfrei muss der obere Block frei gleiten: der
Normaldruck bleibt (σ_yy = −10), es entsteht **kein Schub** (σ_xy = 0), und der untere Block wird
**nicht mitgeschleppt** (u_x = 0 unten). Prüft, dass der Kontakt tangential nicht fälschlich klemmt.

**06 Schiefes Interface** (`06_inclined`): Die gesamte Geometrie wird um 30° um die z-Achse
gedreht, sodass die Kontaktfläche schräg im Raum liegt, und dann axial gedrückt. Das prüft, ob
die **Knoten-Normalen** auf einer nicht achsparallelen Fläche korrekt berechnet werden.

**07 Steifigkeitskontrast** (`07_stiffness`): zwei verformbare Blöcke, aber mit **unterschiedlichem
E** — Block A (unten, Slave) weich (E=1000), Block B (oben) 100× steifer (E=100000),
verschiebungsgesteuert. Prüft den Kontakt zwischen zwei **elastischen Körpern mit
Steifigkeitssprung** (und die Slave/Master-Wahl). Exakte Lösung: Reihenschaltung, σ_yy =
−0.02/(1/E_A+1/E_B) = −19.80198 (konstant), u_y stückweise linear (steiler im weichen Block),
Kontaktdruck = 19.80198.

**08 Hertz'scher Kontakt** (`08_hertz`): ein **gekrümmter (parabolischer) Indenter** auf einer
flachen Foundation, beide elastisch — der klassische Kontakt-Benchmark. Dünne Scheibe mit
z-Fixierung (ebener Verzerrungszustand), Halbmodell mit Symmetrie bei x=0, verschiebungsgesteuert.
Verglichen wird die berechnete **Kontaktdruck-Verteilung** gegen die Hertz-Lösung
p(x)=p₀·√(1−(x/a)²) mit Kontakthalbbreite a=√(4P'R/πE\*) und Maximaldruck p₀=2P'/πa, wobei die Last
P' aus der Reaktion und der effektive Modul 1/E\*=(1−ν²)/E₁+(1−ν²)/E₂ aus beiden Elastizitäten
kommt. Hertz ist eine **Halbraum-Näherung** — nicht maschinengenau, konvergiert aber mit
Netzverfeinerung.

### Kraftgesteuerte Tests (01, 02): minimale Stabilisierung

Bei den kraftgesteuerten Tests wird der obere Block vertikal nur vom Kontakt gehalten. Solange
der Kontaktdruck noch nicht steht, kann er frei „schweben“; das Gleichungssystem ist dann nicht
eindeutig lösbar, und der Löser läuft weg. Das ist keine Eigenschaft des Kontakts, sondern der
Aufgabenstellung. Behoben mit dem kleinstmöglichen Eingriff: einer sehr schwachen y-Feder auf der
Oberseite (`directionalspringpenalty`, `penalty=1e-5`). Ihr Fußabdruck skaliert linear mit der
Steifigkeit und liegt bei ~10⁻⁶ (u_y oben −0.02 bis auf ~5·10⁻⁷ genau), also völlig
vernachlässigbar. Die weggesteuerten Tests (03, 04, 05) und der schiefe Test (06) schreiben die
bewegte Fläche per Dirichlet vor und brauchen keine Feder.

### Die Aktiv-Menge: semismooth NCP und der Parameter `cn`

Ob ein Slave-Knoten Kontakt trägt, entscheidet die nichtglatte Komplementaritätsfunktion (NCP)
der Signorini-Bedingungen (Druck p ≥ 0, Spalt g̃ ≥ 0, p·g̃ = 0):

```
C_n,I = p_I − max(0, p_I − c_n·D̃_II⁻¹·g̃_I) = 0     ⟺     aktiv ⟺ s_n,I = p_I − c_n·D̃_II⁻¹·g̃_I > 0
```

Gelöst wird das mit einer primal-dualen Active-Set-Strategie (= semismooth Newton): die Menge wird
in **jeder** Newton-Iteration neu bestimmt, die äußere Schleife ist konvergiert, sobald sie sich
nicht mehr ändert (Hüeber & Wohlmuth 2005; Gitterle et al. 2010, Gl. 55; Farah 2018, Abschn.
3.5.2). Es gibt bewusst **keinen** festen Iterations-Cutoff.

`c_n` ist **rein algorithmisch**: bei Konvergenz ist g̃ → 0, der Term verschwindet, die konvergierte
Lösung ist also c_n-unabhängig — er beeinflusst nur den Iterationsweg. Zu wählen in der
Größenordnung des E-Moduls des **weicheren** Körpers; hier ist das in allen Tests Block A mit
E = 1000 (auch im Steifigkeitskontrast-Test 07), daher steht in jedem Input `cn=1000.0`. Ein zu
großes c_n macht den Indikator spaltvorzeichen-dominiert und erzeugt Active-Set-Chattering.

### Die Referenzlösungen

Verglichen wird gegen die exakte Handrechnung und – bei den Drucktests (01–03) – zusätzlich gegen
einen **Monolith**: denselben Körper als einen verschmolzenen Block ohne Kontakt
(`reference_monolith_<elem>.inp`). Weil dieser dieselben Elemente und dasselbe Netz verwendet, nur
ohne Kontakt, zeigt der Unterschied genau den Anteil, den allein der Kontakt verursacht.

---

## 3. Was, wo und wie verglichen wird

Aus jedem Lauf kommen die Knotenverschiebungen und die Spannungen/Dehnungen an allen Gaußpunkten.
Daraus wird ausgewertet:

**Verschiebungsfeld (L2-Fehler).** Relativer L2-Fehler über alle Knoten,
√(Σ‖U − U_ref‖²)/√(Σ‖U_ref‖²), gegen die analytische Formel und gegen den Monolith.

**Spannungen an jedem Gaußpunkt.** Für jeden Gaußpunkt wird σ_yy verglichen mit dem analytischen
Wert am Punkt (`err_anal`), am Elementzentrum (`err_anal_ctr`) und mit dem Monolith am selben
Element/Gaußpunkt (`err_mono`, der reine Kontaktanteil). Zusätzlich werden Nebenspannungen und
Schub kontrolliert (bei ν = 0 ≈ 0).

> `err_mono` ist nur dort definiert, wo Zwei-Block- und Monolith-Netz deckungsgleich sind. Beim
> passenden Netz ist das überall; beim nicht passenden Netz nur der untere Block (2×2 wie der
> Monolith), der obere Block (3×3) hat kein deckungsgleiches Gegenstück, daher bleibt `err_mono`
> dort leer. `err_anal` deckt weiterhin alle Elemente ab.

> **hex8** liefert elementweise konstante Spannung (= Wert im Zentrum). Beim Eigengewicht kann das
> den linearen Verlauf an den Elementrändern nicht abbilden, daher ist `err_anal` dort scheinbar
> groß (~2.89), obwohl das Element im Zentrum exakt ist und `err_mono` winzig bleibt. hex20/hex20R
> lösen den Verlauf fein auf (`err_anal` ~5·10⁻⁴).

**Kontaktbedingungen.** An jedem Slave-Knoten sitzt ein Lagrange-Multiplikator = der Kontaktdruck.
Er sollte gleichmäßig 10 sein. „Springen“ hieße: starke Schwankung von Knoten zu Knoten,
Vorzeichenwechsel oder Zappeln. Geprüft wird das Vorzeichen, der Mittelwert und die Schwankung.
Zwei Dinge sind zu unterscheiden: die **Streuung von Knoten zu Knoten** (das „Springen“, überall
~10⁻⁶, also gleichmäßig) und die **Abweichung des Niveaus** von 10 (bei Wegsteuerung exakt 0, bei
Kraftsteuerung ~10⁻⁶ = die Kraft der Stabilisierungsfeder).

**Effektive Steifigkeit.** Aus der Reaktionskraft an der Unterseite und der Verschiebung oben:
k = |F/u_oben| (und E_eff = k·L/A). Damit lässt sich direkt ablesen, ob hex8, hex20 und hex20R
gleich steif sind. Sinnvoll bei den Punktlast-Fällen (Druck/Verschiebung); beim Eigengewicht ist
die Last verteilt, dort gibt es keine sinnvolle Einzel-Steifigkeit.

> **Gaußpunkt-Reihenfolge:** Marmot gibt die Gaußpunkte zeta-major aus (die y-Achse ist der
> langsamste Index; empirisch über das Eigengewicht bestimmt). `evaluate.gp_world_coords`
> rekonstruiert damit die Gaußpunkt-Koordinaten passend zu den Spannungswerten.

---

## 4. Ausführen

Conda-Umgebung `next_v26.11`. Ein einzelner Lauf wie bei jedem EdelweissFE-Beispiel, aus dem
jeweiligen Testordner:

```bash
cd 02_pressure
edelweissfe pressure_hex8_matching.inp        # erzeugt EnSight + CSV der Gausspunkte
python ../evaluate.py pressure_hex8_matching.inp   # L2, Gausspunkt-Vergleich, Kontakt, Steifigkeit
```

Der finale Gesamtreport – ein Aufruf, alle Varianten:

```bash
python final_report.py
```

Er läuft über alle sieben Würfel-Tests (01–07, drei Elementtypen × zwei Netze, plus die
Monolith-Referenzläufe) und zusätzlich den Hertz-Test (08) und schreibt:

- **`final_gp_table.txt`** – die große, lesbare Tabelle: für jeden Gaußpunkt Koordinaten,
  Spannungen, Dehnungen und alle Fehler, gruppiert nach Test und Variante.
- **`final_gp_table.csv`** – dasselbe vollständig (alle 6 Spannungen + 6 Dehnungen + alle Fehler).
- **`final_contact_table.csv`** – der Kontaktdruck an jedem Slave-Knoten je Variante.
- Am Bildschirm: (A) die Druck-Familie (Eigengewicht, Druck, Verschiebung und Steifigkeitskontrast)
  mit L2-Fehler, Spannungsfehler, **Steifigkeit** k und Kontakt-Urteil; (B) die Verhaltens-Tests
  (Abheben, Gleiten, schiefe Fläche); (C) den Hertz'schen Kontakt (Druckverteilung gegen Hertz).

---

## 5. Die Dateien im Überblick

| Datei | Aufgabe |
|-------|---------|
| `_gen_inps.py` | Generator, der alle `.inp`-Dateien konsistent erzeugt (samt Stabilisierung und dem gedrehten Patch-Test). Bei Modelländerungen hier ändern und neu ausführen. |
| `contact_setup.py` | Baut aus den Blöcken die Kontaktflächen, den Elementsatz `solids` und den Knotensatz `allnodes`; `setup_inclined` dreht die Geometrie (Normalen-Test), `setup_hertz` verschiebt den Indenter parabolisch (Hertz-Test). Beide erfassen danach die Element-Referenz neu (`setNodes`). |
| `make_contact_elements.py` | Erzeugt die Kontakt-Overlay-Elemente und richtet ihre Normalen aus. |
| `evaluate.py` | Wertet einen einzelnen Lauf aus (L2, Gausspunkt-Vergleich, Kontakt, Steifigkeit, Verhaltens-Tests) und enthält die analytischen Lösungen. Modul für `final_report.py`. |
| `final_report.py` | Der finale Gesamtreport über alle Varianten. |
| `make_plots.py` | Erzeugt Kontroll-/Präsentations-Bilder in `plots/`: pro Test **und Netz** (matching/nonmatching) je ein Verschiebungs-, Spannungs- und Kontaktdruck-Bild (`<test>_<mesh>_disp/stress/pressure.png`, jeweils alle drei Elementtypen), dazu `sliding_<mesh>_ux.png`, `gausspunkte_schema.png` (Erklärung der Gaußpunkte), `accuracy_overview.png`, `stiffness_bars.png`. So ist jeder Durchlauf per Bild kontrollierbar. |

---

## 6. Ergebnisse

Reproduzierbar mit `python final_report.py`; Rohwerte in `final_gp_table.csv` und
`final_contact_table.csv`.

**Weggesteuert (03) läuft alles maschinengenau** – unabhängig von Elementtyp und Netz (L2 ≈ 10⁻¹¹
bis 10⁻¹⁶). Der sauberste Nachweis, dass der Kontakt korrekt arbeitet.

**Kraftgesteuert (01, 02) läuft mit der minimalen Stabilisierung ebenfalls in allen Fällen durch**
(L2 ≈ 10⁻⁷, reiner Kontaktanteil an der Spannung ≈ 10⁻⁶, beides von der winzigen Feder bestimmt).

**Der Kontaktdruck springt in keinem Fall** – überall gleichmäßig, einheitliches Vorzeichen,
Mittelwert 10, Knoten-zu-Knoten-Streuung ~10⁻⁶.

**Steifigkeit:** hex8, hex20 und hex20R liefern **exakt dieselbe** Steifigkeit (k = 500, E_eff =
1000) unter Druck und Wegsteuerung. Auf sauberer Geometrie ist also kein Elementtyp steifer als
ein anderer.

**Abheben (04):** Der Kontakt **trennt sich** in allen Fällen sauber (Kontaktdruck → 0, Spannung →
0) – er ist also ein echter unilateraler Kontakt und keine Verklebung.

**Gleiten (05):** **reibungsfrei** in allen Fällen – der Normaldruck bleibt (σ_yy = −10), es
entsteht kein Schub (≈ 10⁻¹³), der untere Block wird nicht mitgeschleppt (u_x ≈ 0).

**Schiefes Interface (06):** **maschinengenau** in allen Varianten – der Kontakt reproduziert die
exakte gedrehte Lösung bis ~10⁻⁹, die Axialspannung ist −10 bis auf ~10⁻⁸, und der Kontaktdruck
ist an allen Slave-Knoten gleichmäßig −10. Die Knoten-Normalen auf der um 30° geneigten Fläche
sind also exakt korrekt.

> **Hinweis (behobener Fehler in der Testvorbereitung):** Anfangs war dieser Test *nicht* sauber
> (Fehler ~10⁻³, eine Variante konvergierte nicht). Ursache war **nicht** der Kontakt, sondern die
> Reihenfolge in `setup_inclined`: die Volumenelemente werden von boxGen erzeugt und erfassen dabei
> ihre Referenzgeometrie; wenn man die Knoten *danach* dreht, rechnen sie Steifigkeit und Dehnung
> weiter aus der ungedrehten Referenz (B_ungedreht · u_gedreht) → falsche Lösung. Der Fix:
> nach dem Drehen `el.setNodes(el.nodes)` aufrufen, damit die Elemente die gedrehte Referenz neu
> einlesen. Danach ist der Test maschinengenau. (Dieselbe Ursache erklärte auch eine zuvor
> beobachtete unstimmige Spannungsausgabe im gedrehten Zustand – die war nur eine Folge dieser
> falschen Referenz, kein Element-Fehler; marmot und der reine Python-Provider verhielten sich
> identisch.)

**Steifigkeitskontrast (07):** **maschinengenau** für alle Varianten (L2 ~10⁻¹³…10⁻¹⁶). σ_yy =
−19.80198 (Reihenlösung), Kontaktdruck = 19.80198, und k = 990.1 identisch für hex8/hex20/hex20R —
der Kontakt zwischen zwei unterschiedlich steifen Körpern ist also exakt und elementunabhängig.

**Hertz'scher Kontakt (08):** Die berechnete Druckverteilung reproduziert Hertz — mit einem
lehrreichen Element-Unterschied (siehe `08_hertz/hertz_pressure_profile.png`):
- **hex8 (linear):** *glattes* Druckprofil, aber das Maximum liegt systematisch zu hoch
  (~4–7 %, sinkt mit Verfeinerung). Lineare Elemente bilden die gekrümmte Fläche und den steilen
  Randgradienten schlecht ab.
- **hex20 (quadratisch):** Maximum sehr genau (**~1 %**) und im Mittel näher an Hertz, aber der
  Kontaktdruck **oszilliert von Knoten zu Knoten** (Zickzack ~8–10 %). Das ist der bekannte
  **Ecke/Mittelknoten-Effekt** quadratischer Mortar-Kontaktdrücke (die Eck- und Mittelknoten tragen
  unterschiedlich gewichtete Multiplikatoren), am stärksten am Kontaktrand. Die duale Basis dieses
  Codes dämpft ihn im Inneren (dort glatt), am Rand bleibt er; er sinkt nur langsam mit Verfeinerung.

Bei **beiden** ist die *übertragene Gesamtkraft* korrekt (nur die punktweise Verteilung
unterscheidet sich). Hertz ist zudem eine Halbraum-Näherung und auf einem endlichen, uniformen Netz
nie maschinengenau. Mesh: ~11–22 Elemente über die halbe Kontaktbreite. Druckprofile in
`08_hertz/hertz_profile_*.csv`, Punktwolke in `08_hertz/lambda_hertz_*.vtk`, Bild in
`hertz_pressure_profile.png`.

> **Offener Punkt – `hertz_hex20_medium` konvergiert nicht zuverlässig.** Von den vier
> Hertz-Varianten laufen `hex8_medium`, `hex8_fine` und `hex20_fine` durch; `hex20_medium` ist
> seit Einführung der semismooth NCP nur noch **marginal** konvergent und dabei
> **nichtdeterministisch** (paralleler Solver → andere Rundungsreihenfolge): ein Lauf ging über
> 27 Inkremente durch, ein zweiter brach in Inkrement 7 ab. Mit der früheren heuristischen
> Aktiv-Set-Regel lief er deterministisch in 10 Inkrementen. Es ist **kein** c_n-Problem – ein
> Sweep über c_n = 10 … 10⁶ ändert nichts. Ursache ist der kleinere Konvergenzradius des
> semismooth-Newton-Verfahrens: genau hier (quadratische Elemente, mittleres Netz) fällt die
> Knoten-zu-Knoten-Oszillation des Kontaktdrucks am Kontaktrand mit ~10 % am stärksten aus, und
> die Aktiv-Menge wandert dort von Iteration zu Iteration. Das literaturkonforme Gegenmittel ist
> eine Line-Search-/Damped-Newton-Globalisierung (Deuflhard 2004; De Luca–Facchinei–Kanzow 1996),
> **nicht** ein Rückfall auf die Heuristik. Bis dahin wird dieser eine Fall hingenommen; alle
> übrigen 21 Varianten der Testreihe sind unberührt.

**Zusammengefasst:** Der Mortar-Kontakt arbeitet in allen geprüften Fällen korrekt: er überträgt
Druck mit gleichmäßigem Kontaktdruck, trennt sich unter Zug, gleitet reibungsfrei, hält die
Steifigkeit elementunabhängig (auch bei Steifigkeitskontrast), ist auf einer um 30° geneigten
Fläche maschinengenau und reproduziert den Hertz'schen Kontakt im erwarteten Näherungsrahmen.
Weggesteuert (03), Abheben (04), Gleiten (05) und der schiefe Test (06) sind maschinengenau ganz
ohne Eingriff; bei den kraftgesteuerten Fällen (01, 02) steckt nur die dokumentierte, winzige
Stabilisierungsfeder (~10⁻⁶) drin. Kein Restfehler stammt aus einem Fehler des Kontakts. Einzige
Ausnahme im Konvergenzverhalten (nicht in der Genauigkeit) ist `hertz_hex20_medium` – siehe den
Kasten oben.
