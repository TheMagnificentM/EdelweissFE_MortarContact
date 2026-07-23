#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_mortar_mesh.py — Cubit/Abaqus mesh -> EdelweissFE mesh converter

Diese Datei bereitet das Cubit-Mesh (Mesh_POTDejori.inp) für EdelweissFE vor:
1. Sie schneidet den Header ab und startet erst beim Bereich N O D E S.
2. Sie entfernt nicht unterstützte Abaqus-Sektionen (Properties, Materials, Steps, Assemblies).
3. Sie behält alle Knoten (*NODE), Volumenelemente (*ELEMENT), Knotengruppen (*NSET) und Elementgruppen (*ELSET) bei.
4. Sie ersetzt bei Bedarf Elementtypen für bestimmte ELSETs (Konfiguration ganz oben).
5. Sie generiert am Ende der Datei einen neuen Bereich, in dem
   für die Sidesets explizite Kontaktelemente (CONQUAD4 oder CONQUAD8) erzeugt werden.
"""

import os
import re
import sys

# ===========================================================================
# EINSTELLUNGEN (KONFIGURATION)
# ===========================================================================

# Dateinamen (werden unten im Skript relativ zum Ausführungsort verwendet)
INPUT_FILENAME = "Mesh_POTDejori.inp"
OUTPUT_FILENAME = "Mesh_POTDejori_edelweissfe.inp"

# Elementtyp-Ersetzung für bestimmte Elementgruppen (ELSET)
# Hier kann man eintragen, welcher Elementtyp für welche Gruppe genutzt werden soll.
# Beispiel für Hex8: {"CONCRETE": "GC3D8R", "STEEL_SUPPORTS": "C3D8R", "STEEL_ANCHOR": "C3D8R"}
# Beispiel für Hex20: {"CONCRETE": "GC3D20R", "STEEL_SUPPORTS": "C3D20R", "STEEL_ANCHOR": "C3D20R"}
ELSET_TYPE_REPLACEMENT = {
    "CONCRETE": "GC3D20R",
    "STEEL_SUPPORTS": "C3D20R",
    "STEEL_ANCHOR": "C3D20R",
}

# Start-Zeile: Alles VOR dieser Zeile wird komplett verworfen!
START_KEYWORD = "********************************** N O D E S **********************************"

# Sektionen, bei denen das Einlesen des Rests abgebrochen wird
STOP_KEYWORDS = [
    "P R O P E R T I E S", 
    "*SOLID SECTION", 
    "*MATERIAL", 
    "*STEP"
]

# Sektionen, die mittendrin ignoriert/gelöscht werden sollen
UNSUPPORTED_SECTIONS = [
    "*PART", "*END PART",
    "*ASSEMBLY", "*END ASSEMBLY",
    "*INSTANCE", "*END INSTANCE",
]

# Definiere die Kontaktoberflächen (Slave, Master)
CONTACT_SURFACES = [
    ("CONT_SURF_CONC_L", "CONT_SURF_STL_L"),
    ("CONT_SURF_CONC_R", "CONT_SURF_STL_R"),
]

# ===========================================================================
# FACE NODES INDEX MAPPINGS
# ===========================================================================

# C3D8/C3D8R (Linearer Hexaeder) - 4-Knoten Vierecke als Facetten
C3D8_FACE_NODE_IDX = {
    1: [3, 2, 1, 0],   # S1
    2: [4, 5, 6, 7],   # S2
    3: [0, 1, 5, 4],   # S3
    4: [1, 2, 6, 5],   # S4
    5: [2, 3, 7, 6],   # S5
    6: [3, 0, 4, 7],   # S6
}

# C3D20/C3D20R (Quadratischer Hexaeder) - 8-Knoten Vierecke als Facetten
C3D20R_FACE_NODE_IDX = {
    1: [3, 2, 1, 0,  10,  9,  8, 11],   # S1
    2: [4, 5, 6, 7,  12, 13, 14, 15],   # S2
    3: [0, 1, 5, 4,   8, 17, 12, 16],   # S3
    4: [1, 2, 6, 5,   9, 18, 13, 17],   # S4
    5: [2, 3, 7, 6,  10, 19, 14, 18],   # S5
    6: [3, 0, 4, 7,  11, 16, 15, 19],   # S6
}

# CPE4/CPS4 (Linearer Viereck-Flächenelement) - 2-Knoten Segmente als Ränder (2D)
CPE4_FACE_NODE_IDX = {
    1: [0, 1],
    2: [1, 2],
    3: [2, 3],
    4: [3, 0],
}

# CPE8/CPS8 (Quadratischer Viereck-Flächenelement) - 3-Knoten Segmente als Ränder (2D)
CPE8_FACE_NODE_IDX = {
    1: [0, 1, 4],
    2: [1, 2, 5],
    3: [2, 3, 6],
    4: [3, 0, 7],
}

# ===========================================================================
# SKRIPT-LOGIK
# ===========================================================================

def parse_mesh_file(lines):
    """
    Parst die Rohdaten der .inp-Datei.
    Extrahiert Knoten, Elemente, deren Typen, Elsets und Surfaces (Sidesets).
    """
    nodes    = {}
    elements = {}
    el_type  = {}
    elsets   = {}
    surfaces = {}

    mode         = None
    current_name = None
    current_type = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        upper    = stripped.upper()

        if not stripped or stripped.startswith("**"):
            i += 1
            continue

        if stripped.startswith("*"):
            # --- *NODE ---
            if upper.startswith("*NODE"):
                mode = "node"
                i += 1
                continue
            # --- *ELEMENT ---
            elif upper.startswith("*ELEMENT") and "TYPE=" in upper:
                m = re.search(r"TYPE\s*=\s*([^\s,]+)", upper)
                current_type = m.group(1) if m else "UNKNOWN"
                m2 = re.search(r"ELSET\s*=\s*([^\s,]+)", stripped, re.IGNORECASE)
                current_name = m2.group(1) if m2 else None
                if current_name and current_name not in elsets:
                    elsets[current_name.upper()] = []
                mode = "element"
                i += 1
                continue
            # --- *ELSET ---
            elif upper.startswith("*ELSET") or upper.startswith("*NSET"):
                m = re.search(r"(?:NSET|ELSET)\s*=\s*([^\s,]+)", stripped, re.IGNORECASE)
                current_name = m.group(1).upper() if m else None
                if current_name and current_name not in elsets:
                    elsets[current_name] = []
                mode = "elset" if upper.startswith("*ELSET") else "nset"
                i += 1
                continue
            # --- *SURFACE ---
            elif upper.startswith("*SURFACE"):
                m = re.search(r"NAME\s*=\s*([^\s,]+)", stripped, re.IGNORECASE)
                current_name = m.group(1) if m else None
                if current_name:
                    surfaces[current_name.upper()] = []
                mode = "surface"
                i += 1
                continue
            else:
                mode = "other"
                i += 1
                continue

        # Datenzeile einlesen
        if mode == "node":
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 4:
                nid = int(parts[0])
                nodes[nid] = [float(parts[1]), float(parts[2]), float(parts[3])]

        elif mode == "element":
            parts = [p.strip() for p in stripped.split(",")]
            eid  = int(parts[0])
            nids = [int(p) for p in parts[1:] if p]
            # Mehrzeilige Elementdefinitionen auffangen
            while stripped.endswith(","):
                i += 1
                line     = lines[i]
                stripped = line.strip()
                parts2   = [p.strip() for p in stripped.split(",") if p.strip()]
                nids    += [int(p) for p in parts2]
            elements[eid] = nids
            el_type[eid]  = current_type
            if current_name:
                elsets[current_name.upper()].append(eid)

        elif mode == "elset":
            parts = [p.strip() for p in stripped.split(",") if p.strip()]
            if current_name:
                for p in parts:
                    if p.isdigit():
                        elsets[current_name].append(int(p))

        elif mode == "surface":
            # Cubit exportiert Sidesets als: Elementset-Name, Facetten-ID (z. B. S3)
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 2 and current_name:
                elset_name = parts[0].upper()
                face_str   = parts[1].upper()
                m = re.match(r"S(\d+)", face_str)
                if m:
                    face_id = int(m.group(1))
                    if elset_name in elsets:
                        for eid in elsets[elset_name]:
                            surfaces[current_name.upper()].append((eid, face_id))

        i += 1

    return nodes, elements, el_type, elsets, surfaces


def generate_contact_elements(surf_faces, elements, el_type, next_el_id):
    """
    Erzeugt die expliziten Kontaktelemente (CONQUAD4, CONQUAD8, etc.) für eine Surface.
    Erkennt automatisch den Elementtyp des Volumenelements und wählt das passende
    Kontaktelement und die Facetten-Knoten-Mappen.
    """
    con_el_lines = []
    new_el_ids   = []
    detected_type = None

    for (eid, face_id) in surf_faces:
        etype = el_type.get(eid, "").upper()
        
        # Bestimme Facetten-Mappe und Kontaktelement-Typ
        if "20" in etype:
            face_map = C3D20R_FACE_NODE_IDX
            detected_type = "CONQUAD8"
        elif "8" in etype:
            # Achtung: 2D 8-Knoten Elemente haben "8" im Namen, sind aber 2D (z.B. CPE8)
            if "CPE" in etype or "CPS" in etype:
                face_map = CPE8_FACE_NODE_IDX
                detected_type = "CONSEG3"
            else: # Standard Hex8 3D
                face_map = C3D8_FACE_NODE_IDX
                detected_type = "CONQUAD4"
        elif "4" in etype:
            if "CPE" in etype or "CPS" in etype:
                face_map = CPE4_FACE_NODE_IDX
                detected_type = "CONSEG2"
            else:
                # Standard Tet4 oder CPE4/CPS4
                print(f"WARNUNG: Unerwarteter linearer Elementtyp '{etype}' bei Element {eid}.")
                continue
        else:
            # Fallback
            print(f"WARNUNG: Elementtyp '{etype}' bei Element {eid} wird nicht unterstützt.")
            continue

        if face_id not in face_map:
            continue

        parent_nodes = elements[eid]
        idx          = face_map[face_id]
        face_nodes   = [parent_nodes[i] for i in idx]

        node_str = ", ".join(str(n) for n in face_nodes)
        con_el_lines.append(f"  {next_el_id}, {node_str}\n")
        new_el_ids.append(next_el_id)
        next_el_id += 1

    return con_el_lines, new_el_ids, next_el_id, detected_type


def prepare_mesh(input_path, output_path):
    print(f"Bereite Mesh vor: {input_path} -> {output_path}")

    # Pass 1: Rohdaten bereinigen und Elementtypen ersetzen
    raw_lines = []
    started = False

    with open(input_path, "r") as f:
        for line in f:
            stripped = line.strip()
            upper    = stripped.upper()
            
            # Warten, bis das Start-Schlüsselwort gefunden wird
            if not started:
                if START_KEYWORD in line:
                    started = True
                    raw_lines.append(line)
                continue
            
            # Überprüfen, ob wir die Abbruch-Sektion erreichen
            if any(skw in upper for skw in STOP_KEYWORDS):
                break
            
            # Nicht unterstützte Sektionen überspringen
            if any(upper.startswith(kw) for kw in UNSUPPORTED_SECTIONS):
                continue
            
            # Elementtyp-Ersetzung durchführen
            if upper.startswith("*ELEMENT") and "TYPE=" in upper:
                m_elset = re.search(r"ELSET\s*=\s*([^\s,]+)", stripped, re.IGNORECASE)
                if m_elset:
                    elset_name = m_elset.group(1).upper()
                    if elset_name in ELSET_TYPE_REPLACEMENT:
                        new_type = ELSET_TYPE_REPLACEMENT[elset_name]
                        line = re.sub(r"TYPE\s*=\s*[^\s,]+", f"TYPE={new_type}", line, flags=re.IGNORECASE)
                        upper = line.strip().upper()
                        stripped = line.strip()
            
            # Surface-Zeilen anpassen
            if upper.startswith("*SURFACE") and "TYPE=" not in upper:
                line = line.rstrip() + ", TYPE=ELEMENT\n"
            
            raw_lines.append(line)

    # Bereinige eventuelle Leerzeilen am Ende der raw_lines
    while raw_lines and raw_lines[-1].strip() == "**":
        raw_lines.pop()

    # Pass 2: Analysieren des Meshs für die Erzeugung der Kontaktelemente
    nodes, elements, el_type, elsets, surfaces = parse_mesh_file(raw_lines)

    # Pass 3: Kontaktelemente generieren
    next_el_id = max(elements.keys()) + 1 if elements else 1
    contact_blocks = []

    contact_blocks.append("**\n")
    contact_blocks.append("********************************** C O N T A C T   E L E M E N T S **********************************\n")
    contact_blocks.append("**\n")

    for slave_name, master_name in CONTACT_SURFACES:
        for surf_upper in [slave_name, master_name]:
            if surf_upper not in surfaces:
                continue
            surf_faces  = surfaces[surf_upper]
            
            # Vereinfache den Namen: Strippe das "CONT_SURF_" Präfix
            simplified_name = surf_upper.replace("CONT_SURF_", "").lower()
            con_name    = "con_" + simplified_name
            elset_name  = "elset_" + con_name

            con_el_lines, new_el_ids, next_el_id, contact_type = generate_contact_elements(
                surf_faces, elements, el_type, next_el_id
            )

            if new_el_ids and contact_type:
                print(f"  -> {len(new_el_ids)} {contact_type}-Elemente für '{con_name}' generiert.")
                contact_blocks.append("**\n")
                contact_blocks.append(f"*ELEMENT, TYPE={contact_type}, ELSET={elset_name}, provider=edelweiss\n")
                contact_blocks.extend(con_el_lines)
                contact_blocks.append(f"*SURFACE, NAME={con_name}, TYPE=ELEMENT\n")
                contact_blocks.append(f"  {elset_name}, S1\n")

    # Ergebnis wegschreiben (strikt ohne leere Zeilen)
    with open(output_path, "w") as f:
        for line in raw_lines:
            if not line.strip():
                f.write("**\n")
            else:
                f.write(line)
                
        for line in contact_blocks:
            if not line.strip():
                f.write("**\n")
            else:
                f.write(line)

    print(f"Fertig! Bereinigtes Mesh geschrieben nach: {output_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    inp_path    = os.path.join(current_dir, INPUT_FILENAME)
    out_path    = os.path.join(current_dir, OUTPUT_FILENAME)
    prepare_mesh(inp_path, out_path)