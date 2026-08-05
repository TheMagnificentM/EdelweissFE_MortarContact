import numpy as np

def write_vmesh_to_vtk(filepath, nodes, hex_elements, slave_facets=None, master_facets=None, intersection_cells=None):
    """Write FE meshes and intersection polygons to a VTK file for ParaView visualization."""
    # nodes: list or array of shape (N, 3)
    # hex_elements: list of lists of 8 node indices
    # slave_facets: list of lists of 4 node indices on slave boundary
    # master_facets: list of lists of 4 node indices on master boundary
    # intersection_cells: list of list of 3 coordinates representing sub-triangles
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mortar Contact Mesh Intersection Visualization\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        # Collect all points
        all_pts = list(nodes)
        offset_idx = len(all_pts)
        
        # Add points from intersection cells if any
        cell_points_count = 0
        if intersection_cells is not None:
            for cell in intersection_cells:
                for pt in cell:
                    all_pts.append(pt)
                    cell_points_count += 1
                    
        f.write(f"POINTS {len(all_pts)} float\n")
        for pt in all_pts:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")
            
        # Count total cells
        cells_count = len(hex_elements)
        if slave_facets is not None:
            cells_count += len(slave_facets)
        if master_facets is not None:
            cells_count += len(master_facets)
        if intersection_cells is not None:
            cells_count += len(intersection_cells)
            
        total_cell_data_size = len(hex_elements) * 9
        if slave_facets is not None:
            total_cell_data_size += len(slave_facets) * 5
        if master_facets is not None:
            total_cell_data_size += len(master_facets) * 5
        if intersection_cells is not None:
            total_cell_data_size += len(intersection_cells) * 4
            
        f.write(f"CELLS {cells_count} {total_cell_data_size}\n")
        
        # Write Hex elements (type 12)
        for el in hex_elements:
            f.write(f"8 {el[0]} {el[1]} {el[2]} {el[3]} {el[4]} {el[5]} {el[6]} {el[7]}\n")
            
        # Write Slave boundary facets (type 9)
        if slave_facets is not None:
            for el in slave_facets:
                f.write(f"4 {el[0]} {el[1]} {el[2]} {el[3]}\n")
                
        # Write Master boundary facets (type 9)
        if master_facets is not None:
            for el in master_facets:
                f.write(f"4 {el[0]} {el[1]} {el[2]} {el[3]}\n")
                
        # Write Intersection sub-triangles (type 5)
        if intersection_cells is not None:
            curr_offset = offset_idx
            for i in range(len(intersection_cells)):
                f.write(f"3 {curr_offset} {curr_offset+1} {curr_offset+2}\n")
                curr_offset += 3
                
        f.write(f"CELL_TYPES {cells_count}\n")
        for _ in hex_elements:
            f.write("12\n") # VTK_HEXAHEDRON
        if slave_facets is not None:
            for _ in slave_facets:
                f.write("9\n") # VTK_QUAD
        if master_facets is not None:
            for _ in master_facets:
                f.write("9\n") # VTK_QUAD
        if intersection_cells is not None:
            for _ in intersection_cells:
                f.write("5\n") # VTK_TRIANGLE
                
        # Cell data classification to color them in ParaView
        f.write(f"CELL_DATA {cells_count}\n")
        f.write("SCALARS cell_type_id int 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _ in hex_elements:
            f.write("1\n") # Hex elements = 1
        if slave_facets is not None:
            for _ in slave_facets:
                f.write("2\n") # Slave boundary = 2
        if master_facets is not None:
            for _ in master_facets:
                f.write("3\n") # Master boundary = 3
        if intersection_cells is not None:
            for _ in intersection_cells:
                f.write("4\n") # Intersected cells = 4
