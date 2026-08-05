import os
import numpy as np
from edelweissfe.constraints.mortar_geom_utils import (
    project_point_to_plane,
    get_tangent_basis,
    to_plane_coords,
    to_3d_coords,
    sutherland_hodgman_clip,
    triangulate_polygon
)
from testfiles.mortar_tests.test3_mortar_intersection.vtk_exporter import write_vmesh_to_vtk

def run_large_mesh_intersection_test():
    """
    This script tests the 3D contact interface clipping on a full solid mesh setup.
    It builds two 3D HEX8 solid elements (Slave and Master blocks) overlapping in space,
    defines their contact facets, projects the Master patch onto the Slave auxiliary plane,
    clips the intersecting region, and exports the results to VTK format for ParaView.
    """
    
    # 1. Define nodes for Body 1 (Slave Cube) [0, 1]x[0, 1]x[0, 1]
    # Discretized into 1 HEX8 solid element (nodes 0 to 7)
    nodes = [
        [0.0, 0.0, 0.0], # Node 0
        [1.0, 0.0, 0.0], # Node 1
        [1.0, 1.0, 0.0], # Node 2
        [0.0, 1.0, 0.0], # Node 3
        [0.0, 0.0, 1.0], # Node 4
        [1.0, 0.0, 1.0], # Node 5
        [1.0, 1.0, 1.0], # Node 6
        [0.0, 1.0, 1.0], # Node 7
    ]
    hex_elements = [[0, 1, 2, 3, 4, 5, 6, 7]]
    
    # The Slave contact surface facet lies on the top face of the cube (nodes 4, 5, 6, 7)
    slave_facets = [[4, 5, 6, 7]]
    
    # 2. Define nodes for Body 2 (Master Cube) shifted in X/Y and rotated around Y axis by 30 degrees
    # Base ranges: [0.5, 1.5] x [0.2, 1.2] x [1.0, 2.0] before rotation
    # Center of rotation is Master face center: [1.0, 0.7, 1.0]
    base_master_nodes = np.array([
        [0.5, 0.2, 1.0], # Node 8
        [1.5, 0.2, 1.0], # Node 9
        [1.5, 1.2, 1.0], # Node 10
        [0.5, 1.2, 1.0], # Node 11
        [0.5, 0.2, 2.0], # Node 12
        [1.5, 0.2, 2.0], # Node 13
        [1.5, 1.2, 2.0], # Node 14
        [0.5, 1.2, 2.0], # Node 15
    ])
    
    # Apply rotation of 20 degrees around Y axis (pitch) and 15 degrees around X axis (roll)
    rad_y = np.radians(20)
    rad_x = np.radians(15)
    
    R_y = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ])
    R_total = np.dot(R_x, R_y)
    
    # Rotate around the centroid of the bottom face (approx [1.0, 0.7, 1.0])
    center = np.array([1.0, 0.7, 1.0])
    rotated_master_nodes = []
    for pt in base_master_nodes:
        rotated_pt = center + np.dot(R_total, pt - center)
        rotated_master_nodes.append(rotated_pt.tolist())
        
    master_nodes_offset = len(nodes)
    nodes.extend(rotated_master_nodes)
    hex_elements.append([8, 9, 10, 11, 12, 13, 14, 15])
    
    # The Master contact surface facet lies on the bottom face of the master cube (nodes 8, 9, 10, 11)
    master_facets = [[8, 9, 10, 11]]
    
    # Convert node coordinates list to numpy array
    nodes = np.array(nodes)
    
    # 3. Retrieve physical 3D coordinates for Master and Slave boundary elements
    slave_pts = nodes[slave_facets[0]]
    master_pts = nodes[master_facets[0]]
    
    # 4. Construct auxiliary plane based on Slave facet center and normal
    p0 = np.mean(slave_pts, axis=0)
    # Area-weighted normal vector calculation using diagonal cross products
    v1 = slave_pts[2] - slave_pts[0]
    v2 = slave_pts[3] - slave_pts[1]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    
    # Get local tangent basis (t1, t2) spanning the plane perpendicular to the normal vector
    t1, t2 = get_tangent_basis(normal)
    
    # 5. Project Master facet nodes orthogonally onto the Slave auxiliary plane
    projected_master = np.array([project_point_to_plane(p, p0, normal) for p in master_pts])
    
    # 6. Transform projected Master and Slave coordinates from 3D into 2D local plane system
    slave_2d = to_plane_coords(slave_pts, p0, t1, t2)
    master_2d = to_plane_coords(projected_master, p0, t1, t2)
    
    # 7. Apply Sutherland-Hodgman clipping algorithm in 2D to compute overlap polygon
    clip_result_2d = sutherland_hodgman_clip(master_2d, slave_2d)
    
    # 8. Triangulate clipped overlap polygon into triangular integration cells
    triangles_2d = triangulate_polygon(clip_result_2d)
    
    # 9. Map triangular integration cells coordinates back to 3D space
    intersection_cells_3d = []
    for tri in triangles_2d:
        tri_3d = to_3d_coords(tri, p0, t1, t2)
        intersection_cells_3d.append(tri_3d)
        
    # 10. Write detailed coordinates and results to text report
    output_dir = 'testfiles/mortar_tests/test3_mortar_intersection'
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'mesh_intersection_report.txt')
    with open(report_path, 'w') as f:
        f.write("================ LARGE MESH INTERSECTION REPORT ================\n")
        f.write("This test verifies intersection on a multi-element 3D solid mesh.\n\n")
        f.write("Slave facet quad nodes (top face of Slave Hex):\n")
        for i, pt in enumerate(slave_pts):
            f.write(f"  S{i+1}: {pt.tolist()}\n")
        f.write("\nMaster facet quad nodes (bottom face of Master Hex):\n")
        for i, pt in enumerate(master_pts):
            f.write(f"  M{i+1}: {pt.tolist()}\n")
        f.write("\nProjected Master nodes on Slave Plane:\n")
        for i, pt in enumerate(projected_master):
            f.write(f"  PM{i+1}: {pt.tolist()}\n")
        f.write("\nClipped overlap intersection sub-triangles:\n")
        for idx, tri in enumerate(intersection_cells_3d):
            f.write(f"  Cell {idx+1} vertices:\n")
            for j, v in enumerate(tri):
                f.write(f"    v{j+1}: {v.tolist()}\n")
                
    # 11. Export solid mesh elements, boundary facets, and intersection cells to VTK format
    vtk_path = os.path.join(output_dir, 'mesh_intersection.vtk')
    write_vmesh_to_vtk(vtk_path, nodes, hex_elements, slave_facets, master_facets, intersection_cells_3d)
    
    print(f"Saved mesh intersection report to: {report_path}")
    print(f"Saved VTK visualization mesh to: {vtk_path}")

if __name__ == '__main__':
    run_large_mesh_intersection_test()
