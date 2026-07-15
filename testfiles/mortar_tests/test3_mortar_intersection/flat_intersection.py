import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from edelweissfe.constraints.utils.intersection import (
    project_point_to_plane,
    get_tangent_basis,
    to_plane_coords,
    to_3d_coords,
    sutherland_hodgman_clip,
    triangulate_polygon
)

def test_flat_intersection():
    # Slave element: Quad centered on a plane z=0, slightly rotated/skewed
    slave_nodes_3d = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0]
    ])
    
    # Master element: Quad shifted in X and Y, and tilted slightly
    master_nodes_3d = np.array([
        [-0.2, -0.2,  0.2],
        [ 1.8,  0.0, -0.1],
        [ 1.5,  1.5,  0.3],
        [ 0.0,  1.8,  0.1]
    ])
    
    # Define auxiliary plane based on slave element center and normal
    p0 = np.mean(slave_nodes_3d, axis=0)
    v1 = slave_nodes_3d[2] - slave_nodes_3d[0]
    v2 = slave_nodes_3d[3] - slave_nodes_3d[1]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    
    t1, t2 = get_tangent_basis(normal)
    
    # Project all nodes onto the plane along the slave's normal vector!
    projected_slave = np.array([project_point_to_plane(p, p0, normal) for p in slave_nodes_3d])
    projected_master = np.array([project_point_to_plane(p, p0, normal) for p in master_nodes_3d])
    
    # Convert to 2D local plane coordinates
    slave_2d = to_plane_coords(projected_slave, p0, t1, t2)
    master_2d = to_plane_coords(projected_master, p0, t1, t2)
    
    # Clip master polygon with slave polygon
    clip_result_2d = sutherland_hodgman_clip(master_2d, slave_2d)
    
    # Triangulate clipped polygon
    triangles_2d = triangulate_polygon(clip_result_2d)
    
    # Map back to 3D points
    clip_result_3d = to_3d_coords(clip_result_2d, p0, t1, t2)
    
    # Plotting to verify
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Slave element
    slave_poly = np.vstack([slave_nodes_3d, slave_nodes_3d[0]])
    ax.plot(slave_poly[:, 0], slave_poly[:, 1], slave_poly[:, 2], 'blue', label='Slave Element', linewidth=2)
    
    # Plot Master element
    master_poly = np.vstack([master_nodes_3d, master_nodes_3d[0]])
    ax.plot(master_poly[:, 0], master_poly[:, 1], master_poly[:, 2], 'red', label='Master Element', linewidth=2)
    
    # Plot projected Master element to show correctness of orthogonal projection
    proj_master_poly = np.vstack([projected_master, projected_master[0]])
    ax.plot(proj_master_poly[:, 0], proj_master_poly[:, 1], proj_master_poly[:, 2], 'red', linestyle=':', label='Projected Master', alpha=0.6)
    
    # Plot clipped intersection polygon
    if len(clip_result_3d) > 0:
        clip_poly = np.vstack([clip_result_3d, clip_result_3d[0]])
        ax.plot(clip_poly[:, 0], clip_poly[:, 1], clip_poly[:, 2], 'green', label='Intersection Polygon', linewidth=3)
        ax.scatter(clip_result_3d[:, 0], clip_result_3d[:, 1], clip_result_3d[:, 2], color='green', s=50)
        
        # Plot triangulation sub-cells
        for tri in triangles_2d:
            tri_3d = to_3d_coords(tri, p0, t1, t2)
            tri_poly = np.vstack([tri_3d, tri_3d[0]])
            ax.plot(tri_poly[:, 0], tri_poly[:, 1], tri_poly[:, 2], 'orange', linestyle='--', alpha=0.8)
            
    # Limit view to focus closely on all elements including master with equal scales
    all_nodes = np.vstack([slave_nodes_3d, master_nodes_3d, projected_master])
    max_range = np.array([all_nodes[:,0].max()-all_nodes[:,0].min(), all_nodes[:,1].max()-all_nodes[:,1].min(), all_nodes[:,2].max()-all_nodes[:,2].min()]).max() / 2.0
    mid_x = (all_nodes[:,0].max()+all_nodes[:,0].min()) * 0.5
    mid_y = (all_nodes[:,1].max()+all_nodes[:,1].min()) * 0.5
    mid_z = (all_nodes[:,2].max()+all_nodes[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mortar Contact Flat Surface Clipping and Triangulation')
    ax.legend()
    
    output_dir = 'testfiles/mortar_tests/test3_mortar_intersection'
    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, 'flat_intersection_plot.png')
    plt.savefig(output_image)
    print(f"Saved flat intersection plot to: {output_image}")

if __name__ == '__main__':
    test_flat_intersection()
