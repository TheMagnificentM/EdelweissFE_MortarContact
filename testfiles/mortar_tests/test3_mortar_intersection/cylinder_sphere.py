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

def test_curved_cylinder_sphere_projection():
    # Slave element: represent a small quadrilateral patch on a cylinder surface of radius R = 2.0 along Y axis
    # The normal of the cylinder patch at theta = 0 points in the X direction ([1, 0, 0])
    # Let's generate 4 nodes on the cylinder surface (curved, but approximated by a bilinear/linear element)
    # Theta values: -0.25 to 0.25, Y values: -0.5 to 0.5
    R = 2.0
    slave_nodes_3d = np.array([
        [R * np.cos(-0.25), -0.5, R * np.sin(-0.25)],
        [R * np.cos( 0.25), -0.5, R * np.sin( 0.25)],
        [R * np.cos( 0.25),  0.5, R * np.sin( 0.25)],
        [R * np.cos(-0.25),  0.5, R * np.sin(-0.25)]
    ])
    
    # Master element: represent a patch of a sphere (radius 1.0) coming from the outside (X > 2.0)
    # let's position the sphere center at [3.1, 0.1, 0.0] and generate a patch facing the cylinder.
    # The master nodes on the sphere surface:
    r_sphere = 1.0
    c_sphere = np.array([2.9, 0.1, 0.0])
    # The nodes are on the sphere surface facing -X direction:
    master_nodes_3d = np.array([
        c_sphere + np.array([-r_sphere * np.cos(0.3), -0.3, -r_sphere * np.sin(0.3)]),
        c_sphere + np.array([-r_sphere * np.cos(0.3),  0.3, -r_sphere * np.sin(0.3)]),
        c_sphere + np.array([-r_sphere * np.cos(0.3),  0.3,  r_sphere * np.sin(0.3)]),
        c_sphere + np.array([-r_sphere * np.cos(0.3), -0.3,  r_sphere * np.sin(0.3)])
    ])
    
    # 1. Define the auxiliary plane based on the slave element's centroid and area-weighted normal
    p0 = np.mean(slave_nodes_3d, axis=0)
    v1 = slave_nodes_3d[2] - slave_nodes_3d[0]
    v2 = slave_nodes_3d[3] - slave_nodes_3d[1]
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)
    
    t1, t2 = get_tangent_basis(normal)
    
    # 2. Project both slave and master nodes ORTHOGONALLY onto the slave's auxiliary plane
    # In mortar contact, the projection direction is along the slave node/facet normal vector.
    projected_slave = np.array([project_point_to_plane(p, p0, normal) for p in slave_nodes_3d])
    projected_master = np.array([project_point_to_plane(p, p0, normal) for p in master_nodes_3d])
    
    # 3. Convert to local 2D plane coordinates
    slave_2d = to_plane_coords(projected_slave, p0, t1, t2)
    master_2d = to_plane_coords(projected_master, p0, t1, t2)
    
    # 4. Perform polygon clipping in local 2D plane coordinates
    clip_result_2d = sutherland_hodgman_clip(master_2d, slave_2d)
    
    # 5. Triangulate the clipped polygon
    triangles_2d = triangulate_polygon(clip_result_2d)
    
    # 6. Transform the clipped polygon nodes back to 3D on the auxiliary plane
    clip_result_3d = to_3d_coords(clip_result_2d, p0, t1, t2)
    
    # Plotting to verify
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- Visualize original bodies (Sphere and Cylinder) ---
    # 1. Cylinder: x^2 + z^2 = R^2
    theta_cyl = np.linspace(-0.5, 0.5, 20)
    y_cyl = np.linspace(-1.0, 1.0, 20)
    Theta_cyl, Y_cyl = np.meshgrid(theta_cyl, y_cyl)
    X_cyl = R * np.cos(Theta_cyl)
    Z_cyl = R * np.sin(Theta_cyl)
    ax.plot_wireframe(X_cyl, Y_cyl, Z_cyl, color='blue', alpha=0.15, label='Cylinder Body')

    # 2. Sphere centered at c_sphere
    u_sph = np.linspace(np.pi - 0.5, np.pi + 0.5, 20)
    v_sph = np.linspace(np.pi/2 - 0.5, np.pi/2 + 0.5, 20)
    U_sph, V_sph = np.meshgrid(u_sph, v_sph)
    X_sph = c_sphere[0] + r_sphere * np.sin(V_sph) * np.cos(U_sph)
    Y_sph = c_sphere[1] + r_sphere * np.sin(V_sph) * np.sin(U_sph)
    Z_sph = c_sphere[2] + r_sphere * np.cos(V_sph)
    ax.plot_wireframe(X_sph, Y_sph, Z_sph, color='red', alpha=0.15, label='Sphere Body')
    # --------------------------------------------------------

    # Plot Slave element (Cylinder patch)
    slave_poly = np.vstack([slave_nodes_3d, slave_nodes_3d[0]])
    ax.plot(slave_poly[:, 0], slave_poly[:, 1], slave_poly[:, 2], 'blue', label='Slave Facet (Cylinder Patch)', linewidth=2.5)
    
    # Plot Master element (Sphere patch)
    master_poly = np.vstack([master_nodes_3d, master_nodes_3d[0]])
    ax.plot(master_poly[:, 0], master_poly[:, 1], master_poly[:, 2], 'red', label='Master Facet (Sphere Patch)', linewidth=2.5)
    
    # Plot projected Master element to demonstrate projection onto the auxiliary plane
    proj_master_poly = np.vstack([projected_master, projected_master[0]])
    ax.plot(proj_master_poly[:, 0], proj_master_poly[:, 1], proj_master_poly[:, 2], 'purple', linestyle=':', label='Projected Master (Plane)', alpha=0.8)
    
    # Plot intersection result
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
    ax.set_title('Sphere-to-Cylinder Surface Clipping and Triangulation (With Source Bodies)')
    ax.legend(loc='upper right')
    
    output_dir = 'testfiles/mortar_tests/test3_mortar_intersection'
    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, 'cylinder_sphere_intersection_plot.png')
    plt.savefig(output_image)
    print(f"Saved cylinder-sphere intersection plot to: {output_image}")

if __name__ == '__main__':
    test_curved_cylinder_sphere_projection()
