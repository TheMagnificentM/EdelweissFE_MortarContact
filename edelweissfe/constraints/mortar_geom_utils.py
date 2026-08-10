import numpy as np

def project_point_to_plane(p, p0, normal):
    """Project a 3D point p onto a plane defined by a point p0 and a normal vector."""
    v = p - p0
    dist = np.dot(v, normal)
    projected = p - dist * normal
    return projected

def get_tangent_basis(normal):
    """Compute two orthogonal tangent vectors spanning the plane perpendicular to normal."""
    if abs(normal[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normal, v)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(normal, t1)
    t2 /= np.linalg.norm(t2)
    return t1, t2

def to_plane_coords(points, p0, t1, t2):
    """Convert 3D points on the plane to local 2D coordinates using the tangent basis."""
    coords = []
    for p in points:
        v = p - p0
        coords.append([np.dot(v, t1), np.dot(v, t2)])
    return np.array(coords)

def to_3d_coords(coords_2d, p0, t1, t2):
    """Convert local 2D coordinates back to 3D points on the plane."""
    points = []
    for c in coords_2d:
        p = p0 + c[0] * t1 + c[1] * t2
        points.append(p)
    return np.array(points)

def sutherland_hodgman_clip(subject_polygon, clip_polygon):
    """Clip subject_polygon with clip_polygon using Sutherland-Hodgman algorithm in 2D.
    Both polygons should be lists of [x, y] coordinates in counter-clockwise order.
    """
    def inside(p, cp1, cp2):
        # Return True if p is on the left side of the directed edge from cp1 to cp2
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (p[0] - cp1[0]) >= -1e-12

    def intersection(cp1, cp2, s, e):
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        den = dc[0] * dp[1] - dc[1] * dp[0]
        if den == 0.0:
            # Parallel edges. Reached only if `inside` classified the two endpoints
            # differently, which its inclusive tolerance normally prevents for
            # exactly collinear points - but "normally" is not "never", and without
            # this guard the result would be inf/NaN and poison the whole cell.
            # Falling back to the crossing endpoint keeps the polygon closed and
            # degenerates the sliver to zero area, which the caller discards.
            return [e[0], e[1]]
        n3 = 1.0 / den
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    output_list = list(subject_polygon)
    
    # Clip against each edge of the clip polygon
    for i in range(len(clip_polygon)):
        cp1 = clip_polygon[i]
        cp2 = clip_polygon[(i + 1) % len(clip_polygon)]
        
        input_list = output_list
        output_list = []
        if not input_list:
            break
            
        s = input_list[-1]
        for e in input_list:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    output_list.append(intersection(cp1, cp2, s, e))
                output_list.append(e)
            elif inside(s, cp1, cp2):
                output_list.append(intersection(cp1, cp2, s, e))
            s = e
            
    return np.array(output_list)

def triangulate_polygon(poly_coords):
    """Triangulate a simple convex 2D polygon using a triangle fan from the first vertex."""
    triangles = []
    if len(poly_coords) < 3:
        return triangles
    for i in range(1, len(poly_coords) - 1):
        triangles.append([poly_coords[0], poly_coords[i], poly_coords[i+1]])
    return np.array(triangles)


def clip_1d_segments(s_coords: np.ndarray, m_coords: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Compute the 1D overlap interval of a 2D master line segment projected onto a 2D slave line segment.
    
    s_coords: shape (2, 2) coords [x0, x1] of the slave line sub-cell
    m_coords: shape (2, 2) coords [y0, y1] of the master line sub-cell
    
    Returns: (s_start, s_end, L_slave, t_vec, n_vec)
             If no overlap exists, returns (0.0, 0.0, L_slave, t_vec, n_vec).
    """
    v_slave = s_coords[1] - s_coords[0]
    L_slave = np.linalg.norm(v_slave)
    if L_slave < 1e-14:
        return 0.0, 0.0, 0.0, np.zeros(2), np.zeros(2)
        
    t_vec = v_slave / L_slave
    n_vec = np.array([t_vec[1], -t_vec[0]])
    
    # Project master endpoints onto the slave line coordinate s in [0, L_slave]
    s0 = np.dot(m_coords[0] - s_coords[0], t_vec)
    s1 = np.dot(m_coords[1] - s_coords[0], t_vec)
    
    s_min = min(s0, s1)
    s_max = max(s0, s1)
    
    s_start = max(0.0, s_min)
    s_end = min(L_slave, s_max)
    
    if s_end - s_start < 1e-12:
        return 0.0, 0.0, L_slave, t_vec, n_vec
        
    return s_start, s_end, L_slave, t_vec, n_vec

