'''
Author: Pupa
LastEditTime: 2020-11-20 15:47:32
'''
import numpy as np 
import scipy.spatial.transform as transform

def clip_line_TNB(V: np.ndarray):
    v_diff = np.diff(V, axis=0)
    T, B = np.zeros(V.shape), np.zeros(V.shape)
    T[1:,:] += v_diff
    T[:-1,:] += v_diff
    T /= np.linalg.norm(T, axis=1)[:, np.newaxis]
    B[:] = np.cross(V[-1]-V[int(len(V)/2)], V[int(len(V)/2)]-V[0])[np.newaxis, :]
    B /= np.linalg.norm(B, axis=1)[:, np.newaxis]
    return T, np.cross(B, T, axis=1), B



def TNB_frame(V: np.ndarray, N: np.ndarray, cpc_ratio: int = 99):
    assert(len(V) == cpc_ratio*(cpc_ratio+2)+2)
    Ar, Al = V[-2:]
    V = V[:-2].reshape((cpc_ratio+2, cpc_ratio, 3))
    lr_T, lr_B, lr_N = np.zeros(V.shape), np.zeros(V.shape), np.zeros(V.shape)
    for cpc_nz in range(cpc_ratio+2):
        t, n, b = clip_line_TNB(np.row_stack([Al, V[cpc_nz,:], Ar]))
        lr_T[cpc_nz], lr_N[cpc_nz], lr_B[cpc_nz] = t[1:-1], n[1:-1], b[1:-1]
    
    lr_T, lr_N, lr_B = lr_T.reshape((-1,3)), lr_N.reshape((-1,3)), lr_B.reshape((-1,3))
    lr_T = np.row_stack([lr_T, np.r_[0, 0, -1], np.r_[0, 0, 1]])
    lr_N = np.row_stack([lr_N, np.r_[1, 0, 0], np.r_[-1, 0, 0]])
    lr_B = np.row_stack([lr_B, np.r_[0, 1, 0], np.r_[0, 1, 0]])

    Rvec = np.cross(lr_N, N, axis=1)
    Rvec /= np.linalg.norm(Rvec, axis=1)[:, np.newaxis]

    for i in range(len(lr_N)): 
        projection_len = np.clip(np.dot(N[i], lr_N[i]), 0, 1)
        R = transform.Rotation.from_rotvec(Rvec[i]*np.arccos(projection_len))
        lr_T[i], lr_N[i], lr_B[i] = map(R.apply, (lr_T[i], lr_N[i], lr_B[i]) )

    return lr_T, lr_N, lr_B
    


if __name__ == "__main__":
    import igl 
    cpc_V, cpc_F = igl.read_triangle_mesh("./apps/data/12034_CPC.obj")
    surface_N = igl.per_vertex_normals(cpc_V, cpc_F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE) 
    T, N, B = TNB_frame(cpc_V, surface_N, cpc_ratio=99)

    import pyvista as pv 
    scalp = pv.PolyData()
    scalp.points = cpc_V
    scalp.faces  = np.column_stack([np.full((len(cpc_F), 1), 3), cpc_F])
    
    p = pv.Plotter()
    p.add_text("Scalp CPC-TNBFrame", font_size=18)
    p.add_mesh(scalp, ambient=0.06, diffuse=0.75, opacity=1.0, color=(1., 0.8, 0.7))#, scalars="yz_dot", cmap="jet")

    arrow_size = 14
    for i in np.linspace(10, 90, 8+1, dtype=int):
        for j in np.linspace(10, 80, 7+1, dtype=int):
            v_id = 99*i+j
            p.add_mesh(pv.Arrow(start=cpc_V[v_id]*1.01, direction=T[v_id]*arrow_size, scale="auto"), color="red")
            p.add_mesh(pv.Arrow(start=cpc_V[v_id]*1.01, direction=N[v_id]*arrow_size, scale="auto"), color="green")
            p.add_mesh(pv.Arrow(start=cpc_V[v_id]*1.01, direction=B[v_id]*arrow_size, scale="auto"), color="blue")
            # p.add_mesh(scalp_tangent_plane(cpc_V[v_id], N[v_id], arrow_size*2), color=(0, 1, 1), opacity=0.6)
    
    p.show(auto_close=False)
    # viewup = [0, 0, 1]
    # path = p.generate_orbital_path(factor=2.0, n_points=100, viewup=viewup, shift=0.2)
    # p.open_gif("apps/runtime/CPCOrientation.gif")
    # p.orbit_on_path(path, write_frames=True, viewup=viewup)
    # p.close()


    
    