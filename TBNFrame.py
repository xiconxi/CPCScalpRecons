'''
Author: Pupa
LastEditTime: 2020-11-20 14:23:59
'''
import numpy as np 
import scipy.spatial.transform as transform

def curve_line_TBN(V: np.ndarray):
    v_diff = np.diff(V, axis=0)
    T, B = np.zeros(V.shape), np.zeros(V.shape)
    T[1:,:] += v_diff
    T[:-1,:] += v_diff
    T /= np.linalg.norm(T, axis=1)[:, np.newaxis]
    B[:] = np.cross(V[int(len(V)/2)]-V[0], V[-1]-V[int(len(V)/2)])[np.newaxis, :]
    B /= np.linalg.norm(B, axis=1)[:, np.newaxis]
    return T, B, np.cross(T, B, axis=1)



def TBN_frame(V: np.ndarray, N: np.ndarray, cpc_ratio: int = 99):
    assert(len(V) == cpc_ratio*(cpc_ratio+2)+2)
    Ar, Al = V[-2:]
    V = V[:-2].reshape((cpc_ratio+2, cpc_ratio, 3))
    lr_T, lr_B, lr_N = np.zeros(V.shape), np.zeros(V.shape), np.zeros(V.shape)
    for cpc_nz in range(cpc_ratio+2):
        t, b, n = curve_line_TBN(np.row_stack([Al, V[cpc_nz,:], Ar]))
        lr_T[cpc_nz], lr_B[cpc_nz], lr_N[cpc_nz] = t[1:-1], b[1:-1], n[1:-1]
    
    lr_T, lr_B, lr_N = lr_T.reshape((-1,3)), lr_B.reshape((-1,3)), lr_N.reshape((-1,3))
    lr_T = np.row_stack([lr_T, np.r_[0, 0, -1], np.r_[0, 0, 1]])
    lr_B = np.row_stack([lr_B, np.r_[0, 1, 0], np.r_[0, 1, 0]])
    lr_N = np.row_stack([lr_N, np.r_[1, 0, 0], np.r_[-1, 0, 0]])

    Rvec = np.cross(lr_N, N, axis=1)
    Rvec /= np.linalg.norm(Rvec, axis=1)[:, np.newaxis]

    for i in range(len(lr_N)): 
        projection_len = np.clip(np.dot(N[i], lr_N[i]), 0, 1)
        R = transform.Rotation.from_rotvec(Rvec[i]*np.arccos(projection_len))
        lr_T[i], lr_B[i], lr_N[i] = map(R.apply, (lr_T[i], lr_B[i], lr_N[i]) )

    return lr_T, lr_B, lr_N
    


if __name__ == "__main__":
    import igl 
    cpc_V, cpc_F = igl.read_triangle_mesh("/Users/hotpot/Code/TBA/Scalps/obj/v3/1020/cpc/001-001.obj")
    surface_N = igl.per_vertex_normals(cpc_V, cpc_F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE) 
    T, B, N = TBN_frame(cpc_V, surface_N, cpc_ratio=99)

    import pyvista as pv 
    scalp = pv.PolyData()
    scalp.points = cpc_V
    scalp.faces  = np.column_stack([np.full((len(cpc_F), 1), 3), cpc_F])
    
    p = pv.Plotter()
    p.add_text("Scalp TBN Frame", font_size=18)
    p.add_mesh(scalp, ambient=0.06, diffuse=0.75, opacity=1.0, color=(1., 0.8, 0.7))#, scalars="yz_dot", cmap="jet")

    arrow_size = 15
    for i in (10, 30, 50, 70, 90):
        for j in (20, 35, 50, 65, 80):
            v_id = 99*i+j
            p.add_mesh(pv.Arrow(start=cpc_V[v_id]*1.01, direction=N[v_id]*arrow_size, scale="auto"), color="red")
            p.add_mesh(pv.Arrow(start=cpc_V[v_id]*1.01, direction=B[v_id]*arrow_size, scale="auto"), color="blue")
            p.add_mesh(pv.Arrow(start=cpc_V[v_id]*1.01, direction=np.cross(N[v_id], B[v_id])*arrow_size, scale="auto"), color="green")
            # p.add_mesh(scalp_tangent_plane(cpc_V[v_id], N[v_id], arrow_size*2), color=(0, 1, 1), opacity=0.6)
    
    p.show()
    