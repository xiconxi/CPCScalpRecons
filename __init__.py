'''
Author: Pupa
LastEditTime: 2020-11-20 18:52:35
'''
import scipy.io as sio
import numpy as np 
import igl
from scipy.spatial import KDTree 
import scipy.spatial.transform as transform
from scipy.spatial.transform import Rotation as R

from . import CPCSampling
from . import TNBFrame

def border_approximate(Nz, Iz, Al, Ar, k = 30):
    borders = np.zeros((2, k, 3))
    _Nz, _Iz = np.r_[0, Nz[1], Nz[2]], np.r_[0, Iz[1], Iz[2]] 
    for i, angle in enumerate(np.linspace(0.2, 0.8, k+2)[1:-1]):
        cosx = np.cos(angle*np.pi)
        sinx = np.power(np.sin(angle*np.pi), 0.85)
        borders[0, i] = cosx*Al + sinx*_Nz
        borders[1, i] = cosx*Al + sinx*_Iz
    return borders.reshape((-1, 3))

def contrl_vertices(V_src, V_dst, unique=True):
    V_src_ = V_src/np.linalg.norm(V_src, axis=1)[:, np.newaxis]
    V_dst_ = V_dst/np.linalg.norm(V_dst, axis=1)[:, np.newaxis]
    dis, idx = KDTree(V_src_).query(V_dst_) 
    
    if unique == False: return idx

    unique_idx, inverse_idx = np.unique(idx, return_inverse=True)
    Vs, Ws = np.zeros((len(unique_idx), 3)), np.zeros(len(unique_idx))
    for i in range(len(V_dst)): 
        Vs[inverse_idx[i]] += V_dst[i]/(0.001+dis[i])
        Ws[inverse_idx[i]] += 1/(0.001+dis[i])
    Vs /= Ws[:, np.newaxis]

    return unique_idx.astype(np.int), Vs

sV3, sF3 = igl.read_triangle_mesh(__path__[0]+"/data/sphere3.obj")
sV5, sF5 = igl.read_triangle_mesh(__path__[0]+"/data/sphere5.obj")

def MinimalSurface(V, Nz, Iz, Al, Ar):
    Nz *= 1.05
    V_borders = border_approximate(Nz, Iz, Al, Ar)
    V = np.row_stack([Nz, Iz, Al, Ar, V_borders, V])
    b, bc =  contrl_vertices(sV3, V)
    b, bc =  contrl_vertices(sV5, bc)
    nilr_idx = contrl_vertices(sV5, np.row_stack([Nz, Iz, Al, Ar]), unique=False)

    V = igl.harmonic_weights(sV5, sF5, b, bc, 3)
    return V, sF5, bc, nilr_idx

def calibrate_nilr(Nz, Iz, Al, Ar):
    R = np.eye(3)
    center = (Al+Ar)/2.0
    R[0] = (Ar-center)/np.linalg.norm(Ar - center)
    R[1] = (Nz-center) - R[0] * np.dot(Nz -center, R[0])
    R[1] /= np.linalg.norm(R[1])
    R[2] = np.cross(R[0], R[1])
    return R, center

def spherical_clip(V, clip_radius, clip_angle=0.20*np.pi):
    '''function to remove ear neighbors and eye front points'''
    V_r = np.linalg.norm(V, axis=1)
    V_x = np.linalg.norm(V[:, 1:], axis=1) > clip_radius
    
    V_y = np.array([np.dot([-np.sin(clip_angle*0.5), np.cos(clip_angle*0.5)], v[1:]) > 0 for v in V/V_r[:, np.newaxis]])
    V_select = V_x * V_y
    return V[V_select]

def GetFrame(Tx: np.array, By: np.array, Nz: np.array, angle_degree):
    r = R.from_rotvec(np.deg2rad(angle_degree%360)*Nz)
    _Tx, _By = r.apply(Tx), r.apply(By)
    return _Tx, _By, Nz

def ScalpReconstruct(V, nilr, cpc_inners=99, fibonacci_samples = 9801 * 0): 
    if fibonacci_samples == 0:   
        R, t = calibrate_nilr(nilr[0], nilr[1], nilr[2], nilr[3])
        nilr = np.dot(nilr-t, R.T)
        
        mriV = np.dot(V-t, R.T)
        if len(V) > 1000:
            mriV = spherical_clip(np.dot(V-t, R.T), np.linalg.norm(nilr[-1])*0.8)
        # np.savetxt("runtime/mriV.obj", np.dot(mriV, R)+T, fmt="v %f %f %f")
        
        mV, mF, mCtrl, nilr_idx = MinimalSurface(mriV, nilr[0], nilr[1], nilr[2], nilr[3]) 
        Nzidx, Izidx, Alidx, Aridx = nilr_idx

        cpc_V, cpc_CPC, cpc_F = CPCSampling.generate_cpcmesh(mV, mF, Nzidx, Izidx, Alidx, Aridx, n=cpc_inners)
        cpc_N = igl.per_vertex_normals(cpc_V, cpc_F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE)
        T, B, N = TNBFrame.TNB_frame(cpc_V, cpc_N, cpc_inners)
        cpc_V, T, B, N = np.dot(cpc_V, R)+t, np.dot(T, R), np.dot(B, R), np.dot(N, R)
        return cpc_V, cpc_F, cpc_CPC, T, B, N

    else:
        _V, _F, _V_CPC, _T, _N, _B = ScalpReconstruct(V, nilr, cpc_inners=699)
        index, fb_CPC, fb_F = CPCSampling.load_fibonacci(fibonacci_samples)
        fb_V, fb_T, fb_N, fb_B = _V[index], _T[index], _N[index], _B[index]
        return fb_V, fb_F, fb_CPC, fb_T, fb_B, fb_N
        
if __name__ == "__main__":
    m = sio.loadmat("data/result.mat")
    cpc_V, cpc_F, cpc_CPC, T, B, N = ScalpReconstruct(m['head_R'], m['ref_R'], 499)


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
    # np.savetxt("runtime/1.obj", cpc_V, fmt="v %f %f %f")
    # np.savetxt("runtime/2.obj", m["head_R"], fmt="v %f %f %f")
