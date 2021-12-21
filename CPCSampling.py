import numpy as np
import openmesh as om

def gen_inner_cpc_faces(inner_sample=99):
    F = np.zeros((inner_sample-1, inner_sample-1, 2, 3), dtype=np.uint16)
    for i in range(inner_sample - 1):
        for j in range(inner_sample - 1):
            F[i,j,0] = np.array([i * inner_sample + j + 1, i * inner_sample + j, (i + 1) * inner_sample + j + 1], dtype=np.uint16)
            F[i,j,1] = np.array([(i + 1) * inner_sample + j, (i + 1) * inner_sample + j + 1, i * inner_sample + j], dtype=np.uint16)   
    return F.reshape((-1,3))  

def gen_padded_cpc_faces(n = 99):
    n_v = 99*101+2
    F = []
    for i in range(n):
        for j in range(n - 1):
            F.append(np.array([i * n + j + 1, i * n + j, (i + 1) * n + j + 1], dtype=np.uint16))
            F.append(np.array([(i + 1) * n + j, (i + 1) * n + j + 1, i * n + j], dtype=np.uint16))
        F.append(np.array([(i + 1) * n, i * n, n_v-1], dtype=np.uint16))
        F.append(np.array([(i + 1) * n - 1, (i + 2) * n - 1, n_v-2], dtype=np.uint16))

    for j in range(n - 1):
        F.append(np.array([(n+1) * n + j + 1, (n+1) * n + j, j + 1], dtype=np.int32))
        F.append(np.array([j,  j + 1, (n+1) * n + j], dtype=np.int32))
    F.append(np.array([0, (n+1) * n, n_v -1], dtype=np.int32))
    F.append(np.array([(n+2) * n - 1, n - 1, n_v -2], dtype=np.int32))
    F = np.array(F).reshape((-1, 3)).astype(np.int32)
    return F

def gen_inner_cpc_coord(inner_sample=99):
    n = inner_sample
    VCPC = np.zeros((n,n,2), dtype=np.float32)
    cpc_linspace = np.linspace(0, 1, n + 2)[1:-1].reshape((-1,1))
    VCPC[:n, :n, 1], VCPC[:n, :n, 0] = np.meshgrid(cpc_linspace, cpc_linspace)
    return VCPC.reshape((-1,2))

def gen_padded_cpc_coord(n_sample=99):
    n = n_sample
    VCPC = np.zeros((n+2,n,2), dtype=np.float32)
    cpc_linspace = np.linspace(0, 1, n + 2)[1:-1].reshape((-1,1))
    VCPC[:n, :n, 1], VCPC[:n, :n, 0] = np.meshgrid(cpc_linspace, cpc_linspace)
    VCPC[-2, :, 0], VCPC[-1,:,0] = 1, 0
    VCPC[-2, :, 1] = VCPC[-1, :, 1] = cpc_linspace.reshape(-1)
    VCPC = np.vstack([VCPC.reshape((-1,2)), [0.5, 1-1e-3], [0.5, 1e-3]])
    return VCPC.reshape((-1,2))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def convert_to_trimesh(V, F):
    m = om.TriMesh()
    for e in V:
        m.add_vertex(e)
    for e in F:
        m.add_face([m.vertex_handle(e[0]), m.vertex_handle(e[1]), m.vertex_handle(e[2])])
    return m


def convert_to_uv(cpc):
    # (u, v)
    # v = (std::sin(u*3.1415926))*(v-0.5)+0.5;
    # m.set_texcoord2D(vh, TriMesh::TexCoord2D(1-u, v));
    uv = np.zeros(cpc.shape, dtype=np.float)
    for i, row in enumerate(cpc):
        uv[i, 1] = np.sin(row[1]*np.pi)*(row[0]-0.5)+0.5
        uv[i, 0] = 1-row[1]
    return uv 


def fibonacci_resampling(N = 9801):
    inv_golden_ratio = 2/(1+5**0.5)
    index = np.arange(N)+0.5
    P_lr = (np.copy(index) * inv_golden_ratio)%1
    P_nz = np.arccos(1-2*index/N)/np.pi 
    return np.column_stack([P_lr, P_nz])

def load_fibonacci(N = 9801):
    assert(N%9801 == 0)
    import os
    file_path = os.path.dirname(os.path.realpath(__file__))
    CPC = np.loadtxt( file_path + "/" + "data/fb-"+str(N)+".txt")
    index = np.load(file_path + "/" + "data/fb-699-"+str(N)+".npy").astype(np.int32)
    m = om.read_trimesh( file_path + "/" + "data/fb-"+str(N)+".obj")
    SSR_V, SSR_F = m.points(), m.face_vertex_indices()
    return index, CPC, SSR_F


def get_border_line_strip(m, Al_vh, Ar_vh):
    border1 = []
    border2 = []
    for voh in m.voh(Al_vh):
        if m.is_boundary(voh):

            border1.append(m.point(Al_vh))
            border_hh = voh
            while m.to_vertex_handle(border_hh) != Ar_vh:
                border1.append(m.point(m.to_vertex_handle(border_hh)))
                border_hh = m.next_halfedge_handle(border_hh)
            border1.append(m.point(Ar_vh))
            border1 = np.array(border1).reshape((-1, 3))

    for vih in m.vih(Al_vh):
        if m.is_boundary(vih):
            border2.append(m.point(Al_vh))
            border_hh = vih
            while m.to_vertex_handle(border_hh) != Ar_vh:
                border2.append(m.point(m.to_vertex_handle(border_hh)))
                border_hh = m.prev_halfedge_handle(border_hh)
            border2.append(m.point(Ar_vh))
            border2 = np.array(border2).reshape((-1, 3))

    if border1[0][1] > 0:
        return border1, border2
    else:
        return border2, border1


def iterative_Cz_Jurcak(m, Al_vh, Ar_vh, Nz_vh, Iz_vh, Cz=np.r_[0, 200, 200], iterate_cnt = 4):
    '''Cz = midpoint(Nz->Cz->Iz)
       Cz = midpoint(Al->Cz->AR)
    '''
    NCI = intersection_line_strip(m, Nz_vh, Iz_vh, Cz, True)
    # np.savetxt(str(iterate_cnt)+"NCI.obj", NCI[:80], fmt="v %f %f %f")
    _Cz = inner_iso_resample(NCI, 1).reshape(-1)
    LCR = intersection_line_strip(m, Al_vh, Ar_vh, _Cz, True)
    # np.savetxt(str(iterate_cnt)+"LCR.obj", LCR[:80], fmt="v %f %f %f")
    _Cz = inner_iso_resample(LCR, 1).reshape(-1)
    # print("Auto Cz: ", np.linalg.norm(_Cz-Cz))
    if np.linalg.norm(_Cz-Cz) > 0.005 and iterate_cnt: 
        return iterative_Cz_Jurcak(m, Al_vh, Ar_vh, Nz_vh, Iz_vh, _Cz, iterate_cnt-1)
    else:
        return _Cz 

# This function should be only used in boundary mesh
def get_longitude_line_strip(m):
    longitude_n, longitude_d = np.array([1, 0, 0]), 0.0
    for he in m.halfedges():
        _pf, _pt = m.point(m.from_vertex_handle(he)), m.point(m.to_vertex_handle(he))
        _alpha = (longitude_d - np.dot(longitude_n, _pf)) / np.dot(longitude_n, _pt - _pf)
        if 0 <= _alpha <= 1.0:
            line_strip = [_pf + _alpha * (_pt - _pf)]
            moving_hh = he
            while m.is_boundary(moving_hh) == False:
                _pf = m.point(m.from_vertex_handle(moving_hh))
                _pt = m.point(m.to_vertex_handle(moving_hh))
                _pa = m.point(m.to_vertex_handle(m.next_halfedge_handle(moving_hh)))
                alpha_next = (longitude_d - np.dot(longitude_n, _pt)) / np.dot(longitude_n, _pa - _pt)
                alpha_prev = (longitude_d - np.dot(longitude_n, _pa)) / np.dot(longitude_n, _pf - _pa)
                if 0 <= alpha_next <= 1.0:
                    moving_hh = m.opposite_halfedge_handle(m.next_halfedge_handle(moving_hh))
                    line_strip.append(_pt + alpha_next * (_pa - _pt))
                elif 0 <= alpha_prev <= 1.0:
                    moving_hh = m.opposite_halfedge_handle(m.prev_halfedge_handle(moving_hh))
                    line_strip.append(_pa + alpha_prev * (_pf - _pa))

            moving_hh = m.opposite_halfedge_handle(he) 
            while m.is_boundary(moving_hh) == False:
                _pf = m.point(m.from_vertex_handle(moving_hh))
                _pt = m.point(m.to_vertex_handle(moving_hh))
                _pa = m.point(m.to_vertex_handle(m.next_halfedge_handle(moving_hh)))
                alpha_next = (longitude_d - np.dot(longitude_n, _pt)) / np.dot(longitude_n, _pa - _pt)
                alpha_prev = (longitude_d - np.dot(longitude_n, _pa)) / np.dot(longitude_n, _pf - _pa)
                if 0 <= alpha_next <= 1.0:
                    moving_hh = m.opposite_halfedge_handle(m.next_halfedge_handle(moving_hh))
                    line_strip.insert(0, _pt + alpha_next * (_pa - _pt))
                elif 0 <= alpha_prev <= 1.0:
                    moving_hh = m.opposite_halfedge_handle(m.prev_halfedge_handle(moving_hh))
                    line_strip.insert(0, _pa + alpha_prev * (_pf - _pa))            
    return np.array(line_strip, dtype=np.float32) 


def intersection_line_strip(m, f_vh, t_vh, p, normal_inverse=False):
    clip_n = normalize(np.cross( p - m.point(f_vh), p - m.point(t_vh)))
    if normal_inverse == True: clip_n *= -1
    clip_d = np.dot(clip_n, p)
    # print(m.point(f_vh), m.point(t_vh), clip_n, clip_d)
    for vih in m.vih(f_vh):
        moving_hh = m.prev_halfedge_handle(vih)
        _pf = m.point(m.from_vertex_handle(moving_hh))
        _pt = m.point(m.to_vertex_handle(moving_hh))
        alpha = (clip_d-np.dot(clip_n, _pf)) / np.dot(clip_n, _pt - _pf)
        # print("circle >> ", alpha, np.dot(clip_n, _pt - _pf) )
        if (0 <= alpha <= 1.0) and (np.dot(clip_n, _pt - _pf) > 0):
            line_strip = [m.point(f_vh), _pf + alpha * (_pt - _pf)]
            moving_hh = m.opposite_halfedge_handle(moving_hh)
            while m.from_vertex_handle(moving_hh) != t_vh:
                _pf = m.point(m.from_vertex_handle(moving_hh))
                _pt = m.point(m.to_vertex_handle(moving_hh))
                _pa = m.point(m.to_vertex_handle(m.next_halfedge_handle(moving_hh)))
                alpha_next = (clip_d-np.dot(clip_n, _pt)) / np.dot(clip_n, _pa - _pt)
                alpha_prev = (clip_d-np.dot(clip_n, _pa)) / np.dot(clip_n, _pf - _pa)
                if -1e-10 <= alpha_next <= 1.0+1e-10:
                    moving_hh = m.opposite_halfedge_handle(m.next_halfedge_handle(moving_hh))
                    line_strip.append(_pt + alpha_next * (_pa - _pt))
                elif -1e-10 <= alpha_prev <= 1.0+1e-10:
                    moving_hh = m.opposite_halfedge_handle(m.prev_halfedge_handle(moving_hh))
                    line_strip.append(_pa + alpha_prev * (_pf - _pa))
                else:
                    assert True
            _pf = m.point(m.from_vertex_handle(moving_hh))
            _pt = m.point(m.to_vertex_handle(moving_hh))
            alpha = (clip_d - np.dot(clip_n, _pf)) / np.dot(clip_n, _pt - _pf)
            line_strip.append(_pf + alpha * (_pt - _pf))
            return np.array(line_strip, dtype=np.float32)


def inner_iso_resample(line_strip, samples):
    accumulate_len = np.add.accumulate(np.array([np.linalg.norm(i) for i in np.diff(line_strip, axis=0)]))
    linspace_len = np.linspace(0, accumulate_len[-1], samples+2)[1:-1]

    linspace_point = np.zeros((samples, 3))

    sorted_idx = np.searchsorted(accumulate_len, linspace_len)
    accumulate_len = np.concatenate([np.r_[0], accumulate_len])

    for i, _l in enumerate(sorted_idx):
        alpha = (linspace_len[i] - accumulate_len[_l]) / (accumulate_len[_l+1] - accumulate_len[_l])
        linspace_point[i] = line_strip[_l] * (1 - alpha) + line_strip[_l+1] * alpha

    return linspace_point


def generate_cpcmesh(scalp_v, scalp_f, Nzidx, Izidx, Alidx, Aridx, Cz=np.r_[0, 0, 300], n=99):
    m = convert_to_trimesh(scalp_v, scalp_f)
    Ar_vh, Al_vh= m.vertex_handle(Aridx), m.vertex_handle(Alidx)
    Nz_vh, Iz_vh = m.vertex_handle(Nzidx), m.vertex_handle(Izidx)
    Cz = iterative_Cz_Jurcak(m, Al_vh, Ar_vh, Nz_vh, Iz_vh, Cz)

    # latitude: Al(0)->Ar(1)
    # longitude: Nz(0)->Iz(1)
    V = np.zeros((n+2, n, 3)) # total size: 2 + 102*100
    VCPC = np.zeros((n+2, n, 2))
    cpc_latitude1 = intersection_line_strip(m, Al_vh, Ar_vh, m.point(Iz_vh), True)
    cpc_latitude0 = intersection_line_strip(m, Al_vh, Ar_vh, m.point(Nz_vh), True)

    # np.savetxt("cpc0.obj", cpc_latitude0, fmt="v %f %f %f")
    # np.savetxt("cpc1.obj", cpc_latitude1, fmt="v %f %f %f")

    cpc_linspace = np.linspace(0, 1, n + 2)[1:-1].reshape((-1,1))

    # np.savetxt("111.obj",  intersection_line_strip(m, Nz_vh, Iz_vh, Cz, True), fmt="v %f %f %f")
    for i, p in enumerate(inner_iso_resample(intersection_line_strip(m, Nz_vh, Iz_vh, Cz, True), n)):
        V[i] = inner_iso_resample(intersection_line_strip(m, Al_vh, Ar_vh, p, True), n)
    
    VCPC[:n, :n, 1], VCPC[:n, :n, 0] = np.meshgrid(cpc_linspace, cpc_linspace)

    V[-2] = inner_iso_resample(cpc_latitude1, n)
    V[-1] = inner_iso_resample(cpc_latitude0, n)
    V = np.concatenate([np.concatenate(V, axis=0), m.point(Ar_vh).reshape(-1, 3), m.point(Al_vh).reshape(-1,3)], axis=0)
    
    VCPC[-2] = np.concatenate([np.full((n, 1), 1), cpc_linspace], axis=1)
    VCPC[-1] = np.concatenate([np.full((n, 1), 0), cpc_linspace], axis=1)
    VCPC = np.concatenate([np.concatenate(VCPC, axis=0), np.array([[1e-3, 0.5], [1-1e-3, 0.5]])], axis=0)

    F = []
    for i in range(n):
        for j in range(n - 1):
            F.append(np.array([i * n + j + 1, i * n + j, (i + 1) * n + j + 1], dtype=np.uint32))
            F.append(np.array([(i + 1) * n + j, (i + 1) * n + j + 1, i * n + j], dtype=np.uint32))
        F.append(np.array([(i + 1) * n, i * n, V.shape[0]-1], dtype=np.uint16))
        F.append(np.array([(i + 1) * n - 1, (i + 2) * n - 1, V.shape[0]-2], dtype=np.uint32))

    for j in range(n - 1):
        F.append(np.array([(n+1) * n + j + 1, (n+1) * n + j, j + 1], dtype=np.int32))
        F.append(np.array([j,  j + 1, (n+1) * n + j], dtype=np.int32))
    F.append(np.array([0, (n+1) * n, V.shape[0] - 1], dtype=np.int32))
    F.append(np.array([(n+2) * n - 1, n - 1, V.shape[0] - 2], dtype=np.int32))
    F = np.array(F).reshape((-1, 3)).astype(np.int32)


    return V.astype(np.float32), VCPC.astype(np.float32), F




if __name__ == "__main__":
    m = om.read_trimesh("C:/Users/27890/Documents/TPen/runtime/lsmesh.obj")
    Nzidx, Izidx, Alidx, Aridx =  213 ,7684, 12891,  8668

    if True:
        m = om.read_trimesh("E:/TMSGeo/output/v2_optimized/msurface/12008_CPC.obj")
        Nzidx, Izidx, Alidx, Aridx =  212 , 18720, 12891,  8668
    
    SSR_V, SSR_F = m.points(), m.face_vertex_indices()
   

    V, CPC, F = generate_cpcmesh(SSR_V, SSR_F, Nzidx, Izidx, Alidx, Aridx, np.r_[-0.0 ,2.,1003.])
    with open("lsmesh-cpc.obj", "w") as f:
        np.savetxt(f, np.column_stack([V, CPC[:, :]]), fmt="v %.4f %.4f %.4f 0 %f %f")
        np.savetxt(f, F+1, fmt="f %d %d %d")

    # m = convert_to_trimesh(SSR_V, SSR_F)
    # Ar_vh, Al_vh= m.vertex_handle(Aridx), m.vertex_handle(Alidx)
    # Nz_vh, Iz_vh = m.vertex_handle(Nzidx), m.vertex_handle(Izidx)

    # iterative_Cz_Jurcak(m, Al_vh, Ar_vh, Nz_vh, Iz_vh)
    # # lines1 = intersection_line_strip(m, Nz_vh, Iz_vh, np.r_[0, 0, 0])
    # # lines1 = inner_iso_resample(lines1, 1)
    # # with open("runtime/Alr.line.obj", "w") as obj_f:
    # #     np.savetxt(obj_f, lines1, fmt="v %f %f %f")
    # #     print("l ", np.arange(1, len(lines1)+1).__str__()[1:-1], file=obj_f)

    # # iterative_Cz_Jurcak(m, Al_vh, Ar_vh, Nz_vh, Iz_vh)


    # # V, CPC, F = generate_cpcmesh(SSR_V, SSR_F, Alidx, Aridx, Nzidx, Izidx)