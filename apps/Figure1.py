'''
Author: Pupa
LastEditTime: 2020-11-15 20:25:28


'''
import sys 
sys.path.append("../../")

import pyvista as pv 
import openmesh as om 
import numpy as np 
import pyvistaqt as pvqt
import matplotlib.pyplot as plt 
from pyvista import examples
import igl 

import CPCScalpRecons

arrow_size = 15
wireframe_color = (0.3, 0.3, 0.3)
tangent_plane_color = (0, 1, 1)
clip_plane_color = (1, 1, 0)

## Geometric helper for visulization
def ear_clip_plane(V: np.ndarray, band_id):
    Ar, Al = V[-2:]
    CAr, CAl = V[-2:] + V[band_id]
    direction = np.cross(Al, CAl)
    return pv.Plane(V[band_id]*.55, direction, np.linalg.norm(Al)*3.0, np.linalg.norm(V[band_id])*1.3)


def polyline_from_points(points, radius=0.5):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly.tube(radius=radius)

#################################################################

# def CPC_tangent(V: np.ndarray):
#     assert(len(V) == 9801)
#     V.shape = (99, 99, 3)
#     B = np.zeros((99, 99, 3))
#     for cpc_nz in range(99):
#         for j in range(98):
#             B[cpc_nz, j] = V[cpc_nz, j+1]-V[cpc_nz, j]
#         B[cpc_nz, 98] = B[cpc_nz, 97]
#     B = B.reshape((-1, 3))
#     B /= np.linalg.norm(B, axis=1)[:, np.newaxis]
#     return B 


def draw_tnb_frame(p: pv.Plotter, v, t, n, b, arrow_size=arrow_size):
    p.add_mesh(pv.Arrow(start= v, direction=t*arrow_size, scale="auto"), color="red")
    p.add_mesh(pv.Arrow(start= v, direction=n*arrow_size, scale="auto"), color="green")
    p.add_mesh(pv.Arrow(start= v, direction=b*arrow_size, scale="auto"), color="blue")
    p.add_mesh(pv.Plane(v, n, arrow_size*4, arrow_size*4), color=tangent_plane_color, opacity=0.6)


def draw_cpc_wireframe(p: pv.Plotter, V: np.ndarray, n = 10):
    for i in np.linspace(0, 100, n+1, dtype=np.int)[1:-1]-1:
        p.add_mesh( polyline_from_points(V[i,:]), color=wireframe_color)
    for j in np.linspace(0, 100, n+1, dtype=np.int)[1:-1]-1:
        p.add_mesh( polyline_from_points(V[:,j]), color=wireframe_color)


def draw_orient_definition(p: pv.Plotter, V: np.ndarray, N: np.ndarray):
    pass 

if __name__ == "__main__":
    V, F = igl.read_triangle_mesh("./data/12034_CPC.obj")
    N = igl.per_vertex_normals(V, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    T, N, B = CPCScalpRecons.TNBFrame.TNB_frame(V, N, cpc_ratio=99)
    scalp = pv.PolyData()
    scalp.points, scalp.faces = V, np.column_stack([np.full((len(F), 1), 3), F])

    p = pv.Plotter()
    # p.add_text("Scalp TBN Frame", font_size=18)
    scalp_ = p.add_mesh(scalp, ambient=0.06, diffuse=0.75, opacity=1, color=(1., 0.8, 0.7))#, scalars="yz_dot", cmap="jet")

    draw_cpc_wireframe(p, V[:9801,:].reshape((99,99,-1)))

    cpc_1, cpc_2 = 49, 49

    draw_tnb_frame(p, V[99*cpc_1+cpc_2], T[99*cpc_1+cpc_2], N[99*cpc_1+cpc_2], B[99*cpc_1+cpc_2], 35)
    p.add_mesh(ear_clip_plane(V, 99*cpc_1+cpc_2), color=clip_plane_color, opacity=0.8)

    # draw_coord_frame(p, V[:9801,:], N[:9801,:])
    # 
    # p.add_mesh(scalp_tangent_plane(V[v_id], N[v_id], arrow_size*2), color=tangent_plane_color, opacity=0.6)

    # # p.add_mesh(pv.Arrow(start=V[99*49+50], direction=N[99*49+50]*arrow_size, scale="auto"), color="red")
    # p.add_mesh(pv.Line(pointa=V[99*cpc_1+cpc_2], pointb=V[99*cpc_1+cpc_2]+N[99*cpc_1+cpc_2]*arrow_size), color="red")

    p.show_axes_all()
    p.show()