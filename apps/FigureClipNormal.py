'''
Author: Pupa
LastEditTime: 2020-11-15 20:25:28


'''


import pyvista as pv 
import numpy as np 
import igl 


wireframe_color = (0.3, 0.3, 0.3)

def cpc_clip_normal(V: np.ndarray, Al, Ar):
    assert(V.shape == (9801, 3))
    N = np.cross(V-Al, V-Ar)
    return N/np.linalg.norm(N, axis=1)[:, np.newaxis]

def polyline_from_points(points, radius=0.5):
    poly = pv.PolyData()
    poly.points = points
    the_cell = np.arange(0, len(points), dtype=np.int_)
    the_cell = np.insert(the_cell, 0, len(points))
    poly.lines = the_cell
    return poly.tube(radius=radius)

def draw_cpc_wireframe(p: pv.Plotter, V: np.ndarray, n = 10):
    for i in np.linspace(0, 100, n+1, dtype=np.int)[1:-1]-1:
        p.add_mesh( polyline_from_points(V[i,:]), color=wireframe_color)
    for j in np.linspace(0, 100, n+1, dtype=np.int)[1:-1]-1:
        p.add_mesh( polyline_from_points(V[:,j]), color=wireframe_color)


if __name__ == "__main__":
    V, F = igl.read_triangle_mesh("./data/12034_CPC.obj")
    N = igl.per_vertex_normals(V, F, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA)
    scalp = pv.PolyData()
    scalp.points, scalp.faces = V, np.column_stack([np.full((len(F), 1), 3), F])
    scalp["clip_normal"] = np.zeros(V.shape)
    scalp["clip_normal"][:9801] = cpc_clip_normal(V[:9801], V[-2], V[-1])

    p = pv.Plotter(window_size=(512, 512))
    p.set_background("white")
    # p.add_text("Scalp TBN Frame", font_size=18)
    draw_cpc_wireframe(p, V[:9801,:].reshape((99,99,-1)))
    p.add_mesh(scalp, ambient=0.06, diffuse=0.75, opacity=1, color=(1., 0.8, 0.7))#, scalars="yz_dot", cmap="jet")
    p.add_mesh(scalp.glyph(orient="clip_normal", factor=15, tolerance=0.05), show_scalar_bar=False)

    # p.show_axes_all()
    p.link_views()
    p.show(screenshot='./runtime/clip_normal.png')
    