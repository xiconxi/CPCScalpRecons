# from CPCScalpRecons import ScalpReconstruct/

import sys 
sys.path.append("../../")


from CPCScalpRecons import ScalpReconstruct, CPCSampling

from scipy import io as sio 
import numpy as np 
from scipy.spatial import KDTree

def generate_fibonacci():
    m = sio.loadmat("../data/result.mat")
    V, F, V_CPC, T, N, B = ScalpReconstruct(m['head_R'], m['ref_R'], cpc_inners=699)
    tree = KDTree(np.column_stack([V_CPC, np.zeros( (len(V_CPC), 1) )]))
    for cpc_scale in [9801, 9801*2, 9801*3]:
        print(cpc_scale)
        CPC_uniform = CPCSampling.fibonacci_resampling(cpc_scale)
        dd, sub_index = tree.query(np.column_stack([CPC_uniform, np.zeros( (len(CPC_uniform), 1) )]), k=1)
        # sub_index = [np.argmin(np.linalg.norm(V_CPC-cpc, axis=1)) for cpc in CPC_uniform]
        sub_CPC = V_CPC[sub_index] # CPC_uniform
        sub_V   = V[sub_index]
        
        np.savetxt("../data/fb-699-"+str(cpc_scale)+".npy", sub_index)
        np.savetxt("../data/fb-"+str(cpc_scale)+".txt", sub_CPC)
        with open("../data/fb-"+str(cpc_scale)+".obj", "w") as f:
            np.savetxt(f, sub_V, fmt="v %.4f %.4f %.4f")
            # to generate the triangle face, use meshlab's ball pivoting algorithm

def test_fibonacci_loading():
    CPC, F = CPCSampling.load_fibonacci(9801)
    print(F)

# with open("lsmesh-cpc.obj", "w") as f:
#     # np.savetxt(f, sub_V, fmt="v %.4f %.4f %.4f")
#     # np.savetxt(f, F+1, fmt="f %d %d %d")
#     np.savetxt(f, np.column_stack([V, V_CPC[:, :]]), fmt="v %.4f %.4f %.4f 0 %f %f")
#     np.savetxt(f, F+1, fmt="f %d %d %d")

if __name__ == "__main__":
    generate_fibonacci()
    test_fibonacci_loading()