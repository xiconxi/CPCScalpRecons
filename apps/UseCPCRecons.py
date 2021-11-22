# from CPCScalpRecons import ScalpReconstruct/


from CPCScalpRecons import ScalpReconstruct, CPCSampling

from scipy import io as sio 

import numpy as np 

m = sio.loadmat("CPCScalpRecons/data/result.mat")
V, F, V_CPC, T, N, B = ScalpReconstruct(m['head_R'], m['ref_R'], cpc_inners=599)


CPC_uniform = CPCSampling.fibonacci_resampling(9801)

sub_index = [np.argmin(np.linalg.norm(V_CPC-cpc, axis=1)) for cpc in CPC_uniform]

sub_V   = V[sub_index]
sub_CPC = V_CPC[sub_index] # CPC_uniform




with open("lsmesh-cpc-fibonacci.obj", "w") as f:
    np.savetxt(f, sub_V, fmt="v %.4f %.4f %.4f")



# with open("lsmesh-cpc.obj", "w") as f:
#     # np.savetxt(f, sub_V, fmt="v %.4f %.4f %.4f")
#     # np.savetxt(f, F+1, fmt="f %d %d %d")
#     np.savetxt(f, np.column_stack([V, V_CPC[:, :]]), fmt="v %.4f %.4f %.4f 0 %f %f")
#     np.savetxt(f, F+1, fmt="f %d %d %d")