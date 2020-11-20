- Install:

```shell
pip install -r requirements.txt
conda install -c conda-forge igl, pyvista
```

- Usage: 

```python
from ScalpRecons import ScalpReconstruct
V, F, cpc, T, B, N = ScalpReconstruct(Vs, nilr, cpc_inners=99)
```