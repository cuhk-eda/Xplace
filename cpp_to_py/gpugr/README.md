# GGR: Superfast Full-Scale GPU-Accelerated Global Routing
GGR is a superfast full-scale GPU-accelerated global router developed by the research team supervised by Prof. Evangeline F. Y. Young and Prof. Martin D.F. Wong at The Chinese University of Hong Kong (CUHK). It includes an efficient and high-quality Z-shape pattern routing and a GPU-accelerated maze router GAMER.

More details are in the following paper:

Shiju Lin, Jinwei Liu, Evangeline F.Y. Young and Martin D.F. Wong. "[GAMER: GPU-Accelerated Maze Routing](https://ieeexplore.ieee.org/document/9799536)". In IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 42, no. 2, pp. 583-593, Feb. 2023. 

Shiju Lin and Martin D. F. Wong. "[Superfast Full-Scale GPU-Accelerated Global Routing](https://doi.org/10.1145/3508352.3549474)". In Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design (ICCAD '22). Association for Computing Machinery, New York, NY, USA, Article 51, 1–8. 

**GGR is integrated in Xplace now!**

## Notes and Limitations
- We default use `N=0` (without CUDA graph optimization). When `N >= 1`, an unknown error would occur before terminating the Python process. To enable CUDA graph optimization, please set `use_tf = true` in `cpp_to_py/gpugr/gr/MazeRoute.cu` and re-compile the project. 
- GGR is **deterministic** when `N=0` but not test in `N >= 1`.
- We currently only support LEF/DEF format.
- The runtime of this version is a little bit slower than the version described in the GGR paper because we use a stronger but slower parser and make the algorithm deterministic yet robust while sacrificing the runtime.

## Parameters
Please refer to `cpp_to_py/gpugr/PyBindCppMain.cpp`.

- `device_id`: GPU id.
- `route_xSize` / `route_ySize`: Given GR GridGraph size. If the given size is `0`, use the `GRIDGRAPH` definition in DEF file. If the given size is `0` and there is no definition in DEF, we set it as `512`.
- `rrrIters`: The number of rip-up and re-route iterations (maze route). If `rrrIters = 0`, perform the pattern route only.
- `csrn_scale`: The size of coarsen grid in maze routing.
- `route_guide`: The file name of output route guide.

## Citation
If you find **GGR** useful in your research, please consider to cite:
```bibtex
@inproceedings{lin2022ggr,
author = {Lin, Shiju and Wong, Martin D. F.},
booktitle = {Proceedings of the 41st IEEE/ACM International Conference on Computer-Aided Design},
title = {Superfast Full-Scale GPU-Accelerated Global Routing},
year = {2022},
}

@article{lin2023gamer,
  author={Lin, Shiju and Liu, Jinwei and Young, Evangeline F. Y. and Wong, Martin D. F.},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={GAMER: GPU-Accelerated Maze Routing}, 
  year={2023},
```


## Example
```python
# main_test_gr.py
import torch
import time

from utils import IOParser
from cpp_to_py import gpugr
from src import Flute
from src.core.route_force import calc_gr_wl_via, estimate_num_shorts

num_threads = 20
gpu_id = 0
Flute.register(num_threads)
torch.cuda.synchronize("cuda:{}".format(gpu_id))

# 1) Benchmark Setting
root = "your_path"
design_name = "ispd19_test9"
params = {
    "benchmark": "iccad2019",
    "lef": "%s/%s/%s.input.lef" % (root, design_name, design_name),
    "def": "%s/%s/%s.input.def" % (root, design_name, design_name),
    "design_name": design_name,
}
route_guide_file = "test.guide"

# 2) LEF/DEF Parser
print("--- Start GR ---")
start_gr_time = time.time()
parser = IOParser()
rawdb, gpdb = parser.read(
    params, verbose_log=True, lite_mode=True, random_place=False, num_threads=num_threads
)

# 3) Construct Global Routing Database and Run GGR
gpugr.load_gr_params(
    {
        "device_id": gpu_id,
        "route_xSize": 0,
        "route_ySize": 0,
        "rrrIters": 1,
        "route_guide": route_guide_file,
    }
)
grdb = gpugr.create_grdatabase(rawdb, gpdb)
routeforce = gpugr.create_routeforce(grdb)
routeforce.run_ggr()
end_gr_time = time.time()
print("--- End GR ---")

# 4) Report Global Routing Statistics
skip_m1_route = True
m1direction = gpdb.m1direction()  # 0 for H, 1 for V, metal1's layer idx is 0
hId = 1 if m1direction else 0
vId = 0 if m1direction else 1
if skip_m1_route:
    hId = hId + 2 if hId == 0 else hId
    vId = vId + 2 if vId == 0 else vId

dmd_map, wire_dmd_map, via_dmd_map = routeforce.dmd_map()
cap_map: torch.Tensor = routeforce.cap_map()

cg_mapH = dmd_map[hId::2].sum(dim=0) / cap_map[hId::2].sum(dim=0)
cg_mapV = dmd_map[vId::2].sum(dim=0) / cap_map[vId::2].sum(dim=0)
cg_mapHV = torch.stack((cg_mapH, cg_mapV))
cg_mapHV = torch.where(cg_mapHV > 1, cg_mapHV - 1, 0)

numOvflNets = routeforce.num_ovfl_nets()
gr_wirelength, gr_numVias = calc_gr_wl_via(grdb, routeforce)
gr_numShorts = estimate_num_shorts(routeforce, gpdb, cap_map, wire_dmd_map, via_dmd_map)

gr_time = end_gr_time - start_gr_time

print(
    "#OvflNets: %d, GR WL: %d, GR #Vias: %d, #EstShorts: %d | GR Time: %.4f"
    % (numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, gr_time)
)
```

## Contact

[Shiju Lin](https://appsrv.cse.cuhk.edu.hk/~sjlin/) (sjlin@cse.cuhk.edu.hk) and [Lixin Liu](https://liulixinkerry.github.io/) (lxliu@cse.cuhk.edu.hk)


## License

GGR is an open source project licensed under a BSD 3-Clause License that can be found in the [LICENSE](../../LICENSE) file.