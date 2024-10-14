# This is a modified Xplace

This repository holds a modified XPlace that removes features other than placement, while leaving the interface open for other tools to call. The original repository can be found [here](https://github.com/cuhk-eda/Xplace).

## About Xplace
Xplace is a fast and extensible GPU-accelerated global placement framework developed by the research team supervised by Prof. Evangeline F. Y. Young at The Chinese University of Hong Kong (CUHK). It achieves around 3x speedup per GP iteration compared to DREAMPlace and shows high extensibility.

More details are in the following paper:

Lixin Liu, Bangqi Fu, Shiju Lin, Jinwei Liu, Evangeline F.Y. Young, Martin D.F. Wong. "[Xplace: An Extremely Fast and Extensible Placement Framework](https://ieeexplore.ieee.org/document/10373583)". In IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), doi: 10.1109/TCAD.2023.3346291.

Lixin Liu, Bangqi Fu, Martin D. F. Wong, and Evangeline F. Y. Young. "[Xplace: an extremely fast and extensible global placement framework](https://doi.org/10.1145/3489517.3530485)". In Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC '22). Association for Computing Machinery, New York, NY, USA, 1309–1314. 

## How to Build

```bash
cd $XPLACE_HOME
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ..
make -j40 && make install
```
## License
Xplace is an open source project licensed under a BSD 3-Clause License that can be found in the [LICENSE](LICENSE) file.

