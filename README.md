# Xplace

Xplace is a fast and extensible GPU accelerated global placement framework developed by the research team supervised by Prof. Evangeline F. Y. Young at The Chinese University of Hong Kong (CUHK). It achieves around 3x speedup per GP iteration compared to the state-of-the-art global placer DREAMPlace and shows high extensiblity.


As shown in the following figure, Xplace framework is built on top of PyTorch and consists of serveral independent modules. One can easily extend Xplace by applying new scheduling techniques, new gradient functions, new placement metrics and so on.

<div align="center">
  <img src="assets/xplace_overview.png" width="300"/>
</div>

More details are in the following paper:

Lixin Liu, Bangqi Fu, Martin D. F. Wong, and Evangeline F. Y. Young. "[Xplace: an extremely fast and extensible global placement framework](https://doi.org/10.1145/3489517.3530485)". In Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC '22). Association for Computing Machinery, New York, NY, USA, 1309–1314. 

(For the Xplace-NN, please refer to branch [neural](https://github.com/cuhk-eda/Xplace/tree/neural))

## Requirements
- CMake >= 3.12
- GCC >= 7.5.0
- Boost >= 1.56.0
- CUDA >= 11.0
- Python >= 3.8
- PyTorch >= 1.10.1
- Cairo


## Setup
1. Clone the Xplace repository. We'll call the directory that you cloned Xplace as `$XPLACE_HOME`.
```console
git clone --recursive https://github.com/cuhk-eda/Xplace
```
2. Build the shared libraries used in Xplace.
```console
cd $XPLACE_HOME
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ..
make -j40 && make install
```

## Get started

- To run GP only flow for all the designs in ISPD2005 dataset:
```console
python main.py --dataset_root your_path --dataset ispd2005 --run_all True --load_from_raw True --write_placement True
```

- To run GP + DP flow for `adaptec1` in ISPD2005 dataset:
```console
python main.py --dataset_root your_path --dataset ispd2005 --design_name adaptec1 --load_from_raw True --write_placement True --detail_placement True
```

**Note**: For ISPD2005 dataset, [NTUplace3](http://eda.ee.ntu.edu.tw/research.htm) is used as the detailed placement engine. For ISPD2015 dataset, please run GP only flow and launch [ABCDPlace](https://github.com/limbo018/DREAMPlace) to perform detailed placement.

- Each run will generate serveral output files in `./result/exp_id`. These files can provide valuable information for parameter tuning.
```
In ./result/exp_id
   - eval    # parameter curves and the visualization of placement
   - log     # log and statistics
   - output  # global placement solution files
```

## Parameters
Please refer to `main.py`.


## Load design from preprocessed `pt` file (Optional)
The following script will dump the parsed design into a single torch `pt` file so Xplace can load the design from the `pt` file instead of parsing the input file from scratch. 

```console
cd $XPLACE_HOME
python utils/convert_design_to_torch_data.py --dataset_root your_path --dataset ispd2005
```
Preprocessed data is saved in `./data/cad`.

When developing a new global placement technique in Xplace, we highly suggest using the `pt` mode to save the parser time. (set `--load_from_raw False`)

```console
python main.py --dataset ispd2005 --run_all True --load_from_raw False
```

**Note**: Please remember to use the raw mode (set `--load_from_raw True`) when running detailed placement or measuring the total running time.

## Xplace Placement Results

Benchmark | Placement Solutions
|:---:|:---:|
ISPD2005 | [Google Drive](https://drive.google.com/drive/folders/1fUzkT9ymV3n0XxfWXA0mR3WQX55hR1PB?usp=sharing)
ISPD2015 (w/o fence) | [Google Drive](https://drive.google.com/drive/folders/1UsKQ1FQ4fFi4pdJ0VoCoCCjLakhoS20Q?usp=sharing)

## Citation
If you find **Xplace** useful in your research, please consider to cite:
```bibtex
@inproceedings{liu2022xplace,
    author={Liu, Lixin and Fu, Bangqi and Wong, Martin D. F. and Young, Evangeline F. Y.},
    booktitle={Proceedings of the 59th ACM/IEEE Design Automation Conference},
    title={Xplace: An Extremely Fast and Extensible Global Placement Framework},
    year={2022},
}
```

Thanks the authors of [ePlace](https://dl.acm.org/doi/10.1145/2699873), [RePlAce](https://github.com/The-OpenROAD-Project/RePlAce), and [DREAMPlace](https://github.com/limbo018/DREAMPlace) for their great work.
```bibtex
@article{lu2015eplace,
    author={Lu, Jingwei and Chen, Pengwen and Chang,   Chin-Chih and Sha, Lu and Huang, Dennis Jen-Hsin and   Teng, Chin-Chi and Cheng, Chung-Kuan},
    journal={ACM Trans. Des. Autom. Electron. Syst.},
    title={ePlace: Electrostatics-Based Placement Using   Fast Fourier Transform and Nesterov's Method},
    year={2015},
}

@article{cheng2019replace,
    author={Cheng, Chung-Kuan and Kahng, Andrew B. and Kang, Ilgweon and Wang, Lutong},
    journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
    title={RePlAce: Advancing Solution Quality and Routability Validation in Global Placement}, 
    year={2019},
}

@article{lin2021dreamplace,
    author={Lin, Yibo and Jiang, Zixuan and Gu, Jiaqi and Li, Wuxi and Dhar, Shounak and Ren, Haoxing and Khailany, Brucek and Pan, David Z.},
    journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
    title={DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement}, 
    year={2021},
}
```


## Contact

[Lixin Liu](https://liulixinkerry.github.io/) (lxliu@cse.cuhk.edu.hk)
 and Bangqi Fu (bqfu21@cse.cuhk.edu.hk)


## License

READ THIS LICENSE AGREEMENT CAREFULLY BEFORE USING THIS PRODUCT. BY USING THIS PRODUCT YOU INDICATE YOUR ACCEPTANCE OF THE TERMS OF THE FOLLOWING AGREEMENT. THESE TERMS APPLY TO YOU AND ANY SUBSEQUENT LICENSEE OF THIS PRODUCT.

License Agreement for Xplace

Copyright (c) 2022 by The Chinese University of Hong Kong

All rights reserved

CU-SD LICENSE (adapted from the original BSD license) Redistribution of the any code, with or without modification, are permitted provided that the conditions below are met.

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name nor trademark of the copyright holder or the author may be used to endorse or promote products derived from this software without specific prior written permission.

Users are entirely responsible, to the exclusion of the author, for compliance with (a) regulations set by owners or administrators of employed equipment, (b) licensing terms of any other software, and (c) local, national, and international regulations regarding use, including those regarding import, export, and use of encryption software.

THIS FREE SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR ANY CONTRIBUTOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, EFFECTS OF UNAUTHORIZED OR MALICIOUS NETWORK ACCESS; PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.