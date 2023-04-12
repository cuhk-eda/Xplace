# Xplace-NN

Xplace-NN is a neural-enhanced extension of Xplace developed by the research team supervised by Prof. Evangeline F. Y. Young at The Chinese University of Hong Kong (CUHK). Xplace-NN supports deterministic mode now.

## About
Xplace is a fast and extensible GPU-accelerated global placement framework developed by the research team supervised by Prof. Evangeline F. Y. Young at The Chinese University of Hong Kong (CUHK). It achieves around 3x speedup per GP iteration compared to DREAMPlace and shows high extensibility.


As shown in the following figure, Xplace framework is built on top of PyTorch and consists of several independent modules. One can easily extend Xplace by applying scheduling techniques, gradient functions, new placement metrics, and so on.

<div align="center">
  <img src="img/xplace_overview.png" width="300"/>
</div>

More details are in the following paper:

Lixin Liu, Bangqi Fu, Martin D. F. Wong, and Evangeline F. Y. Young. "[Xplace: an extremely fast and extensible global placement framework](https://doi.org/10.1145/3489517.3530485)". In Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC '22). Association for Computing Machinery, New York, NY, USA, 1309–1314. 

(For the Xplace, please refer to branch [main](https://github.com/cuhk-eda/Xplace/tree/main))

## Requirements
- [CMake](https://cmake.org/) >= 3.12
- [GCC](https://gcc.gnu.org/) >= 7.5.0
- [Boost](https://www.boost.org/) >= 1.56.0
- [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 11.3
- [Python](https://www.python.org/) >= 3.8
- [PyTorch](https://pytorch.org/) >= 1.12.0
- [Cairo](https://www.cairographics.org/)
- [Innovus®](https://www.cadence.com/content/cadence-www/global/en_US/home/tools/digital-design-and-signoff/soc-implementation-and-floorplanning/innovus-implementation-system.html) (version 20.14, optional, for detailed routing and design rule checking)

## Setup
1. Clone the Xplace repository. We'll call the directory that you cloned Xplace as `$XPLACE_HOME`.
```bash
git clone --recursive https://github.com/cuhk-eda/Xplace
git checkout neural
```
2. Build the shared libraries used in Xplace.
```bash
cd $XPLACE_HOME
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ..
make -j40 && make install
```

## Prepare data
The following script will automatically download `ispd2005`, `ispd2015`, and `iccad2019` benchmarks in `./data/raw`. It also preprocesses `ispd2015` benchmark to fix some errors when routing them by Innovus®.
```bash
cd $XPLACE_HOME/data
./download_data.sh
```

## Get started
The [pre-trained FNO model](https://github.com/cuhk-eda/Xplace/tree/neural/misc) is provided.
- To run GP + DP flow for ISPD2005 dataset:
```bash
# only run adaptec1
python main.py --dataset ispd2005 --design_name adaptec1 --load_from_raw True --detail_placement True

# run all the designs in ispd2005
python main.py --dataset ispd2005 --run_all True --load_from_raw True --detail_placement True
```

- To run GP + DP flow for ISPD2015 dataset:
```bash
# only run mgc_fft_1
python main.py --dataset ispd2015_fix --design_name mgc_fft_1 --load_from_raw True --detail_placement True

# run all the designs in ispd2015
python main.py --dataset ispd2015_fix --run_all True --load_from_raw True --detail_placement True
```

- To run Routability GP + DP flow for ISPD2015 dataset:
```bash
# run all the designs in ispd2015 with routability optimization
python main.py --dataset ispd2015_fix --run_all True --load_from_raw True --detail_placement True --use_cell_inflate True
```

**NOTE**: We default enable the deterministic mode. If you don't need determinism and want to run placement in an extremely fast mode, please try to set `--deterministic False` in the Python arguments.

- Each run will generate several output files in `./result/exp_id`. These files can provide valuable information for parameter tuning.
```
In ./result/exp_id
   - eval    # parameter curves and the visualization of placement solutions
   - log     # log and statistics
   - output  # placement solutions
```

- To train FNO (optional):
```console
python main_train_fno.py
```

## Parameters
Please refer to `main.py` and `main_train_fno.py`.

## Run custom dataset
You can use the argument `--custom_path` to run your custom LEF/DEF or bookshelf benchmark.

Suppose there is a LEF/DEF benchmark named `toy` in `data/raw`, you can use the following command line to run the GP + DP flow:
```bash
python main.py --custom_path lef:data/raw/toy_input.lef,def:data/raw/toy_input.def,design_name:toy,benchmark:test --load_from_raw True --detail_placement True
```


## Load design from preprocessed `pt` file (Optional)
The following script will dump the parsed design into a single torch `pt` file so Xplace can load the design from the `pt` file instead of parsing the input file from scratch. 

```bash
cd $XPLACE_HOME/data
python convert_design_to_torch_data.py --dataset ispd2005
python convert_design_to_torch_data.py --dataset ispd2015_fix
python convert_design_to_torch_data.py --dataset iccad2019
```
Preprocessed data is saved in `./data/cad`.

To develop a new global placement technique in Xplace, we highly suggest using the `pt` mode to save the parser time. (set `--load_from_raw False`)

```bash
python main.py --dataset ispd2005 --run_all True --load_from_raw False
```

**NOTE**: 
1. Please remember to use the raw mode (set `--load_from_raw True`) when measuring the total running time.
2. We currently do not support `pt` mode in the routability-driven global placement.
3. If you want to run `pt` mode for the custom dataset, you need to add the custom dataset path in `utils/get_design_params.py`.

## GPU-accelerated place and global route flow (Xplace + GGR)
Set `--final_route_eval True` in Python arguments to invoke the internal global router [GGR](https://dl.acm.org/doi/10.1145/3508352.3549474) to run GPU-accelerated PnR flow. The flow will output the **placement DEF** and the **global routing guide** in `./result/exp_id/output`. Besides, GR metrics are reported in the log and recorded in `./result/exp_id/log/route.csv`. 

- To run Place and Global Route flow for ISPD2015 dataset:
```bash
python main.py --dataset ispd2015_fix --run_all True --load_from_raw True --detail_placement True --use_cell_inflate True --final_route_eval True
```

More details about using GGR in Xplace can be found in [cpp_to_py/gpugr/README.md](cpp_to_py/gpugr/README.md).

## Evaluate the routability of Xplace's solution 
We provide three ways to evaluate the routability of a placement solution:

1. Set `--final_route_eval True` to invoke [GGR](https://dl.acm.org/doi/10.1145/3508352.3549474) to evaluate the placement solution.

2. Use [CU-GR](https://github.com/cuhk-eda/cu-gr) to evaluate the placement solution by global routing. Please refer to [tool/cugr_ispd2015_fix/README.md](tool/cugr_ispd2015_fix/README.md) for instructions.

3. (Optional). If Innovus® has been properly installed in your OS, you may try to use Innovus® to detailedly route the placement solution. Please refer to [tool/innovus_ispd2015_fix/README.md](tool/innovus_ispd2015_fix/README.md) for instructions.


## Citation
If you find **Xplace** or **Xplace-NN** useful in your research, please consider to cite:
```bibtex
@inproceedings{liu2022xplace,
    author={Liu, Lixin and Fu, Bangqi and Wong, Martin D. F. and Young, Evangeline F. Y.},
    booktitle={Proceedings of the 59th ACM/IEEE Design Automation Conference},
    title={Xplace: An Extremely Fast and Extensible Global Placement Framework},
    year={2022},
}
```

Thanks the authors of [ePlace](https://dl.acm.org/doi/10.1145/2699873), [RePlAce](https://github.com/The-OpenROAD-Project/RePlAce), [DREAMPlace](https://github.com/limbo018/DREAMPlace), and [FNO](https://github.com/zongyi-li/fourier_neural_operator) for their great work.
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

@inproceedings{li2021fourier,
    author={Zongyi Li and Nikola Borislavov Kovachki and Kamyar Azizzadenesheli and Burigede liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
    booktitle={International Conference on Learning Representations},
    title={Fourier Neural Operator for Parametric Partial Differential Equations},
    year={2021},
}
```


## Contact

[Lixin Liu](https://liulixinkerry.github.io/) (lxliu@cse.cuhk.edu.hk)
 and Bangqi Fu (bqfu21@cse.cuhk.edu.hk)


## License

Xplace is an open source project licensed under a BSD 3-Clause License that can be found in the [LICENSE](LICENSE) file.
