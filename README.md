# Xplace-NN

Xplace-NN is a neural-enhanced extension of Xplace developed by the research team supervised by Prof. Evangeline F. Y. Young at The Chinese University of Hong Kong (CUHK).

More details are in the following paper:

Lixin Liu, Bangqi Fu, Martin D. F. Wong, and Evangeline F. Y. Young. "[Xplace: an extremely fast and extensible global placement framework](https://doi.org/10.1145/3489517.3530485)". In Proceedings of the 59th ACM/IEEE Design Automation Conference (DAC '22). Association for Computing Machinery, New York, NY, USA, 1309–1314. 

(For the Xplace, please refer to branch [main](https://github.com/cuhk-eda/Xplace/tree/main))

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
git checkout neural
```
2. Build the shared libraries used in Xplace.
```console
cd $XPLACE_HOME
mkdir build && cd build
cmake -DPYTHON_EXECUTABLE=$(which python) ..
make -j40 && make install
```

## Get started
The [pre-trained FNO model](https://github.com/cuhk-eda/Xplace/tree/neural/misc) is provided.

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

- To train FNO (optional):
```console
python main_train_fno.py
```
We observed non-determinism of FNO training loss, and we guessed it might cause by the floating point atomic add when dynamically generating a density map. One possible way to mitigate the non-determinism is to pre-generate all used density maps, but it is memory-consuming.

## Parameters
Please refer to `main.py` and `main_train_fno.py`.


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

## Xplace-NN Placement Results

Benchmark | Placement Solutions
|:---:|:---:|
ISPD2005 | [Google Drive](https://drive.google.com/drive/folders/1oV9xlp2VcP0ShZLjdXQhyO9HP5eKNr8t?usp=sharing)

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
