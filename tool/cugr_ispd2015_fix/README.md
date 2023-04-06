## Run CUGR Global Routing for ISPD 2015 Fix

**NOTE**: CU-GR [binary](tool/cugr_ispd2015_fix/CUGR/iccad19gr) is already included under the root.

1. Run CUGR to evaluate the placement solutions
```bash
python -u run_cugr.py --dataset_root yourpath/Xplace/data/raw/ispd2015_fix --placement_root yourpath/Xplace/result/2000-01-01-00:00:00 | tee test_xplace_route.log
```
2. Parse CUGR log and report GR metrics, the result will be saved in `test_xplace_route.csv`
```bash
python run_cugr.py --parse --log_file ./test_xplace_route.log
```