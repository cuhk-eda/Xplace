## Run Innovus Detailed Routing for ISPD 2015 Fix

**NOTE**: If Innovus® has been properly installed in your OS, you may try to use Innovus® to detailedly route the placement solution.

1. Copy `ispd2015_fix` benchmark into this folder
```bash
cp -r $XPLACE_HOME/data/raw/ispd2015_fix $XPLACE_HOME/tool/innovus_ispd2015_fix
```
2. Copy Xplace solutions into this folder
```bash
cd $XPLACE_HOME/tool/innovus_ispd2015_fix
python update_placement_def.py yourpath/Xplace/result/2000-01-01-00:00:00
```
this script will create a folder named `ispd2015_fix_xplace_route`.
Now the file structure of the current folder looks like:
```
.
├── innovus_work
├── ispd2015_fix
├── ispd2015_fix_xplace_route
├── parse_log.py
└── update_placement_def.py
```
3. Run innovus detailed routing and wait around two days
```bash
rm -rf innovus.*
cd $XPLACE_HOME/tool/innovus_ispd2015_fix/innovus_work
innovus -stylus -init run_all_route_xplace_route.tcl
```
4. Parse innovus log and report DR metrics, the result will be saved in `./report/innovus.csv`
```bash
cd $XPLACE_HOME/tool/innovus_ispd2015_fix
python parse_log.py ./innovus_work/innovus.log
```