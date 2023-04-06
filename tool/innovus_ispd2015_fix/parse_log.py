import os
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser('')
parser.add_argument('log_file', type=str, default="./innovus_work/innovus.log")
args = parser.parse_args()

log_root = args.log_file
df = pd.DataFrame(columns=["placer", "design", "DRWL", "#DRVias", "#DRCs", "place_design_time", "DR Total Time (s)"])


with open(log_root, 'r') as f:
    loglines = f.readlines()

design_lines = []

for lid, line in enumerate(loglines):
    if line.startswith("=========== "):
        if len(design_lines) > 0:
            design_lines[-1][1] = lid
        design_lines.append([lid, None])
    if line.startswith("Finish all!"):
        design_lines[-1][1] = lid - 1
        break

print("#Designs:", len(design_lines))

for idx, (lhs, rhs) in enumerate(design_lines):
    design_line = loglines[lhs:rhs]
    findTotalWL, findTotalVia, findDRC = False, False, False
    DrWL, numVia, numDrc = None, None, None
    DrTotalTime, DrCpuTime = -1, -1
    PlaceTime = -1
    tmp = design_line[0].replace("=", "").strip()
    placer, design_name = tmp.split("/")
    for lid, line in enumerate(design_line):
        if "#% Begin route_design " in line and "place_design" in design_line[lid - 1]:
            PlaceTime = int(design_line[lid - 1].split()[2])
        if "End route_design" in line:
            DrCpuTimeStr = line.split("total cpu=")[-1].split(", ")[0]
            DrTotalTimeStr = line.split("real=")[-1].split(", ")[0]
            DrCpuTime = sum(x * int(t) for x, t in zip([3600, 60, 1], DrCpuTimeStr.split(":"))) # seconds
            DrTotalTime = sum(x * int(t) for x, t in zip([3600, 60, 1], DrTotalTimeStr.split(":"))) # seconds

        if line.startswith("Wire Length Statistics :"):
            findTotalWL = True
        if findTotalWL and "|     Total      |" in line:
            findTotalWL = False
            line = line.replace("|     Total      |", "")
            line = line.replace("|", "")
            line = line.replace("um", "")
            DrWL = float(line.strip())

        if line.startswith("Via Count Statistics :"):
            findTotalVia = True
        if findTotalVia and "|     Total      |" in line:
            findTotalVia = False
            line = line.replace("|     Total      |", "")
            line = line.replace("|", "")
            numVia = int(line.strip())

        if "Violation Summary By Layer and Type" in line:
            findDRC = True
        if findDRC and line.strip().startswith("Totals"):
            findDRC = False
            line = line.replace("Totals", "").strip()
            items = [int(i) for i in line.split(" ") if len(i) > 0]
            numDrc = items[-1]
        if "Verification Complete : 0 Viols" in line:
            findDRC = False
            numDrc = 0

    df.loc[idx] = [placer, design_name, DrWL, numVia, numDrc, PlaceTime, DrTotalTime]

df = df.sort_values(by=['placer', 'design'])
print(df)

csv_file = log_root.replace("innovus_work", "report").replace(".log", ".csv")
if not os.path.exists(os.path.dirname(csv_file)):
    os.makedirs(os.path.dirname(csv_file))
df.to_csv(csv_file)
