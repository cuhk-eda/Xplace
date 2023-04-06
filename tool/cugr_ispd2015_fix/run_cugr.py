import os
import re
import pandas as pd
import argparse
pd.set_option('display.float_format', lambda x: '%.0f' % x)

parser = argparse.ArgumentParser('')
parser.add_argument('--dataset_root', type=str, default="yourpath/Xplace/data/raw/ispd2015_fix")
parser.add_argument('--placement_root', type=str, default="yourpath/Xplace/result/2000-01-01-00:00:00")
parser.add_argument('--log_file', type=str, default="./test_xplace_route.log")
parser.add_argument('-p', '--parse_log', action='store_true')
args = parser.parse_args()

def add_gcell(def_file, out_def_file):
    defcontent = ""
    findGcellDef = False
    die_lx, die_ly, die_hx, die_hy = None, None, None, None
    with open(def_file, 'r') as f:
        defLines = f.readlines()

    # compute num of route bins
    for lid, line in enumerate(defLines):
        if "DIEAREA" in line:
            die_lx, die_ly, die_hx, die_hy = [int(i) for i in re.findall('[0-9]+', line)]
            break

    route_size = 512
    die_ratio = (die_hx - die_lx) / (die_hy - die_ly)
    route_xSize = route_size if die_ratio <= 1 else round(route_size / die_ratio)
    route_ySize = route_size if die_ratio >= 1 else round(route_size / die_ratio)

    # update def file
    for lid, line in enumerate(defLines):
        if "GCELLGRID" in line:
            findGcellDef = True
            break

        if "DIEAREA" in line:
            die_lx, die_ly, die_hx, die_hy = [int(i) for i in re.findall('[0-9]+', line)]

        if "END DESIGN" in line:
            print("WARNING: Cannot find GCELLGRID in DEF. Add definition X: %i Y: %i" % (route_xSize, route_ySize))
            gcellContent = ""

            yStart = die_ly
            yDo = route_ySize
            yStep = (die_hy - die_ly) // route_ySize
            lastYStart = yStep * (yDo - 1) + yStart
            lastYDo = 2
            lastYStep = die_hy - lastYStart
            gcellContent += "GCELLGRID Y %i DO %i STEP %i ;\nGCELLGRID Y %i DO %i STEP %i ;\n" % (
                lastYStart, lastYDo, lastYStep, yStart, yDo, yStep
            )

            xStart = die_lx
            xDo = route_xSize
            xStep = (die_hx - die_lx) // route_xSize
            lastXStart = xStep * (xDo - 1) + xStart
            lastXDo = 2
            lastXStep = die_hx - lastXStart
            gcellContent += "GCELLGRID X %i DO %i STEP %i ;\nGCELLGRID X %i DO %i STEP %i ;\n" % (
                lastXStart, lastXDo, lastXStep, xStart, xDo, xStep
            )

            defcontent += "\n%s" % gcellContent

        defcontent += line

    if not findGcellDef:
        with open(out_def_file, 'w') as f:
            f.write(defcontent)
        print("Update DEF file with GCELLGRID.")
    else:
        os.system("cp %s %s" % (def_file, out_def_file))
        print("Find GCELLGRID. Not update.")


def parse_log(log_file):
    print("Reading %s..." % log_file)
    df = pd.DataFrame(columns=["placer", "design", "GR WL", "GR #Vias", "GR EstShort", "GR Score", "GR RunTime"])

    with open(log_file, 'r') as f:
        loglines = f.readlines()
    design_lines = []

    for lid, line in enumerate(loglines):
        if line.startswith("============ "):
            placer, design_name = line.split("============ start eval ")[-1].split(" ============")[0].split(" ")
            if len(design_lines) > 0:
                design_lines[-1][1] = lid
            design_lines.append([lid, None, placer, design_name])
        if " Terminated..." in line:
            design_lines[-1][1] = lid
    if design_lines[-1][1] == None:
        print("Warning: cannot find the EOL of the last design, discard it.")
        design_lines = design_lines[:-1]

    print("#Designs:", len(design_lines))

    for idx, (lhs, rhs, placer, design_name) in enumerate(design_lines):
        design_line = loglines[lhs:rhs]
        findEstScore = False
        wl, vias, short, score, gr_time = None, None, None, None, None
        for line in design_line:
            if "--- Estimated Scores ---" in line:
                findEstScore = True
            if findEstScore:
                if "wirelength | " in line:
                    wl = float(line.split("|")[1].strip())
                if "# vias     | " in line:
                    vias = float(line.split("|")[1].strip())
                if "short      | " in line:
                    short = float(line.split("|")[1].strip())
                if "total score = " in line:
                    score = float(line.split("total score = ")[1].strip())
                if "Writing guides to file..." in line:
                    gr_time = float(line.split("]")[0].split("[")[1].strip())
        df.loc[idx] = [placer, design_name, wl, vias, short, score, gr_time]
    print(df)
    df.to_csv(log_file.replace(".log", ".csv"))


def run_cugr(lef_file, def_file, num_threads=32, rrrIters=3):
    cugr_dir = "./CUGR"
    out_def_file = "%s/out_def.def" % cugr_dir
    add_gcell(def_file, out_def_file)
    cmd = "ln -s %s/PORT9.dat .; ln -s %s/POST9.dat .; ln -s %s/POWV9.dat .; \
               %s/iccad19gr -lef %s -def %s -output tmp.guide -threads %d -rrrIters %d; \
               rm tmp.guide ; rm heatmap.txt ; \
               rm PORT9.dat POST9.dat POWV9.dat ; \
               rm %s ; \
            " % (cugr_dir, cugr_dir, cugr_dir, 
                 cugr_dir, lef_file, out_def_file, num_threads, rrrIters,
                 out_def_file)
    print(cmd)
    os.system(cmd)


def run_eval():
    dataset_root = args.dataset_root
    all_designs = sorted([i for i in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, i))])

    placers_info = {
        "xplace_route": os.path.join(args.placement_root, "/output/placement_%(design_name)s_dp.def"),
    }
    for placer_name, def_file_info in placers_info.items():
        for design_name in all_designs:
            print("============ start eval %s %s ============" % (placer_name, design_name))
            lef_file = "%s/%s/%s.lef" % (dataset_root, design_name, design_name)
            def_file = def_file_info % {"design_name": design_name}
            run_cugr(lef_file, def_file)


if __name__ == "__main__":
    if args.parse_log:
        parse_log(args.log_file)
    else:
        run_eval()