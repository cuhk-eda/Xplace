import os
import sys
import csv
import torch
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.io_parser import IOParser
from utils.get_design_params import get_multiple_design_params

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='./raw/')
parser.add_argument("--dataset", type=str, default="ispd2015")

def main():
    args = parser.parse_args()
    params_mul = get_multiple_design_params(args.dataset_root, args.dataset)
    convert_design_to_torch_data_multiple(params_mul)


def convert_design_to_torch_data_multiple(params_mul):
    # start generating...
    benchmark = params_mul[0]["benchmark"]
    dataset_dir = os.path.join("./cad/%s" % benchmark)

    datalist_path = os.path.join(dataset_dir, "datalist.csv")
    if not os.path.exists(os.path.dirname(datalist_path)):
        os.makedirs(os.path.dirname(datalist_path))
    csvfile = open(datalist_path, "w")
    csvwriter = csv.writer(
        csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )

    for params in params_mul:
        design_path = "%s/%s.pt" % (dataset_dir, params["design_name"])
        print("---- processing %s ----" % params["design_name"])
        design_info = convert_design_to_torch_data_kernel(params)
        torch.save(design_info, design_path)
        csvwriter.writerow([design_path])

    csvfile.close()


def convert_design_to_torch_data_kernel(params):
    t1 = time.time()
    parser = IOParser()
    rawdb, gpdb = parser.read(
        params, verbose_log=False, lite_mode=True, random_place=False
    )
    t2 = time.time()
    design_info = parser.preprocess_design_info(gpdb)
    t3 = time.time()
    print(
        "%s | Parsing time: %.4fs Generate Tensor time: %.4fs"
        % (params["design_name"], (t2 - t1), (t3 - t2))
    )
    del gpdb, rawdb, parser

    return design_info


if __name__ == "__main__":
    main()
