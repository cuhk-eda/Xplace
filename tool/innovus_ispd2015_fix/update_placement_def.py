import os
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('placement_root', type=str, default="yourpath/Xplace/result/2000-01-01-00:00:00")
args = parser.parse_args()

if __name__ == "__main__":
    dataset_root = "./ispd2015_fix"
    all_designs = sorted([i for i in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, i))])

    placers_info = {
        "xplace_route": os.path.join(args.placement_root, "/output/placement_%(design_name)s_dp.def"),
    }

    dst_pattern = "./ispd2015_fix_%(placer_name)s/%(design_name)s/%(design_name)s.def"

    for placer_name, def_file_info in placers_info.items():
        for design_name in all_designs:
            cur_design_src = def_file_info % {
                "design_name": design_name,
            }
            cur_design_dst = dst_pattern % {
                "placer_name": placer_name,
                "design_name": design_name,
            }
            if not os.path.exists(os.path.dirname(cur_design_dst)):
                os.makedirs(os.path.dirname(cur_design_dst))
            cmd = "cp -r %s %s" % (cur_design_src, cur_design_dst)
            print(cmd)
            os.system(cmd)
