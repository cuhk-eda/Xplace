# This script removes the fence region in ispd2019_test5

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# set dataset root
dataset_root = "./raw"
# remove fence region related information, including GROUP, REGION
removeDefFence = True


def generate_one_raw_design(input_root, output_root, design_name):
    print("==============")
    design_root = os.path.join(input_root, design_name)
    #################### lef ####################
    defFile = os.path.join(design_root, "%s.input.def" % design_name)
    lefFile = os.path.join(design_root, "%s.input.lef" % design_name)
    outputLef = "%s/%s/%s.input.lef" % (output_root, design_name, design_name)
    if not os.path.exists(os.path.dirname(outputLef)):
        os.makedirs(os.path.dirname(outputLef))

    if design_name == "ispd19_test5":
        with open(defFile) as t:
            defLines = t.readlines()

        #################### def ####################
        defoutputFile = "%s/%s/%s.input.def" % (output_root, design_name, design_name)

        defcontent = generateDefContent(defLines, design_name)

        with open(defoutputFile, 'w') as f:
            f.write(defcontent)
        
        print(defoutputFile)
    else:
        os.system("cp %s %s/%s/%s.input.def" % (defFile, output_root, design_name, design_name))

    os.system("cp %s %s/%s/%s.input.lef" % (lefFile, output_root, design_name, design_name))

def generateDefContent(defLines, design_name):
    defcontent = ""
    
    isRegion, isGroup = False, False
    for lid, line in enumerate(defLines):

        if removeDefFence:
            if "REGIONS" in line and "END" not in line:
                isRegion = True
                continue
            if "END REGIONS" in line:
                isRegion = False
                continue
            if "GROUPS" in line and "END" not in line:
                isGroup = True
                continue
            if "END GROUPS" in line:
                isGroup = False
                continue
            if isGroup or isRegion:
                continue

        defcontent += line
    return defcontent

if __name__ == "__main__":
    raw_root = os.path.join(dataset_root, "ispd2019")
    all_designs = os.listdir(raw_root)
    sorted(all_designs)

    # Raw design clean
    print("-------------------------------------------------")
    output_root = os.path.join(dataset_root, "ispd2019_no_fence")
    for design_name in all_designs:
        generate_one_raw_design(raw_root, output_root, design_name)