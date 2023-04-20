# This script fixes some problems reported by Innovus Nanoroute when running ISPD 2015

import os
import sys
import re
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# set dataset root
dataset_root = "./raw"
# remove fence region related information, including GROUP, REGION
removeDefFence = True 
# Since there are so many DRCs caused by SNet Vias (spacing) when routing by nanoroute, 
# we remove these SNet Vias to avoid them. Note that such modification won't affect placement.
removeDefSNetVias = True
# Some fixed cell is not placed on manufacture grid, we move them to the nearest grid
fixDefPlaceOnManGrid = True # only for mgc_fft_b, mgc_matrix_mult_b
# Some cell_type pin is not placed on manufacture grid, we move them to the nearest grid
fixLefPinOnManGrid = True # only for mgc_des_perf_a, mgc_matrix_mult_a, mgc_matrix_mult_b, mgc_matrix_mult_c
# single track definition for multiple layers is not supported by CUGR parser (rsyn)
fixDefTracksLayers = True # only for mgc_superblue_*
# It seems that the cell "SOURCE DIST" would cause some errors...
removeDefSrcInst = True # only for mgc_superblue_*
# innovus cannot support Via definition in DEF file
moveViaToLef = True 
# NDR should be defined in LEF since it is used by macro pin
moveNDRToLef = True 
# Add default to fix nanoroute via error. 
# FIXME: I dont' know which vias in superblue should be set with default...
addDefaultForLefVia = True # only for mgc_superblue_*
# PG Nets will cause a lot of violations with macro OBS, use EXCEPTPGNET to avoid that
eceptPGNetsForObs = True


def generate_one_raw_design(input_root, output_root, design_name):
    print("==============")
    design_root = os.path.join(input_root, design_name)
    #################### lef ####################
    techFile = os.path.join(design_root, "tech.lef")
    cellFile = os.path.join(design_root, "cells.lef")
    defFile = os.path.join(design_root, "floorplan.def")
    constraintsFile = os.path.join(design_root, "placement.constraints")
    lefoutputFile = "%s/%s/%s.lef" % (output_root, design_name, design_name)
    techoutputFile = "%s/%s/tech.lef" % (output_root, design_name)
    celloutputFile = "%s/%s/cells.lef" % (output_root, design_name)
    if not os.path.exists(os.path.dirname(lefoutputFile)):
        os.makedirs(os.path.dirname(lefoutputFile))

    with open(techFile) as t:
        techLines = t.readlines()
    with open(cellFile) as t:
        cellLines = t.readlines()
    with open(defFile) as t:
        defLines = t.readlines()

    lefcontent, (techcontent, cellcontent), (_, lefManGrid) = generateLefContent(techLines, cellLines, defLines, design_name)

    with open(lefoutputFile, 'w') as f:
        f.write(lefcontent)

    with open(techoutputFile, 'w') as f:
        f.write(techcontent)

    with open(celloutputFile, 'w') as f:
        f.write(cellcontent)
    
    print(lefoutputFile)

    #################### def ####################
    defoutputFile = "%s/%s/%s.def" % (output_root, design_name, design_name)

    defcontent = generateDefContent(defLines, design_name)

    with open(defoutputFile, 'w') as f:
        f.write(defcontent)
    
    print(defoutputFile)
    os.system("cp %s %s/%s/placement.constraints" % (constraintsFile, output_root, design_name))

def generate_one_placer_def(input_root, output_root, design_name, placer):
    print("==============")
    design_root = os.path.join(input_root + placer)
    #################### lef ####################
    defFile = os.path.join(design_root, "%s.def" % design_name)
    if not os.path.exists(defFile):
        print("Error! cannot find %s" % defFile)
        return
    defoutputFile = "%s_%s/%s.def" % (output_root, placer, design_name)
    if not os.path.exists(os.path.dirname(defoutputFile)):
        os.makedirs(os.path.dirname(defoutputFile))

    with open(defFile) as t:
        defLines = t.readlines()

    defcontent = generateDefContent(defLines, design_name)

    with open(defoutputFile, 'w') as f:
        f.write(defcontent)
    
    print(defoutputFile)


def generateLefContent(techLines, cellLines, defLines, design_name):
    lefdbunit = 1000
    lefManGrid = 0.001
    jumpHeader = False
    hasUnit = False
    lefcontent = "VERSION 5.8 ;\nBUSBITCHARS \"[]\" ;\nDIVIDERCHAR \"/\" ;\n"
    techcontent = "VERSION 5.8 ;\nBUSBITCHARS \"[]\" ;\nDIVIDERCHAR \"/\" ;\n"
    for lid, line in enumerate(techLines):
        if not jumpHeader:
            if "UNITS" in line:
                jumpHeader = True
                hasUnit = True
            else:
                continue
        if hasUnit:
            if "DATABASE MICRONS" in line:
                tmp = line.replace("DATABASE MICRONS", "")
                tmp = tmp.replace(";", "")
                lefdbunit = int(tmp.strip())
                print("lefdbunit: %d" % lefdbunit)
                hasUnit = False
        if "MANUFACTURINGGRID" in line:
            lefManGrid = float(line.split("MANUFACTURINGGRID")[1].replace(";", "").strip())
            print("lefManGrid:", lefManGrid)
        if "PROPERTY " in line:
            prefix = re.split(r'PROPERTY', line, 1)[0]
            newline = prefix + re.search(r'"(.*?)"', line).group(1) + '\n'
            # print("Rewrite:\n%s%s" % (line, newline))
            line = newline
        if "TOPOFSTACKONLY" in line:
            line = line.replace("TOPOFSTACKONLY", "")
        if "END LIBRARY" in line:
            techcontent += line
            continue

        if line.startswith("VIA ") and "superblue" in design_name and addDefaultForLefVia:
            line = line.replace("\n", " DEFAULT\n")

        techcontent += line
        lefcontent += line
    lefcontent += "\n"

    if moveViaToLef:
        lefcontent += generateLefViaFromDef(techLines, cellLines, defLines, lefdbunit, design_name)
    if moveNDRToLef:
        lefcontent += generateLefNDRFromDef(defLines, lefdbunit)

    # for cell lef
    macroStart = False
    cellcontent = ""
    findObs = False
    for lid, line in enumerate(cellLines):
        if not macroStart:
            if "MACRO" in line:
                macroStart = True
            else:
                cellcontent += line
                continue

        if fixLefPinOnManGrid:
            if "des_perf_a" in design_name:
                if "296.2961" in line:
                    line = line.replace("296.2961", "296.295")
                if "296.8061" in line:
                    line = line.replace("296.8061", "296.805")
            if "matrix_mult_a" in design_name:
                if "RECT 683.8535 367.49 683.9535 368 ;" in line:
                    line = line.replace("683.8535", "683.855")
                    line = line.replace("683.9535", "683.955")
                if "RECT 708.0535 367.49 708.1535 368 ;" in line:
                    line = line.replace("708.0535", "708.055")
                    line = line.replace("708.1535", "708.155")
            if "matrix_mult_b" in design_name:
                if "235.518 " in line:
                    line = line.replace("235.518 ", "235.520 ")
                if "236.028 " in line and "RECT" in line:
                    line = line.replace("236.028 ", "236.030 ")
                if "235.773 " in line and "RECT" in line:
                    line = line.replace("235.773 ", "235.775 ")
            if "matrix_mult_c" in design_name:
                if "235.518 " in line:
                    line = line.replace("235.518 ", "235.520 ")
                if "236.028 " in line and "RECT" in line:
                    line = line.replace("236.028 ", "236.030 ")
                if "235.773 " in line and "RECT" in line:
                    line = line.replace("235.773 ", "235.775 ")

        if eceptPGNetsForObs:
            if "OBS" in line:
                findObs = True
            if findObs:
                if "END" in line:
                    findObs = False
                if "LAYER" in line:
                    line = line.replace(";", "EXCEPTPGNET ;")

        cellcontent += line
        lefcontent += line
    
    lefInfo = (lefdbunit, lefManGrid)

    return lefcontent, (techcontent, cellcontent),lefInfo

def generateDefContent(defLines, design_name):
    defcontent = ""
    
    isSNet, isVia, isNdr = False, False, False
    isRegion, isGroup = False, False
    jumpToEndComponents = False
    for lid, line in enumerate(defLines):
        if fixDefTracksLayers:
            if "TRACKS X" in line or "TRACKS Y" in line:
                prevContent = line.split("LAYER")[0].strip()
                layerNames = line.split("LAYER")[1].replace(";", " ").strip()
                layerNames = layerNames.split(" ")
                if len(layerNames) > 1:
                    for layerName in layerNames:
                        defcontent += "%s LAYER %s ;\n" % (prevContent, layerName)
                    continue

        if removeDefSNetVias:
            if "SPECIALNETS" in line and "END" not in line:
                isSNet = True
                defcontent += generateDefSNet(defLines)
                continue
            if "END SPECIALNETS" in line:
                isSNet = False
                continue
            if isSNet:
                continue

        if moveViaToLef:
            if "VIAS" in line and "END" not in line:
                isVia = True
                continue
            if "END VIAS" in line:
                isVia = False
                continue
            if isVia:
                continue

        if moveNDRToLef:
            if "NONDEFAULTRULES" in line and "END" not in line:
                isNdr = True
                continue
            if "END NONDEFAULTRULES" in line:
                isNdr = False
                continue
            if isNdr:
                continue
        
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

        if fixDefPlaceOnManGrid and "fft_b" in design_name:
            if "+ PLACED ( 661359 799490 ) N" in line:
                line = line.replace("661359", "661360")
            if "+ PLACED ( 799490 769536 ) N" in line:
                line = line.replace("769536", "769535")
            if "END COMPONENTS" in line:
                jumpToEndComponents = False
                continue
            if jumpToEndComponents:
                continue
            if "- h0 h0" in line and not jumpToEndComponents:
                jumpToEndComponents = True
                defcontent += """   - h0 h0 
      + FIXED ( 469910 641190 ) N ;
   - h1 h1 
      + FIXED ( 487815 435385 ) N ;
   - h2 h2 
      + FIXED ( 11925 539600 ) N ;
   - h3 h3 
      + FIXED ( 12690 8940 ) N ;
   - h4 h4 
      + FIXED ( 12960 262690 ) N ;
   - h5 h5 
      + FIXED ( 488385 238550 ) N ;
END COMPONENTS
"""
                continue
        if fixDefPlaceOnManGrid and "edit_dist_a" in design_name:
            if "END COMPONENTS" in line:
                jumpToEndComponents = False
                continue
            if jumpToEndComponents:
                continue
            if "- h1 h1" in line and not jumpToEndComponents:
                jumpToEndComponents = True
                defcontent += """   - h1 h1 
      + FIXED ( 55600 485305 ) N ;
   - h2 h2 
      + FIXED ( 55075 303345 ) N ;
   - h3 h3 
      + FIXED ( 54610 43650 ) N ;
   - h6 h6 
      + FIXED ( 572870 56040 ) N ;
   - h5 h5 
      + FIXED ( 570070 311740 ) N ;
   - h4 h4 
      + FIXED ( 567470 489045 ) N ;
END COMPONENTS
"""
                continue
        if fixDefPlaceOnManGrid and "matrix_mult_a" in design_name:
            if "+ PLACED ( 761508 1499490 ) N" in line: 
                line = line.replace("761508", "761510")
            if "+ PLACED ( 737308 1499490 ) N" in line: 
                line = line.replace("737308", "737310")
            if "+ PLACED ( 761708 1499490 ) N" in line: 
                line = line.replace("761708", "761710")
        if fixDefPlaceOnManGrid and "matrix_mult_b" in design_name:
            if "+ FIXED ( 49400 1082561 ) N ;" in line: 
                line = line.replace("1082561", "1082560")
            if "+ FIXED ( 1191335 728359 ) N ;" in line:
                line = line.replace("728359", "728360")
        if fixDefPlaceOnManGrid and "matrix_mult_c" in design_name:
            if "+ FIXED ( 49400 1082561 ) N ;" in line: 
                line = line.replace("1082561", "1082560")
        if removeDefSrcInst and "SOURCE DIST" in line and "superblue" in design_name:
            continue

        defcontent += line
    return defcontent

def generateDefSNet(defLines):
    isSNets = False
    snets = []
    curSnet = None
    for lid, line in enumerate(defLines):
        if "SPECIALNETS" in line and "END" not in line:
            isSNets = True
            continue
        if not isSNets:
            continue
        if "END SPECIALNETS" in line:
            if curSnet is not None:
                snets.append(curSnet)
            break
        tmp = line.strip()
        if len(tmp) == 0:
            continue
        if tmp[0] == '-':
            if curSnet is not None:
                snets.append(curSnet)
            curSnet = []
        if "via" in line.lower() and "metal" in line.lower() and ("NEW" in line or "ROUTED" in line):
            # Remove snet vias (incur a lot of DRCs)
            continue
        curSnet.append(line)
    print("#Snets: %d" % len(snets))
    allDefSnetsContent = "SPECIALNETS %d ;\n" % len(snets)
    for curSnet in snets:
        if len(curSnet) > 1:
            if "+ ROUTED" not in curSnet[1] and "metal" in curSnet[1].lower():
                curSnet[1] = curSnet[1].replace("NEW", "+ ROUTED")
        allDefSnetsContent += curSnet[0]
        allDefSnetsContent += "".join(curSnet[1:])
    allDefSnetsContent += "END SPECIALNETS\n"
    return allDefSnetsContent

def generateLefViaFromDef(techLines, cellLines, defLines, lefdbunit, design_name):
    lefViaPattern = """VIA %(viaName)s
    LAYER %(routeL1)s ;
%(r1rect)s
    LAYER %(cutL)s ;
%(viarect)s
    LAYER %(routeL2)s ;
%(r2rect)s
END %(viaName2)s
"""
    allVias = set()
    for lid, line in enumerate(techLines + cellLines):
        if line.startswith("VIA "):
            line = line.strip()
            viaName = line.split(" ")[1].strip()
            allVias.add(viaName)

    isVIAS = False
    vias = []
    curVia = None
    for lid, line in enumerate(defLines):
        if "VIAS" in line and "END" not in line:
            isVIAS = True
            continue
        if not isVIAS:
            continue
        if "END VIAS" in line:
            if curVia is not None:
                vias.append(curVia)
            break
        tmp = line.strip()
        if len(tmp) == 0:
            continue
        if tmp[0] == '-':
            if curVia is not None:
                vias.append(curVia)
            curVia = []
        curVia.append(line)

    print("#Vias: %d" % len(vias))
    allLefViasContent = ""
    for curVia in vias:
        viaName, routeL1, cutL, routeL2 = None, None, None, None
        layerNameSet = set()
        for viaLine in curVia:
            viaLine = viaLine.strip()
            if viaLine[0] == '-':
                viaName = viaLine[1:].strip()
                continue
            layerName = viaLine.split(" ")[2]
            layerNameSet.add(layerName)
        if viaName in allVias:
            # via re-defined
            continue
        for layerName in layerNameSet:
            if routeL1 is None and "metal" in layerName.lower():
                routeL1 = layerName
            elif "via" in layerName.lower():
                cutL = layerName
            else:
                routeL2 = layerName
        layerToRect = {
            routeL1: [],
            cutL: [],
            routeL2: [],
        }
        layerToRectSeg = {
            routeL1: "",
            cutL: "",
            routeL2: "",
        }
        for viaLine in curVia:
            viaLine = viaLine.strip()
            if viaLine[0] == '-':
                continue
            layerName = viaLine.split(" ")[2]
            axis = viaLine.split(layerName)[-1]
            axis = axis.replace(";", "").replace("(", "").replace(")", "").strip()
            rect = [int(i) for i in axis.split(" ") if len(i) > 0]
            layerToRect[layerName].append(rect)
        for layerName, rects in layerToRect.items():
            rectContent = ""
            for idx, rect in enumerate(rects):
                if idx > 0:
                    rectContent += "\n"
                rect = [i / lefdbunit for i in rect]
                rectContent += "        RECT %.4f %.4f %.4f %.4f ;" % (rect[0], rect[1], rect[2], rect[3])
            layerToRectSeg[layerName] = rectContent
        viaName2 = viaName
        if "superblue" in design_name and addDefaultForLefVia:
            # NOTE: fix nanoroute error
            # FIXME: I dont' know which vias in superblue should be set with default...
            viaName += " DEFAULT"
        curlefViaPattern = lefViaPattern % {
            "viaName" : viaName,
            "viaName2" : viaName2,
            "routeL1" : routeL1,
            "routeL2" : routeL2,
            "cutL" : cutL,
            "r1rect" : layerToRectSeg[routeL1],
            "viarect" : layerToRectSeg[cutL],
            "r2rect" : layerToRectSeg[routeL2],
        }
        allLefViasContent += curlefViaPattern + "\n"

    return allLefViasContent

def generateLefNDRFromDef(defLines, lefdbunit):
    isNDRS = False
    ndrs = []
    curNdr = None
    for lid, line in enumerate(defLines):
        if "NONDEFAULTRULES" in line and "END" not in line:
            isNDRS = True
            continue
        if not isNDRS:
            continue
        if "END NONDEFAULTRULES" in line:
            if curNdr is not None:
                ndrs.append(curNdr)
            break
        tmp = line.strip()
        if len(tmp) == 0:
            continue
        if tmp[0] == '-':
            if curNdr is not None:
                ndrs.append(curNdr)
            curNdr = []
        curNdr.append(line)

    print("#NDRs: %d" % len(ndrs))
    allLefNdrsContent = ""
    for curNdr in ndrs:
        hasHardSpacing = False
        layers = []
        useVias = []
        ndrName = None
        for ndrLine in curNdr:
            ndrLine = ndrLine.strip()
            if ndrLine[0] == '-':
                ndrName = ndrLine[1:].strip()
                continue
            if "HARDSPACING" in ndrLine:
                hasHardSpacing = True
                continue
            if "LAYER" in ndrLine:
                ndrLine = ndrLine.replace(";", "")
                layerName = ndrLine.split(" ")[2]
                width = int(ndrLine.split("WIDTH ")[-1].split(" ")[0]) / lefdbunit
                spacing = int(ndrLine.split("SPACING ")[-1].split(" ")[0]) / lefdbunit
                layers.append((layerName, width, spacing))
            if "VIA" in ndrLine:
                ndrLine = ndrLine.replace(";", "")
                viaName = ndrLine.split(" ")[2]
                useVias.append(viaName)
        ndrContent = "NONDEFAULTRULE %s\n" % ndrName
        if hasHardSpacing:
            ndrContent += "  HARDSPACING ;\n"
        for layerName, width, spacing in layers:
            ndrContent += "  LAYER %s\n" % layerName
            ndrContent += "    WIDTH %.4f ;\n" % width
            ndrContent += "    SPACING %.4f ;\n" % spacing
            ndrContent += "  END %s\n" % layerName
        for viaName in useVias:
            ndrContent += "  USEVIA %s ;\n" % viaName
        ndrContent += "END %s\n\n"  % ndrName

        allLefNdrsContent += ndrContent
    return allLefNdrsContent

if __name__ == "__main__":
    raw_root = os.path.join(dataset_root, "ispd2015")
    all_designs = os.listdir(raw_root)
    sorted(all_designs)

    # Raw design clean
    print("-------------------------------------------------")
    output_root = os.path.join(dataset_root, "ispd2015_fix")
    for design_name in all_designs:
        generate_one_raw_design(raw_root, output_root, design_name)