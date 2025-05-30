import os


def find_benchmark(dataset_root, benchmark):
    bm_to_root = {
        "ispd2005": os.path.join(dataset_root, "ispd2005"),
        "ispd2006": os.path.join(dataset_root, "ispd2006"),
        "mms": os.path.join(dataset_root, "mms"),
        "dac2012": os.path.join(dataset_root, "iccad2012dac2012"),
        "ispd2015": os.path.join(dataset_root, "ispd2015"),
        "ispd2015_fix": os.path.join(dataset_root, "ispd2015_fix"),
        "ispd2018": os.path.join(dataset_root, "ispd2018"),
        "ispd2019_no_fence": os.path.join(dataset_root, "ispd2019_no_fence"),
        "iccad2019": os.path.join(dataset_root, "iccad2019"),
        "ispd2018": os.path.join(dataset_root, "ispd2018"),
        "iccad2015": os.path.join(dataset_root, "iccad2015"),
        "iccad2015.ot": os.path.join(dataset_root, "iccad2015.ot"),
    }
    root = bm_to_root[benchmark]
    all_designs = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    return root, all_designs


def get_single_design_params(dataset_root, benchmark, design_name, placement=None):
    if benchmark in ["ispd2005", "ispd2006", "mms"]:
        return single_ispd2005(dataset_root, design_name, benchmark, placement)
    elif benchmark == "dac2012":
        return single_dac2012(dataset_root, design_name, placement)
    elif benchmark == "ispd2015":
        return single_ispd2015(dataset_root, design_name, placement)
    elif benchmark == "ispd2015_fix":
        return single_ispd2015_fix(dataset_root, design_name, placement)
    elif benchmark == "ispd2018":
        return single_ispd2018(dataset_root, design_name, placement)
    elif benchmark == "ispd2019_no_fence":
        return single_ispd2019_no_fence(dataset_root, design_name, placement)
    elif benchmark == "iccad2019":
        return single_iccad2019(dataset_root, design_name, placement)
    elif benchmark == "ispd2018":
        return single_ispd2018(dataset_root, design_name, placement)
    elif benchmark.startswith("iccad2015"):
        return single_iccad2015(dataset_root, benchmark, design_name, placement)
    else:
        raise NotImplementedError("benchmark %s is not found" % benchmark)


def get_multiple_design_params(dataset_root, benchmark):
    root, all_designs = find_benchmark(dataset_root, benchmark)
    params_mul = []
    for design_name in all_designs:
        params = get_single_design_params(dataset_root, benchmark, design_name)
        params_mul.append(params)
    return params_mul


def single_ispd2005(dataset_root, design_name, benchmark, placement=None):
    benchmark = benchmark
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "bookshelf_variety": "ispd2005",
        "aux": "%s/%s/%s.aux" % (root, design_name, design_name),
        "design_name": design_name,
    }
    if placement is not None:
        params["pl"] = placement
    return params


def single_dac2012(dataset_root, design_name, placement=None):
    benchmark = "dac2012"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "bookshelf_variety": "dac2012",
        "aux": "%s/%s/%s.aux" % (root, design_name, design_name),
        "design_name": design_name,
    }
    if placement is not None:
        params["pl"] = placement
    return params


def single_ispd2015(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "ispd2015"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "tech_lef": "%s/%s/tech.lef" % (root, design_name),
        "cell_lef": "%s/%s/cells.lef" % (root, design_name),
        "def": "%s/%s/floorplan.def" % (root, design_name) if placement is None else placement,
        "design_name": design_name,
    }
    return params


def single_ispd2015_fix(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "ispd2015_fix"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "lef": "%s/%s/%s.lef" % (root, design_name, design_name),
        "def": "%s/%s/%s.def" % (root, design_name, design_name) if placement is None else placement,
        "design_name": design_name,
    }
    return params


def single_ispd2018(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "ispd2018"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "lef": "%s/%s/%s.input.lef" % (root, design_name, design_name),
        "def": "%s/%s/%s.input.def" % (root, design_name, design_name) if placement is None else placement,
        "design_name": design_name,
    }
    return params


def single_ispd2019_no_fence(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "ispd2019_no_fence"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "lef": "%s/%s/%s.input.lef" % (root, design_name, design_name),
        "def": "%s/%s/%s.input.def" % (root, design_name, design_name) if placement is None else placement,
        "design_name": design_name,
    }
    return params


def single_iccad2019(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "iccad2019"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "lef": "%s/%s/%s.input.lef" % (root, design_name, design_name),
        "def": "%s/%s/%s.input.def" % (root, design_name, design_name) if placement is None else placement,
        "design_name": design_name,
    }
    return params


def single_ispd2018(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "ispd2018"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "lef": "%s/%s/%s.input.lef" % (root, design_name, design_name),
        "def": "%s/%s/%s.input.def" % (root, design_name, design_name)
        if placement is None
        else placement,
        "design_name": design_name,
    }
    return params


def single_iccad2015(dataset_root, benchmark, design_name, placement=None):
    # configuration
    # benchmark = "iccad2015"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "tech_lef": "%s/tech.lef" % (root),
        "cell_lef": "%s/%s/%s.lef" % (root, design_name, design_name),
        "def": "%s/%s/%s.def" % (root, design_name, design_name) if placement is None else placement,
        "verilog": "%s/%s/%s.v" % (root, design_name, design_name),
        "early_lib": "%s/%s/%s_Early.lib" % (root, design_name, design_name),
        "late_lib": "%s/%s/%s_Late.lib" % (root, design_name, design_name),
        "sdc": "%s/%s/%s.sdc" % (root, design_name, design_name),
        "design_name": design_name,
    }
    return params


def get_custom_design_params(args):
    params = dict(
        [
            [item.strip() for item in token.strip().split(":")]
            for token in args.custom_path.split(",")
            if len(token) > 0
        ]
    )
    if "benchmark" not in params.keys():
        raise ValueError("Cannot find 'benchmark' in args.custom_path")
    if "design_name" not in params.keys():
        raise ValueError("Cannot find 'design_name' in args.custom_path")
    args.dataset = params["benchmark"]
    args.design_name = params["design_name"]
    return params


def get_custom_json_params(args, logger):
    import json
    arg_dict = vars(args)
    with open(args.custom_json, 'r') as f:
        params = json.load(f)
    if "benchmark" not in params.keys():
        raise ValueError("Cannot find 'benchmark' in args.custom_path")
    if "design_name" not in params.keys():
        raise ValueError("Cannot find 'design_name' in args.custom_path")
    args.dataset = params["benchmark"]
    args.design_name = params["design_name"]
    if "lefs" in params.keys():
        logger.info("Detect json LEF/DEF mode. Please make sure that tech_lef are included first.")
        lefs = params["lefs"]
        for i in range(len(lefs)):
            # Simple heuristic to find tech_lef (Only for ASAP7, Nangate45, Sky130, GF180)
            if i == 0:
                continue
            if "tech" in lefs[i] or ".tlef" in lefs[i]:
                lefs[i], lefs[0] = lefs[0], lefs[i]
                break
        params["lefs"] = lefs
    if "output_path" in params.keys():
        args.output_path = params["output_path"]
    for key in params.keys():
        if key in arg_dict.keys():
            arg_dict[key] = params[key]
    return params