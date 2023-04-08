import os


def find_benchmark(dataset_root, benchmark):
    bm_to_root = {
        "ispd2005": os.path.join(dataset_root, "ispd2005"),
        "dac2012": os.path.join(dataset_root, "iccad2012dac2012"),
        "ispd2015": os.path.join(dataset_root, "ispd2015"),
        "ispd2015_without_fence": os.path.join(dataset_root, "ispd2015_without_fence"),
        "ispd2015_fix": os.path.join(dataset_root, "ispd2015_fix"),
        "iccad2019": os.path.join(dataset_root, "iccad2019"),
    }
    root = bm_to_root[benchmark]
    all_designs = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    return root, all_designs


def get_single_design_params(dataset_root, benchmark, design_name, placement=None):
    if benchmark == "ispd2005":
        return single_ispd2005(dataset_root, design_name, placement)
    elif benchmark == "dac2012":
        return single_dac2012(dataset_root, design_name, placement)
    elif benchmark == "ispd2015":
        return single_ispd2015(dataset_root, design_name, placement)
    elif benchmark == "ispd2015_without_fence":
        return single_ispd2015_without_fence(dataset_root, design_name, placement)
    elif benchmark == "ispd2015_fix":
        return single_ispd2015_fix(dataset_root, design_name, placement)
    elif benchmark == "iccad2019":
        return single_iccad2019(dataset_root, design_name, placement)
    else:
        raise NotImplementedError("benchmark %s is not found" % benchmark)


def get_multiple_design_params(dataset_root, benchmark):
    root, all_designs = find_benchmark(dataset_root, benchmark)
    params_mul = []
    for design_name in all_designs:
        params = get_single_design_params(dataset_root, benchmark, design_name)
        params_mul.append(params)
    return params_mul


def single_ispd2005(dataset_root, design_name, placement=None):
    benchmark = "ispd2005"
    root, all_designs = find_benchmark(dataset_root, benchmark)
    if design_name not in all_designs:
        raise ValueError("Design Name %s should in %s" % (design_name, all_designs))
    params = {
        "benchmark": benchmark,
        "bookshelf_variety": "ispd2005",
        "aux": "%s/%s/%s.aux" % (root, design_name, design_name),
        "pl": "%s/%s/%s.pl" % (root, design_name, design_name) if placement is None else placement,
        "design_name": design_name,
    }
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
        "pl": "%s/%s/%s.pl" % (root, design_name, design_name) if placement is None else placement,
        "design_name": design_name,
    }
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


def single_ispd2015_without_fence(dataset_root, design_name, placement=None):
    # configuration
    benchmark = "ispd2015_without_fence"
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


def get_custom_design_params(args):
    params = dict([
        [item.strip() for item in token.strip().split(":")] 
        for token in args.custom_path.split(",") if len(token) > 0
    ])
    if "benchmark" not in params.keys():
        raise ValueError("Cannot find 'benchmark' in args.custom_path")
    if "design_name" not in params.keys():
        raise ValueError("Cannot find 'design_name' in args.custom_path")
    args.dataset = params["benchmark"]
    args.design_name = params["design_name"]
    return params