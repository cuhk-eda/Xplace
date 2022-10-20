import os


def find_benchmark(dataset_root, benchmark):
    bm_to_root = {
        "ispd2005": os.path.join(dataset_root, "ispd2005"),
        "ispd2015": os.path.join(dataset_root, "ispd2015"),
        "ispd2015_without_fence": os.path.join(dataset_root, "ispd2015_without_fence"),
    }
    root = bm_to_root[benchmark]
    all_designs = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    return root, all_designs


def get_single_design_params(dataset_root, benchmark, design_name, placement=None):
    if benchmark == "ispd2005":
        return single_ispd2005(dataset_root, design_name, placement)
    elif benchmark == "ispd2015":
        return single_ispd2015(dataset_root, design_name, placement)
    elif benchmark == "ispd2015_without_fence":
        return single_ispd2015_without_fence(dataset_root, design_name, placement)
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
