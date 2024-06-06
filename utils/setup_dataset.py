from .get_design_params import get_single_design_params, get_custom_design_params, get_custom_json_params


def find_design_params(args, logger, placement=None):
    if args.custom_path != "":
        params = get_custom_design_params(args)
    elif args.custom_json != "":
        params = get_custom_json_params(args, logger)
    else:
        params = get_single_design_params(
            args.dataset_root, args.dataset, args.design_name, placement
        )
    setup_given_solution(args, logger, params, placement)
    log_design_params(logger, params)
    setup_design_args(args)
    return params


def setup_given_solution(args, logger, params, placement=None):
    if placement is not None:
        args.given_solution = placement
    if args.given_solution != "":
        placement = args.given_solution
        logger.info("Find given placement solution: %s" % placement)
        if ".pl" in placement:
            if "pl" in params.keys():
                logger.info("Overwrite pl file %s by %s" % (params["pl"], placement))
            params["pl"] = placement
        elif ".def" in placement:
            if "def" in params.keys():
                logger.info("Overwrite def file %s by %s" % (params["def"], placement))
            params["def"] = placement


def log_design_params(logger, params: dict):
    content = "Design Info:\n"
    num_items = 0
    if "benchmark" in params.keys():
        content += f"benchmark: {params['benchmark']}\n"
        num_items += 1
    if "design_name" in params.keys():
        content += f"design_name: {params['design_name']}\n"
        num_items += 1
    for key, value in params.items():
        if key == "design_name" or key == "benchmark":
            continue
        content += f"{key}: {value}"
        if num_items < len(params) - 1:
            content += "\n"
        num_items += 1
    logger.info(content)


def setup_design_args(args):
    if args.design_name in ["adaptec1", "bigblue1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 1.0
    elif args.design_name in ["adaptec2", "adaptec3", "adaptec4", "bigblue2"]:
        args.num_bin_x = args.num_bin_y = 1024
        args.target_density = 1.0
    elif args.design_name in ["bigblue3", "bigblue4"]:
        args.num_bin_x = args.num_bin_y = 2048
        args.target_density = 1.0
    elif args.design_name in ["adaptec5"]:
        args.target_density = 0.5
        args.num_bin_x = args.num_bin_y = 1024
    elif args.design_name in ["newblue1"]:
        args.target_density = 0.8
        args.num_bin_x = args.num_bin_y = 512
    elif args.design_name in ["newblue2"]:
        args.target_density = 0.9
        args.num_bin_x = args.num_bin_y = 1024
    elif args.design_name in ["newblue3"]:
        args.target_density = 0.8
        args.num_bin_x = args.num_bin_y = 2048
    elif args.design_name in ["newblue4"]:
        args.target_density = 0.5
        args.num_bin_x = args.num_bin_y = 1024
    elif args.design_name in ["newblue5"]:
        args.target_density = 0.5
        args.num_bin_x = args.num_bin_y = 1024
    elif args.design_name in ["newblue6"]:
        args.target_density = 0.8
        args.num_bin_x = args.num_bin_y = 2048
    elif args.design_name in ["newblue7"]:
        args.target_density = 0.8
        args.num_bin_x = args.num_bin_y = 2048
    elif args.design_name in ["mgc_des_perf_1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.91
    elif args.design_name in ["mgc_fft_1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.835
    elif args.design_name in ["mgc_fft_2"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.65
    elif args.design_name in ["mgc_fft_a"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.5
    elif args.design_name in ["mgc_fft_b"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.6
    elif args.design_name in ["mgc_matrix_mult_1"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.802
    elif args.design_name in ["mgc_matrix_mult_2"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.8
    elif args.design_name in ["mgc_matrix_mult_a"]:
        args.num_bin_x = args.num_bin_y = 512
        args.target_density = 0.6
    elif args.design_name in ["mgc_superblue12"]:
        args.num_bin_x, args.num_bin_y = 1024, 1024
        args.target_density = 0.65
    elif args.design_name in ["mgc_superblue14"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.56
    elif args.design_name in ["mgc_superblue19"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.53
    elif args.design_name in ["mgc_des_perf_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.429
    elif args.design_name in ["mgc_des_perf_b"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.497
    elif args.design_name in ["mgc_edit_dist_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.455
    elif args.design_name in ["mgc_matrix_mult_b"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.60
    elif args.design_name in ["mgc_matrix_mult_c"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.60
    elif args.design_name in ["mgc_pci_bridge32_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.384
    elif args.design_name in ["mgc_pci_bridge32_b"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.143
    elif args.design_name in ["mgc_superblue11_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.65
    elif args.design_name in ["mgc_superblue16_a"]:
        args.num_bin_x, args.num_bin_y = 512, 512
        args.target_density = 0.55
    return args