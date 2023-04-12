from utils import *
from src import run_placement_main_nesterov


def run_placement_single(args, logger):
    logger.info("=================")
    logger.info("Start place %s/%s" % (args.dataset , args.design_name))
    assert torch.cuda.is_available()
    torch.cuda.synchronize("cuda:{}".format(args.gpu))
    set_random_seed(args)
    setup_dataset_args(args)
    res = run_placement_main_nesterov(args, logger)
    return res


def run_placement_all(args, logger):
    logger.info("Run all designs in dataset %s." % args.dataset)
    place_df = pd.DataFrame(columns=["design", "dp_hpwl", "gp_hpwl", "top5overflow", "overflow", "gp_time", "lg+dp_time", "gp_per_iter", "place_time"])
    route_df = pd.DataFrame(columns=["design", "#OvflNets", "GR WL", "GR #Vias", "GR EstShort", "RC Hor", "RC Ver"])
    mul_params = sorted(
        get_multiple_design_params(args.dataset_root, args.dataset), key=lambda params: params["design_name"]
    )
    for i, params in enumerate(mul_params):
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        place_metrics, route_metrics = run_placement_single(cur_args, logger)
        place_df.loc[i] = [cur_args.design_name, *place_metrics]
        if route_metrics is not None:
            route_df.loc[i] = [cur_args.design_name, *route_metrics]
    place_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "run_all.csv")
    place_df.to_csv(place_csv_path)
    print(place_df)
    if len(route_df) > 0:
        route_csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "route.csv")
        route_df.to_csv(route_csv_path)
        print(route_df)


def run_placement_main(args, logger):
    if args.run_all:
        run_placement_all(args, logger)
    else:
        run_placement_single(args, logger)