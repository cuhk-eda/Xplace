from utils import *
from src import run_placement_main_nesterov, run_placement_main_adam


def run_placement_single(args, logger):
    logger.info("=================")
    logger.info("Start place %s/%s" % (args.dataset , args.design_name))
    set_random_seed(args)
    setup_dataset_args(args)
    if args.use_eplace_nesterov:
        res = run_placement_main_nesterov(args, logger)
    else:
        raise NotImplementedError("Xplace-NN cannot run with Adam currently.")
        res = run_placement_main_adam(args, logger)
    return res


def run_placement_all(args, logger):
    logger.info("Run all designs in dataset %s." % args.dataset)
    df = pd.DataFrame(columns=["design", "dp_hpwl", "gp_hpwl", "top5overflow", "overflow", "gp_time", "dp_time", "gp_per_iter"])
    mul_params = sorted(
        get_multiple_design_params(args.dataset_root, args.dataset), key=lambda params: params["design_name"]
    )
    for i, params in enumerate(mul_params):
        cur_args = copy.deepcopy(args)
        cur_args.design_name = params["design_name"]
        result = run_placement_single(cur_args, logger)
        df.loc[i] = [cur_args.design_name, *result]
    csv_path = os.path.join(args.result_dir, args.exp_id, args.log_dir, "run_all.csv")
    df.to_csv(csv_path)
    print(df)


def run_placement_main(args, logger):
    if args.run_all:
        run_placement_all(args, logger)
    else:
        run_placement_single(args, logger)