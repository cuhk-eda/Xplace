from utils import *
from src import run_placement_main, Flute

def get_option():
    parser = argparse.ArgumentParser('Xplace')
    # general setting
    parser.add_argument('--dataset_root', type=str, default='data/raw', help='the parent folder of dataset')
    parser.add_argument('--dataset', type=str, default='ispd2015_fix', help='dataset name')
    parser.add_argument('--design_name', type=str, default='mgc_superblue12', help='design name')
    parser.add_argument('--custom_path', type=str, default='', help='custom design path, set it as token1:path1,token2:path2 e.g. lef:data/test.lef,def:data/test.def,design_name:mydesign,benchmark:mybenchmark')
    parser.add_argument('--custom_json', type=str, default='', help='custom json path, support multi-lefs.')
    parser.add_argument('--given_solution', type=str, default='', help='Given placement solution. Will overwrite other given .pl and .def.')
    parser.add_argument('--load_from_raw', type=str2bool, default=True, help='If True, parse and load from benchmark files. If False, load from pt')
    parser.add_argument('--run_all', type=str2bool, default=False, help='If True, run all designs in the given dataset. If False, run the given design_name only.')
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--num_threads', type=int, default=20, help='threads')
    parser.add_argument('--deterministic', type=str2bool, default=True, help='use deterministic mode')

    # global placement params
    parser.add_argument('--global_placement', type=str2bool, default=True, help='perform gp') 
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--inner_iter', type=int, default=10000, help='#inner iters')
    parser.add_argument('--wa_coeff', type=float, default=4.0, help='wa coeff')
    parser.add_argument('--num_bin_x', type=int, default=512, help='#binX for density function')
    parser.add_argument('--num_bin_y', type=int, default=512, help='#binY for density function')
    parser.add_argument('--density_weight', type=float, default=8e-5, help='the weight of density loss')
    parser.add_argument('--density_weight_coef', type=float, default=1.05, help='the ratio of density_weight')
    parser.add_argument('--target_density', type=float, default=1.0, help='placement target density')
    parser.add_argument('--use_filler', type=str2bool, default=True, help='placement filler')
    parser.add_argument('--noise_ratio', type=float, default=0.025, help='noise ratio for initialization')
    parser.add_argument('--ignore_net_degree', type=int, default=100, help='threshold of net degree to ignore in wirelength calculation')
    parser.add_argument('--use_eplace_nesterov', type=str2bool, default=True, help='enable eplace nesterov optimizer')
    parser.add_argument('--clamp_node', type=str2bool, default=True, help='enable eplace node clamp trick')
    parser.add_argument('--use_precond', type=str2bool, default=True, help='apply precond')
    parser.add_argument('--stop_overflow', type=float, default=0.07, help='stop overflow in scheduler')
    parser.add_argument('--enable_skip_update', type=str2bool, default=True, help='enable skip update')
    parser.add_argument('--enable_sample_force', type=str2bool, default=True, help='enable sample force')
    parser.add_argument("--mixed_size", type=str2bool, default=False, help="enable mixed size placement")

    # global routing params
    parser.add_argument('--use_cell_inflate', type=str2bool, default=False, help='use cell inflation')
    parser.add_argument('--min_area_inc', type=float, default=0.01, help='threshold of terminating inflation')
    parser.add_argument('--use_route_force', type=str2bool, default=False, help='use routing force')
    parser.add_argument('--route_freq', type=int, default=1000, help='routing freq')
    parser.add_argument('--num_route_iter', type=int, default=400, help='number of routing iters')
    parser.add_argument('--route_weight', type=float, default=0, help='the weight of route')
    parser.add_argument('--congest_weight', type=float, default=0, help='the weight of congested force')
    parser.add_argument('--pseudo_weight', type=float, default=0, help='the weight of pseudo net')
    parser.add_argument('--visualize_cgmap', type=str2bool, default=False, help='visualize congestion map')

    # timing opt params
    parser.add_argument('--timing_opt', type=str2bool, default=False, help='perform timing optimization') 
    parser.add_argument('--timing_freq', type=int, default=1, help='timing freq')
    parser.add_argument('--calibration', type=str2bool, default=True, help='perform timer calibration') 
    parser.add_argument('--calibration_step', type=float, default=0.1, help='timing calibration step')
    parser.add_argument('--timing_start_iter', type=int, default=100, help='start iteration of timing optimization')
    parser.add_argument('--timing_init_weight', type=float, default=0.05, help='initial timing wirelength weight')
    parser.add_argument('--decay_factor', type=float, default=0.3, help='decay factor of timing weight')
    parser.add_argument('--decay_boost', type=float, default=3, help='dynamic decay boost factor')
    parser.add_argument('--wire_resistance_per_micron', type=float, default=2.535, help='unit wire resistance, normalized across all layers')
    parser.add_argument('--wire_capacitance_per_micron', type=float, default=0.16e-15, help='unit wire capacitance, normalized across all layers')

    # detailed placement and evaluation
    parser.add_argument('--legalization', type=str2bool, default=True, help='perform lg') 
    parser.add_argument('--detail_placement', type=str2bool, default=True, help='perform dp') 
    parser.add_argument('--dp_engine', type=str, default="default", help='choose dp engine') 
    parser.add_argument('--eval_by_external', type=str2bool, default=False, help='eval dp sol by external binary') 
    parser.add_argument('--eval_engine', type=str, default="ntuplace4dr", help='choose eval engine') 
    parser.add_argument('--final_route_eval', type=str2bool, default=False, help='eval placement solution by GR')

    # logging and saver
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--verbose_cpp_log', type=str2bool, default=False, help='verbose cpp log for debugging')
    parser.add_argument('--cpp_log_level', type=int, default=2, help='0: DEBUG, 1: VERBOSE, 2:INFO')
    parser.add_argument('--result_dir', type=str, default='result', help='log/model root directory') 
    parser.add_argument('--exp_id', type=str, default='', help='experiment id') 
    parser.add_argument('--log_dir', type=str, default='log', help='log directory') 
    parser.add_argument('--log_name', type=str, default='test.log', help='log file name') 
    parser.add_argument('--eval_dir', type=str, default='eval', help='visualization directory') 

    # placement output
    parser.add_argument('--draw_placement', type=str2bool, default=False, help='draw placement') 
    parser.add_argument('--write_placement', type=str2bool, default=True, help='write placement result') 
    parser.add_argument('--write_global_placement', type=str2bool, default=False, help='write global placement result') 
    parser.add_argument('--output_dir', type=str, default="output", help='output directory') 
    parser.add_argument('--output_prefix', type=str, default="placement", help='prefix of placement output file') 

    args = parser.parse_args()

    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
    args.exp_id = "{}_{}".format(args.exp_id, args.design_name)

    if args.dataset == "ispd2015":
        print("We haven't yet support fence region in ispd2015, use ispd2015_fix instead")
        args.dataset = "ispd2015_fix"

    if args.dataset == "ispd2019":
        print("We haven't yet support fence region in ispd2019, use ispd2019_no_fence instead")
        args.dataset = "ispd2019_no_fence"

    if args.custom_path != "":
        get_custom_design_params(args)

    return args

def main():
    args = get_option()
    logger = setup_logger(args, sys.argv)

    set_random_seed(args)
    Flute.register(args.num_threads)

    run_placement_main(args, logger)
    

if __name__ == "__main__":
    main()
