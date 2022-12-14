from utils import *
from src import run_placement_main, Flute

def get_option():
    parser = argparse.ArgumentParser('Xplace')
    # general setting
    parser.add_argument('--dataset_root', type=str, default='/data/ssd/lxliu/design', help='the parent folder of dataset')
    parser.add_argument('--dataset', type=str, default='ispd2005', help='dataset name')
    parser.add_argument('--design_name', type=str, default='adaptec1', help='design name')
    parser.add_argument('--load_from_raw', type=str2bool, default=False, help='If True, parse and load from benchmark files. If False, load from pt')
    parser.add_argument('--run_all', type=str2bool, default=False, help='If True, run all designs in the given dataset. If False, run the given design_name only.')
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--num_threads', type=int, default=20, help='threads')

    # optimization params
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--inner_iter', type=int, default=10000, help='#inner iters')
    parser.add_argument('--wa_coeff', type=float, default=4.0, help='wa coeff')
    parser.add_argument('--num_bin_x', type=int, default=512, help='#binX for density function')
    parser.add_argument('--num_bin_y', type=int, default=512, help='#binY for density function')
    parser.add_argument('--threshold', type=float, default=4.0, help='normalized node area threshold for using naive mode')
    parser.add_argument('--density_weight', type=float, default=8e-5, help='the weight of density loss')
    parser.add_argument('--density_weight_coef', type=float, default=1.05, help='the ratio of density_weight')
    parser.add_argument('--use_init_density_weight', type=str2bool, default=True, help='enable dynamic initialization of density_weight')
    parser.add_argument('--target_density', type=float, default=1.0, help='placement target density')
    parser.add_argument('--use_filler', type=str2bool, default=True, help='placement filler')
    parser.add_argument('--noise_ratio', type=float, default=0.025, help='noise ratio for initialization')
    parser.add_argument('--ignore_net_degree', type=int, default=100, help='threshold of net degree to ignore in wirelength calculation')
    parser.add_argument('--scale_design', type=str2bool, default=True, help='normalize die area')
    parser.add_argument('--use_eplace_nesterov', type=str2bool, default=True, help='enable eplace nesterov optimizer')
    parser.add_argument('--clamp_node', type=str2bool, default=True, help='enable eplace node clamp trick')
    parser.add_argument('--use_precond', type=str2bool, default=True, help='apply precond')
    parser.add_argument('--stop_overflow', type=float, default=0.07, help='stop overflow in scheduler')
    parser.add_argument('--enable_skip_update', type=str2bool, default=True, help='enable skip update')
    parser.add_argument("--loss_type", type=str, default="direct", help="loss type")

    # logging and saver
    parser.add_argument('--log_freq', type=int, default=50) 
    parser.add_argument('--result_dir', type=str, default='result', help='output root directory') 
    parser.add_argument('--exp_id', type=str, default='', help='experiment id') 
    parser.add_argument('--log_dir', type=str, default='log', help='log directory') 
    parser.add_argument('--log_name', type=str, default='test.log', help='log file name') 
    parser.add_argument('--eval_dir', type=str, default='eval', help='visualization directory') 

    # placement related
    parser.add_argument('--draw_placement', type=str2bool, default=False, help='draw placement') 
    parser.add_argument('--write_placement', type=str2bool, default=False, help='write placement result') 
    parser.add_argument('--output_dir', type=str, default="output", help='output directory') 
    parser.add_argument('--output_prefix', type=str, default="placement", help='prefix of placement output file') 
    parser.add_argument('--detail_placement', type=str2bool, default=False, help='perform dp') 
    parser.add_argument('--dp_engine', type=str, default="ntuplace3", help='choose dp engine') 
    args = parser.parse_args()

    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
    if args.dataset == "ispd2015_2":
        args.dataset = "ispd2015_without_fence"

    return args

def main():
    args = get_option()
    logger = setup_logger(args, sys.argv)

    set_random_seed(args)
    Flute.register(args.num_threads)

    run_placement_main(args, logger)
    

if __name__ == "__main__":
    main()