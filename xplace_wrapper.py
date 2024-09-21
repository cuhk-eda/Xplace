from utils import *
from src import run_placement_main, Flute

def set_option():
    args = type('ARGS', (), {})()
    args.bookshelf_variety = 'ispd2005'
    args.aux = '../../testcase/mms/adaptec1/adaptec1.aux'
    args.dataset = "mms"
    args.design_name = "adaptec1"
    args.custom_path = ""
    args.custom_json = ""
    args.given_solution = ""
    args.load_from_raw = True
    args.run_all = False
    args.seed = 0
    args.gpu = 0
    args.num_threads = 16
    args.deterministic = True
    args.global_placement = True
    args.lr = 0.01
    args.inner_iter = 10000
    args.wa_coeff = 4.0
    args.num_bin_x = 512
    args.num_bin_y = 512
    args.density_weight = 8e-05
    args.density_weight_coef = 1.05
    args.target_density = 1.0
    args.use_filler = True
    args.noise_ratio = 0.025
    args.ignore_net_degree = 100
    args.use_eplace_nesterov = True
    args.clamp_node = True
    args.use_precond = True
    args.stop_overflow = 0.07
    args.enable_skip_update = True
    args.enable_sample_force = True
    args.mixed_size = True
    args.use_cell_inflate = False
    args.min_area_inc = 0.01
    args.use_route_force = False
    args.route_freq = 1000
    args.num_route_iter = 400
    args.route_weight = 0
    args.congest_weight = 0
    args.pseudo_weight = 0
    args.visualize_cgmap = False
    args.legalization = True
    args.detail_placement = True
    args.dp_engine = "default"
    args.eval_by_external = False
    args.eval_engine = "ntuplace4dr"
    args.final_route_eval = False
    args.log_freq = 100
    args.verbose_cpp_log = False
    args.cpp_log_level = 2
    args.result_dir = "result"
    args.exp_id = ""
    args.log_dir = "log"
    args.log_name = "log.txt"
    args.eval_dir = "eval"
    args.draw_placement = True
    args.write_placement = True
    args.write_global_placement = False
    args.output_dir = "output"
    args.output_prefix = "placement"
    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + args.exp_id
    return args

def run_xplace(aux_filepath, design_name):
    args = set_option()
    args.aux = aux_filepath
    args.design_name = design_name
    logger = setup_logger(args, sys.argv)
    set_random_seed(args)
    Flute.register(args.num_threads)
    run_placement_main(args, logger)

run_xplace('../../testcase/mms/adaptec1/adaptec1.aux', 'adaptec1')
