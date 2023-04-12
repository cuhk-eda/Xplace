from utils import *
from src import *
from FNO import train_FNO 
import time


def get_option():
    parser = argparse.ArgumentParser('FON')
    # general setting
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--macro_mode', type=str, default="bench", help='macro data mode')
    parser.add_argument('--deterministic', type=str2bool, default=True, help='use deterministic mode')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=60, help='epoches')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--inner_iter', type=int, default=40, help='per episode length in data generator')
    parser.add_argument('--ntrain', type=int, default=8000, help='#samples per epoch')
    parser.add_argument('--step_size', type=int, default=50, help='lr decay step size')
    parser.add_argument('--gamma', type=float, default=0.25, help='lr decay coeff')
    
    # FON parameters
    parser.add_argument('--num_x', type=int, default=256, help='bin x')
    parser.add_argument('--num_y', type=int, default=256, help='bin y')
    parser.add_argument('--width', type=int, default=12, help='channel width')
    parser.add_argument('--neck', type=int, default=24, help='chennel neck')
    parser.add_argument('--modes', type=int, default=128, help='frequency modes')
    parser.add_argument('--scaler', type=float, default=1, help='scaler of training data')

    # logging and saver
    parser.add_argument('--result_dir', type=str, default='result_fno', help='log/model root directory') 
    parser.add_argument('--exp_id', type=str, default='', help='experiment id') 
    parser.add_argument('--log_dir', type=str, default='log', help='log directory') 
    parser.add_argument('--log_name', type=str, default='test.log', help='log file name') 
    parser.add_argument('--model_dir', type=str, default='model', help='visualization directory') 

    args = parser.parse_args()
    return args


def main():
    args = get_option()

    args.result_dir = "./FNO/%s" % (args.result_dir)
    args.exp_id = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + \
                   "_%sx%sx%s" % (args.width, args.neck, args.modes) + args.exp_id

    logger = setup_logger(args, sys.argv)
    set_random_seed(args)

    t1 = time.time()
    train_FNO(args, logger)
    t2 = time.time()

    logger.info('-----------------------------------------')
    logger.info(t2-t1)


if __name__ == "__main__":
    main()