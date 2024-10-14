from utils import *
from src import run_placement_main_nesterov

def run_placement_main(args, logger):
    logger.info("=================")
    assert torch.cuda.is_available()
    torch.cuda.synchronize("cuda:{}".format(args.gpu))
    set_random_seed(args)
    res = run_placement_main_nesterov(args, logger)
    return res
