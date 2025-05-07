
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import *
from src import Flute, load_dataset, GPUTimer
from main import get_option

def main():
    Flute.register(8)
    
    # Read input file
    design_name = "example"
    params = {
        "benchmark": "custom",
        "design_name": "test",
        "lef": f"{design_name}/NangateOpenCellLibrary.lef",
        "lib": f"{design_name}/NangateOpenCellLibrary.lib",
        "def": f"{design_name}/example.def",
        "verilog": f"{design_name}/example.v",
        "sdc": f"{design_name}/example.sdc",
        "spef": f"{design_name}/example.spef",
    }

    args = get_option()
    logger = setup_logger(args, sys.argv)
    data, rawdb, gpdb = load_dataset(args, logger, params)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    data = data.to(device)
    data = data.preprocess()

    gputimer = GPUTimer(data, rawdb, gpdb, params, args)
    
    # timing analysis for extracted RC network
    gputimer.timer.read_spef(params["spef"])
    gputimer.update_timing_spef()
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info("SPEf evaluation: wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % (wns_early, tns_early, wns_late, tns_late))
    
    # timing analysis for normalized FLUTE RC tree
    gputimer.update_timing_eval(data.node_pos)
    wns_early, tns_early, wns_late, tns_late = gputimer.report_timing_slack()
    logger.info("Flute Tree Evaluation wns_early: %.3f, tns_early: %.3f, wns_late: %.3f, tns_late: %.3f" % (wns_early, tns_early, wns_late, tns_late))
        
    
# run main
__name__ == "__main__" and main()
