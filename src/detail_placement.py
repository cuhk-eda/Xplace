import torch
from .database import PlaceData
from .evaluator import get_obj_hpwl
from utils.visualization import draw_fig_with_cairo_cpp
from cpp_to_py import gpudp, routedp
import numba as nb
import numpy as np
import os
import time


class PreprocessDatabaseCache:
    def __init__(self) -> None:
        self.node_size = None
        self.node_weight = None
        self.pin_id2node_id = None
        self.node2pin_list = None
        self.node2pin_list_end = None

    def reset(self):
        self.node_size = None
        self.node_weight = None
        self.pin_id2node_id = None
        self.node2pin_list = None
        self.node2pin_list_end = None


preprocess_db_cache = PreprocessDatabaseCache()


@nb.njit(cache=True)
def rearrange_ndarray(pin_id2node_id, info, info2):
    fix_rhs, iopin_rhs, blkg_rhs, floatiopin_rhs, floatfix_rhs = info
    num_iopin, num_blkg, num_floatiopin, num_floatfix = info2
    for i in range(len(pin_id2node_id)):
        node_id = pin_id2node_id[i]
        if node_id < fix_rhs:
            continue
        elif node_id >= fix_rhs and node_id < iopin_rhs:
            # iopin
            pin_id2node_id[i] = node_id + num_blkg + num_floatfix
        elif node_id >= iopin_rhs and node_id < blkg_rhs:
            # blkg
            pin_id2node_id[i] = node_id - num_iopin
        elif node_id >= blkg_rhs and node_id < floatiopin_rhs:
            # floatiopin
            pin_id2node_id[i] = node_id + num_floatfix
        elif node_id >= floatiopin_rhs and node_id < floatfix_rhs:
            # floatfix
            pin_id2node_id[i] = node_id - num_iopin - num_floatiopin
        else:
            print("Rearrange Error!")
            pin_id2node_id[i] = -1
    return pin_id2node_id


def rearrange_dpdb_node_info(node_pos: torch.Tensor, data: PlaceData):
    # old: mov, float mov, fix, iopin,      blkg, float iopin, float fix
    # new: mov, float mov, fix,  blkg, float fix,       iopin, float iopin
    _, floatmov_rhs, _ = data.node_type_indices[1]
    _, fix_rhs, _ = data.node_type_indices[2]
    _, iopin_rhs, _ = data.node_type_indices[3]
    _, blkg_rhs, _ = data.node_type_indices[4]
    _, floatiopin_rhs, _ = data.node_type_indices[5]
    _, floatfix_rhs, _ = data.node_type_indices[6]

    node_lpos = torch.cat((
        node_pos[:floatmov_rhs].detach() - data.node_size[:floatmov_rhs] / 2,
        data.node_lpos[floatmov_rhs:fix_rhs],
        data.node_lpos[iopin_rhs:blkg_rhs],
        data.node_lpos[floatiopin_rhs:floatfix_rhs],
        data.node_lpos[fix_rhs:iopin_rhs],
        data.node_lpos[blkg_rhs:floatiopin_rhs]
    ), dim=0)

    if preprocess_db_cache.node_size is not None:
        node_size = preprocess_db_cache.node_size.clone()
        node_weight = preprocess_db_cache.node_weight
        pin_id2node_id = preprocess_db_cache.pin_id2node_id
        node2pin_list = preprocess_db_cache.node2pin_list
        node2pin_list_end = preprocess_db_cache.node2pin_list_end
        return node_lpos, node_size, node_weight, pin_id2node_id, node2pin_list, node2pin_list_end

    pin_id2node_id: torch.Tensor = data.pin_id2node_id.clone().int().cpu().numpy()

    node_size = torch.cat((
        data.node_size[:fix_rhs],
        data.node_size[iopin_rhs:blkg_rhs],
        data.node_size[floatiopin_rhs:floatfix_rhs],
        data.node_size[fix_rhs:iopin_rhs],
        data.node_size[blkg_rhs:floatiopin_rhs]
    ), dim=0)

    node_weight_ori = data.node_to_num_pins.squeeze(1)
    node_weight = torch.cat((
        node_weight_ori[:fix_rhs],
        node_weight_ori[iopin_rhs:blkg_rhs],
        node_weight_ori[floatiopin_rhs:floatfix_rhs],
        node_weight_ori[fix_rhs:iopin_rhs],
        node_weight_ori[blkg_rhs:floatiopin_rhs]
    ), dim=0)

    old_node2pin_list_end: torch.Tensor = data.node2pin_list_end.int()
    old_node2pin_list: torch.Tensor = data.node2pin_list.int()

    num_pin_in_iopin = old_node2pin_list_end[iopin_rhs - 1] - old_node2pin_list_end[fix_rhs - 1]
    num_pin_in_blkg = old_node2pin_list_end[blkg_rhs - 1] - old_node2pin_list_end[iopin_rhs - 1]
    num_pin_in_floatiopin = old_node2pin_list_end[floatiopin_rhs - 1] - old_node2pin_list_end[blkg_rhs - 1]
    num_pin_in_floatfix = old_node2pin_list_end[floatfix_rhs - 1] - old_node2pin_list_end[floatiopin_rhs - 1]
    
    node2pin_list = torch.cat((
        old_node2pin_list[:old_node2pin_list_end[fix_rhs-1]],
        old_node2pin_list[old_node2pin_list_end[iopin_rhs - 1]:old_node2pin_list_end[blkg_rhs - 1]],
        old_node2pin_list[old_node2pin_list_end[floatiopin_rhs - 1]:old_node2pin_list_end[floatfix_rhs - 1]],
        old_node2pin_list[old_node2pin_list_end[fix_rhs - 1]:old_node2pin_list_end[iopin_rhs - 1]],
        old_node2pin_list[old_node2pin_list_end[blkg_rhs - 1]:old_node2pin_list_end[floatiopin_rhs - 1]]
    ), dim=0)

    node2pin_list_end = torch.cat((
        old_node2pin_list_end[:fix_rhs],
        old_node2pin_list_end[iopin_rhs:blkg_rhs] - num_pin_in_iopin,
        old_node2pin_list_end[floatiopin_rhs:floatfix_rhs] - num_pin_in_iopin - num_pin_in_floatiopin,
        old_node2pin_list_end[fix_rhs:iopin_rhs] + num_pin_in_blkg + num_pin_in_floatfix,
        old_node2pin_list_end[blkg_rhs:floatiopin_rhs] + num_pin_in_floatfix
    ), dim=0)

    num_iopin = iopin_rhs - fix_rhs
    num_blkg = blkg_rhs - iopin_rhs
    num_floatiopin = floatiopin_rhs - blkg_rhs
    num_floatfix = floatfix_rhs - floatiopin_rhs

    info = (fix_rhs, iopin_rhs, blkg_rhs, floatiopin_rhs, floatfix_rhs)
    info2 = (num_iopin, num_blkg, num_floatiopin, num_floatfix)
    pin_id2node_id = rearrange_ndarray(pin_id2node_id, info, info2)
    pin_id2node_id = torch.from_numpy(pin_id2node_id).to(node_lpos.device)

    if preprocess_db_cache.node_size is None:
        preprocess_db_cache.node_size = node_size
        preprocess_db_cache.node_weight = node_weight
        preprocess_db_cache.pin_id2node_id = pin_id2node_id
        preprocess_db_cache.node2pin_list = node2pin_list
        preprocess_db_cache.node2pin_list_end = node2pin_list_end

    return node_lpos, node_size, node_weight, pin_id2node_id, node2pin_list, node2pin_list_end


def setup_detailed_rawdb(
    node_pos: torch.Tensor, use_cpu_db_: bool, data: PlaceData, args, logger
):
    curr_site_width = 1.0  # prescale_by_site_width
    node_lpos, node_size, node_weight, pin_id2node_id, node2pin_list, node2pin_list_end = rearrange_dpdb_node_info(
        node_pos, data
    )

    mov_lhs, mov_rhs = data.movable_index
    if args.scale_design:
        # scale back
        die_scale = data.die_scale / data.site_width # assume site width == 1 in dp
        node_lpos = node_lpos * die_scale
        node_size = node_size * die_scale
        pin_rel_lpos = data.pin_rel_lpos * die_scale
        die_info = (data.die_info.reshape(2, 2).t() * die_scale).t().reshape(-1)
        region_boxes = (
            (data.region_boxes.reshape(-1, 2, 2).permute(0, 2, 1) * die_scale)
            .permute(0, 2, 1)
            .reshape(-1, 4)
        ) # [:, 0] -> lx, [:, 1] -> hx, [:, 2] -> ly, [:, 3] -> hy
    else:
        pin_rel_lpos = data.pin_rel_lpos
        die_info = data.die_info
        region_boxes = data.region_boxes 

    _, floatmov_rhs, _ = data.node_type_indices[1]
    _, fix_rhs, _ = data.node_type_indices[2]
    _, iopin_rhs, _ = data.node_type_indices[3]
    _, blkg_rhs, _ = data.node_type_indices[4]
    _, floatiopin_rhs, _ = data.node_type_indices[5]
    _, floatfix_rhs, _ = data.node_type_indices[6]
    num_iopin = iopin_rhs - fix_rhs
    num_floatiopin = floatiopin_rhs - blkg_rhs

    die_info = die_info.cpu()
    xl = die_info[0].item()
    xh = die_info[1].item()
    yl = die_info[2].item()
    yh = die_info[3].item()
    num_movable_nodes = mov_rhs - mov_lhs
    num_nodes = node_lpos.shape[0] - num_iopin - num_floatiopin
    site_width = curr_site_width
    row_height = data.row_height / data.site_width

    if not use_cpu_db_ and not node_lpos.is_cuda:
        logger.error("Please set use_cpu_db == True when node_lpos is not on GPU")
        exit(0)

    if use_cpu_db_:
        dp_rawdb = gpudp.create_dp_rawdb(
            node_lpos.cpu(),
            node_size.cpu(),
            node_weight.cpu(),
            pin_rel_lpos.cpu(),
            pin_id2node_id.cpu(),
            data.pin_id2net_id.int().cpu(),
            node2pin_list.cpu(),
            node2pin_list_end.cpu(),
            data.hyperedge_list.int().cpu(),
            data.hyperedge_list_end.int().cpu(),
            data.net_mask.cpu(),
            data.node_id2region_id.int().cpu(),
            region_boxes.cpu(),
            data.region_boxes_end.int().cpu(),
            xl,
            xh,
            yl,
            yh,
            num_movable_nodes,
            num_nodes,
            site_width,
            row_height,
        )
    else:
        dp_rawdb = gpudp.create_dp_rawdb(
            node_lpos,
            node_size,
            node_weight,
            pin_rel_lpos,
            pin_id2node_id,
            data.pin_id2net_id.int(),
            node2pin_list,
            node2pin_list_end,
            data.hyperedge_list.int(),
            data.hyperedge_list_end.int(),
            data.net_mask,
            data.node_id2region_id.int(),
            region_boxes,
            data.region_boxes_end.int(),
            xl,
            xh,
            yl,
            yh,
            num_movable_nodes,
            num_nodes,
            site_width,
            row_height,
        )

    num_sites_x = round((xh - xl) / site_width)
    num_sites_y = round((yh - yl) / row_height)
    logger.info("Finish setup database. #siteX: %d #siteY: %d" % (num_sites_x, num_sites_y))

    return dp_rawdb


def get_ori_scale_factor(data: PlaceData) -> float:
    if data.dataset_format == "bookshelf":
        return 1.0
    else:
        return 1.0 / data.site_width


@nb.njit(cache=True)
def prime_factorization(x):
    lt = []
    while x != 1:
        for i in range(2, int(x + 1)):
            if x % i == 0:  # i is a prime factor
                lt.append(i)
                x = x / i  # get the quotient for further factorization
                break
    return lt


@nb.njit(cache=True)
def compute_scalar(ori_scale_factor):
    scale_factor = ori_scale_factor
    if ori_scale_factor != 1.0:
        inv_scale_factor = int(round(1.0 / ori_scale_factor))
        prime_factors = prime_factorization(inv_scale_factor)
        target_inv_scale_factor = 1
        for factor in prime_factors:
            if factor != 2 and factor != 5:
                target_inv_scale_factor = inv_scale_factor
                break
        scale_factor = 1.0 / target_inv_scale_factor
    return scale_factor


def commit_to_node_pos(node_pos: torch.Tensor, data:PlaceData, dp_rawdb):
    mov_lhs, mov_rhs = data.movable_index
    new_mov_cpos = torch.stack(
        (dp_rawdb.get_curr_lposx(), dp_rawdb.get_curr_lposy()
    ), dim=1).to(node_pos.device)[mov_lhs:mov_rhs] + data.node_size[mov_lhs:mov_rhs] / 2
    node_pos[mov_lhs:mov_rhs].data.copy_(new_mov_cpos)
    return node_pos


def run_lg(node_pos: torch.Tensor, data: PlaceData, args, logger):
    # CPU legalization
    lg_rawdb = setup_detailed_rawdb(node_pos, True, data, args, logger)

    # run LG
    logger.info("Start running Macro Legalization...")
    ml_time = time.time()
    num_bins_x, num_bins_y = data.num_bin_x, data.num_bin_y
    if gpudp.macroLegalization(lg_rawdb, num_bins_x, num_bins_y):
        lg_rawdb.commit()
    logger.info("Finish Macro Legalization. Time: %.4f" % (time.time() - ml_time))

    logger.info("Start running Greedy Legalization...")
    gl_time = time.time()
    num_bins_x, num_bins_y = 1, 64
    gpudp.greedyLegalization(lg_rawdb, num_bins_x, num_bins_y)
    if not lg_rawdb.check(get_ori_scale_factor(data)):
        logger.error("Check failed in Greedy Legalization")
    logger.info("Finish Greedy Legalization. Time: %.4f" % (time.time() - gl_time))

    logger.info("Start running Abacus Legalization...")
    al_time = time.time()
    gpudp.abacusLegalization(lg_rawdb, num_bins_x, num_bins_y)
    if not lg_rawdb.check(get_ori_scale_factor(data)):
        logger.error("Check failed in Abacus Legalization")
    logger.info("Finish Abacus Legalization. Time: %.4f" % (time.time() - al_time))

    lg_rawdb.commit()

    # Commit result
    commit_to_node_pos(node_pos, data, lg_rawdb)
    torch.cuda.synchronize(node_pos.device)
    logger.info("***** Finish Legalization, HPWL: %.4E Time: %.4f *****" % (
        get_obj_hpwl(node_pos, data, args).item(), time.time() - gl_time
    ))

    if args.scale_design:
        node_pos /= data.die_scale

    del lg_rawdb

    return node_pos


def trace_ops(func, *args):
    tracing_file = "test_trace.json"
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace(tracing_file)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2),
        on_trace_ready=trace_handler
        ) as p:
            for iter in range(4):
                func(*args)
                torch.cuda.synchronize()
                p.step()
    print("Finish tracing. Save file in %s. Exit program." % tracing_file)
    exit(0)
    # trace_ops(gpudp.kReorder, dp_rawdb, num_bins_x, num_bins_y, kr_K, kr_iter)


def run_dp(node_pos: torch.Tensor, data: PlaceData, args, logger):
    # GPU Detailed Placement
    dp_rawdb = setup_detailed_rawdb(node_pos, False, data, args, logger)
    # CPU Legality Check
    check_rawdb = setup_detailed_rawdb(node_pos, True, data, args, logger)

    num_bins_x = data.num_bin_x
    num_bins_y = data.num_bin_y
    kr_K = 4
    kr_iter = 2
    gs_bs = 256
    gs_iter = 2
    ism_bs = 2048
    ism_set = 128
    ism_iter = 50

    # use integer coordinate systems in DP for better quality
    scalar = compute_scalar(get_ori_scale_factor(data))

    def dp_handler(dp_func, func_name, *func_args):
        logger.info("Start running %s..." % func_name)
        start_time = time.time()
        if scalar != 1.0:
            logger.info("scale dp_rawdb by %g" % (1.0 / scalar))
            dp_rawdb.scale(1.0 / scalar, True)
        dp_func(dp_rawdb, *func_args)
        if scalar != 1.0:
            logger.info("scale dp_rawdb back by %g" % scalar)
            dp_rawdb.scale(scalar, False)
        # commit lpos for legality check
        torch.cuda.synchronize(node_pos.device)
        check_rawdb.commit_from(dp_rawdb.get_curr_lposx().cpu(), dp_rawdb.get_curr_lposy().cpu())
        if not check_rawdb.check(get_ori_scale_factor(data)):
            dp_rawdb.rollback()
            logger.error("Check failed in %s. Rollback to previous DP iteration." % func_name)
            return
        # update dp_rawdb for next step dp_func and update the final solution
        dp_rawdb.commit()
        commit_to_node_pos(node_pos, data, dp_rawdb)
        logger.info("***** Finish %s, HPWL: %.4E Time: %.4f *****" % (
            func_name, get_obj_hpwl(node_pos, data, args).item(), time.time() - start_time
        ))
    
    dp_handler(gpudp.kReorder, "K-Reorder 1", num_bins_x, num_bins_y, kr_K, kr_iter)
    dp_handler(gpudp.independentSetMatching, "Independent Set Match", num_bins_x, num_bins_y, ism_bs, ism_set, ism_iter)
    dp_handler(gpudp.globalSwap, "Global Swap", num_bins_x // 2, num_bins_y // 2, gs_bs, gs_iter)
    dp_handler(gpudp.kReorder, "K-Reorder 2", num_bins_x, num_bins_y, kr_K, kr_iter)

    if args.scale_design:
        node_pos /= data.die_scale

    del check_rawdb, dp_rawdb

    return node_pos


def run_dp_route_opt(node_pos: torch.Tensor, gpdb, rawdb, ps, data: PlaceData, args, logger):
    # NOTE: we suppose M1's prefer routing direction is 0 (horizontal)
    if ps.enable_route and gpdb.m1direction() == 0:
        func_name = "PA-Refine"
        logger.info("Start running %s" % func_name)
        start_time = time.time()
        node_pos_bk = node_pos.clone()
        mov_lhs, mov_rhs = data.movable_index
        node_lpos = torch.cat((
            node_pos[:mov_rhs].detach() - data.node_size[:mov_rhs] / 2,
            data.node_lpos[mov_rhs:],
        ), dim=0).cpu()
        node_size = data.node_size.cpu()
        if args.scale_design:
            # scale back
            die_scale = data.die_scale / data.site_width # assume site width == 1 in dp
            node_lpos = node_lpos * die_scale
            node_size = node_size * die_scale
            die_info = (data.die_info.reshape(2, 2).t() * die_scale).t().reshape(-1)
        site_width = 1.0
        row_height = data.row_height / data.site_width
        die_info = data.die_info.cpu()
        dieLX = die_info[0].item()
        dieHX = die_info[1].item()
        dieLY = die_info[2].item()
        dieHY = die_info[3].item()
        K = 5
        new_node_lpos = routedp.dp_route_opt(
            node_lpos, node_size, dieLX, dieHX, dieLY, dieHY, 
            site_width, row_height, rawdb, gpdb, K
        )
        new_mov_cpos = new_node_lpos.to(node_pos.device)[mov_lhs:mov_rhs] + data.node_size[mov_lhs:mov_rhs] / 2
        node_pos[mov_lhs:mov_rhs].data.copy_(new_mov_cpos)

        check_rawdb = setup_detailed_rawdb(node_pos, True, data, args, logger)
        if not check_rawdb.check(get_ori_scale_factor(data)):
            logger.error("Check failed in %s. Rollback to previous DP iteration." % func_name)
            node_pos[mov_lhs:mov_rhs].data.copy_(node_pos_bk[mov_lhs:mov_rhs])

        logger.info("***** Finish %s, HPWL: %.4E Time: %.4f *****" % (
            func_name, get_obj_hpwl(node_pos, data, args).item(), time.time() - start_time
        ))

    return node_pos


def commit_node_pos_to_gpdb(node_pos, gpdb, data: PlaceData):
    exact_node_pos = torch.round(node_pos * data.die_scale + data.die_shift)
    exact_node_lpos = torch.round(exact_node_pos - torch.round(data.node_size * data.die_scale) / 2).cpu()
    gpdb.apply_node_lpos(exact_node_lpos)

def write_placement(node_pos, gpdb, id, data: PlaceData, args, logger):
    res_root = os.path.join(args.result_dir, args.exp_id)
    prefix = os.path.join(res_root, args.output_dir, "%s_%s_%s" %(args.output_prefix, args.design_name, id))
    if not os.path.exists(os.path.dirname(prefix)):
        os.makedirs(os.path.dirname(prefix))
    start_write_time = time.time()
    if args.write_global_placement and data.dataset_format == "bookshelf":
        logger.info("Use python to generate .pl file")
        data.write_pl(node_pos, prefix)
    else:
        commit_node_pos_to_gpdb(node_pos, gpdb, data)
        gpdb.write_placement(prefix)
    logger.info("Write placement in %s. Time: %.4f" % (prefix, time.time() - start_write_time))
    return prefix

def external_detail_placement(input_file, data: PlaceData, args, logger, eval_mode=True, dp_engine_name=None):
    eval_flag = "-nolegal -nodetail" if eval_mode else ""

    dp_start_time = None
    dp_end_time = None
    dp_hpwl = -1
    top5overflow = -1

    post_fix = None
    if data.dataset_format == "lefdef":
        post_fix = "def"
        assert input_file.split(".")[-1] == post_fix
    elif data.dataset_format == "bookshelf":
        post_fix = "pl"
        assert input_file.split(".")[-1] == post_fix
    dp_out_file = input_file.replace(".%s" % post_fix, "")

    if dp_engine_name == "ntuplace3" and data.dataset_format == "bookshelf":
        dp_engine = "./thirdparty/placers/ntuplace3/ntuplace3"
        aux_input = data.dataset_path["aux"]
        target_density_cmd = ""
        if args.target_density < 1.0:
            target_density_cmd = " -util %f" % (args.target_density)
        cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
            dp_engine, aux_input, input_file, target_density_cmd, dp_out_file, eval_flag)
        logger.info(cmd)
        # os.system(cmd)
        dp_start_time = time.time()
        output = os.popen(cmd).read()
        dp_end_time = time.time()
        dp_hpwl = float(output.split("========\n         HPWL=")[1].split("Time")[0].strip())
        dp_out_file = dp_out_file + ".ntup.pl"
    elif dp_engine_name == "ntuplace4dr" and data.dataset_format == "lefdef":
        dp_engine = "./thirdparty/placers/ntuplace4dr/ntuplace4dr_binary/placer"
        cmd = dp_engine
        if "lef" in data.dataset_path.keys():
            tech_lef = data.dataset_path["lef"]
            cell_lef = data.dataset_path["lef"]
        else:
            tech_lef = data.dataset_path["tech_lef"]
            cell_lef = data.dataset_path["cell_lef"]
        cmd += " -tech_lef %s" % tech_lef
        cmd += " -cell_lef %s" % cell_lef
        benchmark_dir = os.path.dirname(tech_lef)
        cmd += " -floorplan_def %s" % (input_file)
        cmd += " -out ntuplace_4dr_out"
        cmd += " -placement_constraints %s/placement.constraints" % (benchmark_dir)
        cmd += " -cpu %d" % args.num_threads
        cmd += " -noglobal %s; " % eval_flag
        cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (dp_out_file)
        cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (dp_out_file)
        cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
        cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (dp_out_file)
        cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (dp_out_file)
        if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
            cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
        cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
        cmd += "rm -rf %s/dat ; " % (os.path.dirname(dp_out_file))
        cmd += "rm -rf %s/*.plt ; " % (os.path.dirname(dp_out_file))
        cmd += "rm -rf %s ; " % ("log_result.txt")
        logger.info("%s" % (cmd))
        dp_start_time = time.time()
        output = os.popen(cmd).read()
        dp_end_time = time.time()
        # NOTE: the DP HPWL reported by NTUplace4dr is not normalized by site_width
        unscale_dp_hpwl = float(output.split("=======\n         HPWL=")[1].split("Time")[0].strip())
        scaled_hpwl = float(output.split("=======\n         HPWL=")[1].split("\nHPWL ")[1].split("  (x")[0].strip())
        dp_hpwl = scaled_hpwl
        top5overflow = float(output.split("[CONG] Top 5 Overflow")[1].split("\n")[0].strip())
    elif dp_engine_name == "rippledp" and data.dataset_format == "lefdef":
        dp_engine = "./thirdparty/placers/rippledp/placer"
        cmd = dp_engine
        if "lef" in data.dataset_path.keys():
            tech_lef = data.dataset_path["lef"]
            cell_lef = data.dataset_path["lef"]
            cmd += " -tech_lef %s" % tech_lef
        else:
            tech_lef = data.dataset_path["tech_lef"]
            cell_lef = data.dataset_path["cell_lef"]
            cmd += " -tech_lef %s" % tech_lef
            cmd += " -cell_lef %s" % cell_lef
        benchmark_dir = os.path.dirname(tech_lef)
        cmd += " -floorplan_def %s" % (input_file)
        cmd += " -placement_constraints %s/placement.constraints" % (benchmark_dir)
        cmd += " -output rippedp_out.def"
        cmd += " -cpu %d ; " % args.num_threads
        cmd += "mv rippedp_out.def %s.rippledp.def ; " % (dp_out_file)
        if eval_mode:
            logger.warning("RippleDP cannot support eval mode. Please Check.")
        logger.info("%s" % (cmd))
        dp_start_time = time.time()
        output = os.system(cmd)
        dp_end_time = time.time()
    else:
        raise NotImplementedError("DP Engine %s for %s format unsupported" % (dp_engine_name, data.dataset_format))
    if eval_mode:
        logger.info("Finish external detailed placer validation. Time: %.2f seconds" %
                        (dp_end_time - dp_start_time))
        os.system("rm -rf %s.ntup.def" % dp_out_file)
    else:
        logger.info("Finish external detailed placement. LG+DP Time: %.2f seconds" %
                        (dp_end_time - dp_start_time))
        logger.info("After DP, HPWL: %.4E" % dp_hpwl)
        logger.info("Write detail placement in %s" % dp_out_file)
    # del gpdb, rawdb
    # logger.info("Evaluating detail placement result...")
    # data, rawdb, gpdb = load_dataset(args, logger, dp_out_file)
    # data = data.to(device).preprocess()
    # hpwl = get_obj_hpwl(data.node_pos, data, args).item()
    # info = (iteration + 1, hpwl, data.design_name)
    # draw_fig_with_cairo_cpp(data.node_pos, data.node_size, data, info, args)
    # logger.info("After DP, HPWL: %.4E" % hpwl)

    dp_time = dp_end_time - dp_start_time if dp_end_time is not None else 0.0

    return dp_hpwl, top5overflow, dp_time


def default_detail_placement(node_pos, gpdb, rawdb, ps, data: PlaceData, args, logger):
    dp_start_time = None
    dp_end_time = None
    dp_hpwl = -1

    torch.cuda.synchronize(node_pos.device)
    dp_start_time = time.time()
    node_pos = run_lg(node_pos, data, args, logger)
    torch.cuda.synchronize(node_pos.device)
    lg_end_time = time.time()
    node_pos = run_dp(node_pos, data, args, logger)
    torch.cuda.synchronize(node_pos.device)
    node_pos = run_dp_route_opt(node_pos, gpdb, rawdb, ps, data, args, logger)
    dp_end_time = time.time()
    logger.info("Finish detailed placement. LG Time: %.4f DP Time: %.4f LG+DP Time: %.4f" % (
        lg_end_time - dp_start_time, dp_end_time - lg_end_time, dp_end_time - dp_start_time
    ))
    # Evaluate
    dp_hpwl = get_obj_hpwl(node_pos, data, args).item()
    info = (ps.iter + 1, dp_hpwl, data.design_name)
    if args.draw_placement:
        draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
    logger.info("After DP, HPWL: %.4E" % dp_hpwl)

    lg_time = lg_end_time - dp_start_time
    dp_time = dp_end_time - lg_end_time

    return node_pos, dp_hpwl, lg_time, dp_time


def detail_placement_main(node_pos, gpdb, rawdb, ps, data: PlaceData, args, logger):
    dp_hpwl, top5overflow, lg_time, dp_time = -1, -1, -1, -1
    gp_out_file, dp_out_file = None, None
    preprocess_db_cache.reset()

    post_fix = None
    if data.dataset_format == "lefdef":
        post_fix = ".def"
    elif data.dataset_format == "bookshelf":
        post_fix = ".pl"
    if args.dp_engine in ["ntuplace3", "ntuplace4dr", "rippledp"]:
        # write GP solution for external dp/lg engine
        args.write_global_placement = True
        assert args.write_placement and args.load_from_raw
    if args.write_global_placement and args.write_placement and args.load_from_raw:
        gp_prefix = write_placement(node_pos, gpdb, "gp", data, args, logger)
        gp_out_file = gp_prefix + post_fix

    args.write_global_placement = False # we won't write GP solution anymore

    if args.detail_placement:
        logger.info("------- Start DP -------")
        if args.dp_engine in ["ntuplace3", "ntuplace4dr", "rippledp"]:
            # use external engine to perform lg/dp and write solution
            dp_hpwl, top5overflow, dp_time = external_detail_placement(
                gp_out_file, data, args, logger, eval_mode=False, dp_engine_name=args.dp_engine
            )
        elif args.dp_engine == "default":
            # use default engine (basically follow ABCDPlace) to perform lg/dp
            node_pos, dp_hpwl, lg_time, dp_time = default_detail_placement(
                node_pos, gpdb, rawdb, ps, data, args, logger
            )
            # write solution and evaluate solution by external engine
            if args.write_placement and args.load_from_raw:
                dp_prefix = write_placement(node_pos, gpdb, "dp", data, args, logger)
                dp_out_file = dp_prefix + post_fix
                if args.eval_by_external:
                    logger.info("Eval solution by external DetailedPlacer.")
                    if args.eval_engine != "ntuplace3" and data.dataset_format == "bookshelf":
                        logger.warning("Use ntuplace3 instead of %s to eval bookshelf format" % args.eval_engine)
                        args.eval_engine = "ntuplace3"
                    ext_dp_hpwl, top5overflow, _ = external_detail_placement(
                        dp_out_file, data, args, logger, eval_mode=True, dp_engine_name=args.eval_engine
                    )
                    logger.info("External engine evaluated DP HPWL: %.4E Top-5 OVFL: %.2f" % 
                        (ext_dp_hpwl, top5overflow))
                    dp_hpwl = ext_dp_hpwl
        else:
            raise NotImplementedError("DP Engine %s unsupported" % args.dp_engine)

    preprocess_db_cache.reset()

    return node_pos, dp_hpwl, top5overflow, lg_time, dp_time
