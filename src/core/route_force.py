import torch
from cpp_to_py import gpugr
from .dct2_fft2 import dct2, idct2, idxst_idct, idct_idxst
from .torch_dct import torch_dct_idct
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
import torchvision.transforms


class RouteCache:
    def __init__(self) -> None:
        self.first_run = True
        self.grdb = None
        self.input_mat: torch.Tensor = None
        self.routeforce = None
        self.route_gradmat: torch.Tensor = None
        self.mov_route_grad: torch.Tensor = None
        self.placeable_area = None
        self.target_area = None
        self.whitespace_area = None
        self.mov_node_size_real: torch.Tensor = None
        self.original_filler_area_total = None
        self.original_pin_rel_cpos: torch.Tensor = None
        self.original_target_density = None
        self.original_num_fillers = None
        self.original_mov_node_size: torch.Tensor = None
        self.original_mov_node_size_real: torch.Tensor = None

    def reset(self):
        self.first_run = True
        self.grdb = None
        self.input_mat = None
        self.routeforce = None
        self.route_gradmat = None
        self.mov_route_grad = None
        self.placeable_area = None
        self.target_area = None
        self.whitespace_area = None
        self.mov_node_size_real = None
        self.original_filler_area_total = None
        self.original_pin_rel_cpos = None
        self.original_target_density = None
        self.original_num_fillers = None
        self.original_mov_node_size = None
        self.original_mov_node_size_real = None


route_cache = RouteCache()


def get_route_input_mat():
    return route_cache.input_mat


def draw_cg_fig(args, t: torch.Tensor, info, title):
    design_name, iteration, pic_type = info
    filename = "%s_iter%d_%s.png" % (design_name, iteration, pic_type)
    res_root = os.path.join(args.result_dir, args.exp_id)
    png_path: str = os.path.join(res_root, "route", filename)
    if not os.path.exists(os.path.dirname(png_path)):
        os.makedirs(os.path.dirname(png_path))
    
    t_clamp = t.clamp(max=3.0)
    ratio = t_clamp.shape[1] / t_clamp.shape[0]
    plt.figure(figsize=(6, 5 * ratio))
    ax = sns.heatmap(t_clamp.t().flip(0).cpu().numpy(), cmap="YlGnBu")
    plt.title(title)
    plt.savefig(png_path)
    plt.close()


def evaluate_routability(args, logger, cg_mapHV: torch.Tensor):
    # compute RC
    ace_list = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ace_tsr = torch.tensor(ace_list, device=cg_mapHV.device, dtype=cg_mapHV.dtype)
    tmp: torch.Tensor = torch.sort((cg_mapHV + 1).reshape(2, -1), descending=True)[0]
    rc = torch.cumsum(tmp, 1) / torch.arange(1, tmp.shape[1] + 1, device=tmp.device, dtype=tmp.dtype)
    indices = (tmp.shape[1] * ace_tsr).long()
    selected_rc = rc[:, indices].cpu()
    log_str = "\n           "
    log_str += "\t".join(["ACE"] + ["%.2f%%" % (i * 100) for i in ace_list]) + "\n           "
    log_str += "\t".join(["HOR"] + ["%.4f" % i for i in selected_rc[0]]) + "\n           "
    log_str += "\t".join(["VER"] + ["%.4f" % i for i in selected_rc[1]])
    logger.info('RC Value:%s' % log_str)

    return ace_list, selected_rc

def calc_gr_wl_via(grdb, routeforce):
    step_x, step_y = routeforce.gcell_steps()
    layer_pitch = routeforce.layer_pitch()
    layer_m2_pitch = layer_pitch[1] if len(layer_pitch) > 1 else layer_pitch[0]

    gr_wirelength, gr_numVias = grdb.report_gr_stat()
    gr_wirelength = gr_wirelength * max(step_x, step_y) / layer_m2_pitch
    
    return gr_wirelength, gr_numVias

def estimate_num_shorts(routeforce, gpdb, cap_map, wire_dmd_map, via_dmd_map):
    step_x, step_y = routeforce.gcell_steps()
    layer_width = routeforce.layer_width()
    layer_pitch = routeforce.layer_pitch()
    microns = float(routeforce.microns())
    layer_m2_pitch = layer_pitch[1] if len(layer_pitch) > 1 else layer_pitch[0]

    m1direction = gpdb.m1direction()  # 0 for H, 1 for V, metal1's layer idx is 0
    hId = 1 if m1direction else 0
    vId = 0 if m1direction else 1

    layer_area = torch.tensor(layer_width, device=cap_map.device, dtype=cap_map.dtype)
    layer_area[hId::2].mul_(step_x / microns / layer_m2_pitch / layer_m2_pitch)
    layer_area[vId::2].mul_(step_y / microns / layer_m2_pitch / layer_m2_pitch)
    
    wire_ovfl_map = (wire_dmd_map - cap_map).clamp_(min=0.0)
    routedShortArea = (wire_ovfl_map.sum(dim=(1, 2)) * layer_area).sum()

    via_ovfl_mask = (wire_dmd_map > cap_map).float()
    routedShortViaNum = (via_ovfl_mask * via_dmd_map).sum()

    return (routedShortArea + routedShortViaNum).item()


def get_fft_scale(num_bin_x, num_bin_y, device, scale_w_k=True):
    w_j = (
        torch.arange(num_bin_x, device=device)
        .float()
        .mul(2 * np.pi / num_bin_x)
        .reshape(num_bin_x, 1)
    )
    w_k = (
        torch.arange(num_bin_y, device=device)
        .float()
        .mul(2 * np.pi / num_bin_y)
        .reshape(1, num_bin_y)
    )
    # scale_w_k because the aspect ratio of a bin may not be 1
    # NOTE: we will not scale down w_k in NN since it may distrub the training
    if scale_w_k:
        w_k.mul_(num_bin_y / num_bin_x)
    wj2_plus_wk2 = w_j.pow(2) + w_k.pow(2)
    wj2_plus_wk2[0, 0] = 1.0

    potential_scale = 1.0 / wj2_plus_wk2
    potential_scale[0, 0] = 0.0

    force_x_scale = w_j * potential_scale * 0.5
    force_y_scale = w_k * potential_scale * 0.5

    force_x_coeff = ((-1.0) ** torch.arange(num_bin_x, device=device)).unsqueeze(1)
    force_y_coeff = ((-1.0) ** torch.arange(num_bin_y, device=device)).unsqueeze(0)

    potential_coeff = 1.0

    return (
        potential_scale,
        potential_coeff,
        force_x_scale,
        force_y_scale,
        force_x_coeff,
        force_y_coeff,
    )


def get_route_force(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
    constraint_fn=None, skip_m1_route=True, enable_filler_grad=True
):
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_connected_index
    _, filler_lhs = data.movable_connected_index
    filler_rhs = mov_node_pos.shape[0]
    num_fillers = filler_rhs - filler_lhs
    num_conn_nodes = (mov_rhs - mov_lhs) + (fix_rhs - fix_lhs)

    mov_route_grad = torch.zeros_like(mov_node_pos)
    mov_congest_grad = torch.zeros_like(mov_node_pos)
    mov_pseudo_grad = torch.zeros_like(mov_node_pos)

    # 1) run global routing and compute gradient mat
    grdb, input_mat, routeforce, route_gradmat = None, None, None, None
    if ps.rerun_route:
        output = run_gr_and_fft_main(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos, 
            constraint_fn=constraint_fn, skip_m1_route=skip_m1_route
        )
        grdb, routeforce, input_mat, cg_mapHV, map_raw, map_2d, route_gradmat, gr_metrics = output
        dmd_map, wire_dmd_map, via_dmd_map, cap_map = map_raw
        dmd_map2d, wire_dmd_map2d, via_dmd_map2d, cap_map2d = map_2d
        # ------------------------------------------------------------
        # 2) start force computation
        # 2.1) compute routing wire force
        conn_route_grad = conn_route_force(
            num_conn_nodes, input_mat, wire_dmd_map2d, via_dmd_map2d, cap_map2d,
            route_gradmat, routeforce, args, data
        )
        mov_route_grad[mov_lhs:mov_rhs] = conn_route_grad[mov_lhs:mov_rhs]

        route_cache.grdb = grdb
        route_cache.input_mat = input_mat
        route_cache.routeforce = routeforce
        route_cache.route_gradmat = route_gradmat
        route_cache.mov_route_grad = mov_route_grad
    else:
        grdb = route_cache.grdb
        input_mat = route_cache.input_mat
        routeforce = route_cache.routeforce
        route_gradmat = route_cache.route_gradmat
        mov_route_grad = route_cache.mov_route_grad

    ps.mov_node_to_num_pseudo_pins = torch.zeros_like(mov_node_pos)

    if num_fillers > 0:
        # 2.2) compute congestion region force
        mov_congest_grad[filler_lhs:filler_rhs] += cell_congestion_force(
            data, mov_node_pos, mov_node_size, expand_ratio,
            route_gradmat, routeforce, filler_lhs, filler_rhs, 1.0
        )
        mov_congest_grad[:filler_lhs] += cell_congestion_force(
            data, mov_node_pos, mov_node_size, expand_ratio,
            route_gradmat, routeforce, 0, filler_lhs, -1.0
        )
        
        # 2.3) compute pseudo net force, a force to push fillers to congested area
        num_fillers_selected = int(num_fillers * 0.05)
        mov_pseudo_grad[filler_lhs:filler_lhs + num_fillers_selected] += filler_pseudo_wire_force(
            data, ps, mov_node_pos, mov_node_size, routeforce, input_mat, filler_lhs, filler_lhs + num_fillers_selected
        )
        ps.mov_node_to_num_pseudo_pins[filler_lhs:filler_lhs + num_fillers_selected] += 1
    else:
        mov_congest_grad, mov_pseudo_grad = None, None
        logger.warning("No filler, cannot use filler route force")

    return mov_route_grad, mov_congest_grad, mov_pseudo_grad


def run_gr_and_fft_main(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, constraint_fn=None, **kwargs
):
    output = None
    if ps.rerun_route:
        logger.info("Update gpdb node pos...")
        mov_lhs, mov_rhs = data.movable_index
        mov_node_pos = constraint_fn(mov_node_pos)
        node_pos = mov_node_pos[mov_lhs:mov_rhs]
        node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
        exact_node_pos = torch.round(node_pos * data.die_scale + data.die_shift)
        exact_node_lpos = torch.round(exact_node_pos - torch.round(data.node_size * data.die_scale) / 2).cpu()
        gpdb.apply_node_lpos(exact_node_lpos)

        output = run_gr_and_fft(args, logger, data, rawdb, gpdb, ps, **kwargs)
    return output


def run_gr_and_fft(args, logger, data, rawdb, gpdb, ps, grdb=None, skip_m1_route=True, run_fft=False, visualize=False, report_gr_metrics_only=False, given_gr_params={}):
    route_size = 512
    iteration = ps.iter - 1  # ps.iter is increased before running GR optimization
    die_ratio = (data.__ori_die_hx__ - data.__ori_die_lx__) / (data.__ori_die_hy__ - data.__ori_die_ly__)
    route_xSize = route_size if die_ratio <= 1 else round(route_size / die_ratio)
    route_ySize = route_size if die_ratio >= 1 else round(route_size / die_ratio)

    # 1.1) init GRDatabase
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    gr_params = {"device_id": args.gpu, "route_xSize": route_xSize, "route_ySize": route_ySize}
    gr_params.update(given_gr_params)
    gpugr.load_gr_params(gr_params)

    # 1.2) do GR
    logger.info("--------- Start GR in Iter: %d ---------" % iteration)
    if grdb is None:
        grdb = gpugr.create_grdatabase(rawdb, gpdb)
    routeforce = gpugr.create_routeforce(grdb)
    routeforce.run_ggr()
    logger.info("--------- End GR ---------")

    m1direction = gpdb.m1direction()  # 0 for H, 1 for V, metal1's layer idx is 0
    hId = 1 if m1direction else 0
    vId = 0 if m1direction else 1
    aId = 0
    if skip_m1_route:
        aId = 1
        hId = hId + 2 if hId == 0 else hId
        vId = vId + 2 if vId == 0 else vId

    # 1.3) calculate capacity and demand map
    dmd_map, wire_dmd_map, via_dmd_map = routeforce.dmd_map()
    cap_map: torch.Tensor = routeforce.cap_map()

    dmd_map2d: torch.Tensor = dmd_map[aId:].sum(dim=0)
    wire_dmd_map2d: torch.Tensor = wire_dmd_map[aId:].sum(dim=0)
    via_dmd_map2d: torch.Tensor = via_dmd_map[aId:].sum(dim=0)
    cap_map2d: torch.Tensor = cap_map[aId:].sum(dim=0)

    cg_mapAll = dmd_map2d / cap_map2d  # ignore metal0

    # input_mat = cg_mapAll
    input_mat = torch.where(cg_mapAll > 1, cg_mapAll - 1, 0)
    # input_mat = torch.log(input_mat + 1)

    cg_mapH = dmd_map[hId::2].sum(dim=0) / cap_map[hId::2].sum(dim=0)
    cg_mapV = dmd_map[vId::2].sum(dim=0) / cap_map[vId::2].sum(dim=0)
    cg_mapHV = torch.stack((cg_mapH, cg_mapV))
    cg_mapHV = torch.where(cg_mapHV > 1, cg_mapHV - 1, 0)

    cgOvfl = (dmd_map[aId:] / cap_map[aId:]).max(dim=0)[0].clamp(min=1.0) - 1
    map_raw = (dmd_map, wire_dmd_map, via_dmd_map, cap_map)
    map_2d = (dmd_map2d, wire_dmd_map2d, via_dmd_map2d, cap_map2d)

    # 1.4) compute congestion map's gradient
    route_gradmat = None
    if run_fft:
        fft_scale = get_fft_scale(input_mat.shape[0], input_mat.shape[1], device)
        potential_scale, _, force_x_scale, force_y_scale, _, _ = fft_scale
        fft_coeff = dct2(input_mat)
        force_x_map = idxst_idct(fft_coeff * force_x_scale)
        force_y_map = idct_idxst(fft_coeff * force_y_scale)
        potential_map = idct2(fft_coeff * potential_scale)
        route_gradmat = torch.vstack(
            (force_x_map.unsqueeze(0), force_y_map.unsqueeze(0))
        ).contiguous()  # 2 x M x N

    # 1.5) print stat
    logger.info(
        "cgMap max: %.4f mean: %.4f std: %.4f | cgOvfl max: %.4f mean: %.4f std: %.4f"
        % (
            input_mat.max().item(),
            input_mat.mean().item(),
            input_mat.std().item(),
            cgOvfl.max().item(),
            cgOvfl.mean().item(),
            cgOvfl.std().item(),
        )
    )
    logger.info(
        "cgMapH max: %.4f mean: %.4f std: %.4f | cgMapV max: %.4f mean: %.4f std: %.4f"
        % (
            cg_mapHV[0].max().item(),
            cg_mapHV[0].mean().item(),
            cg_mapHV[0].std().item(),
            cg_mapHV[1].max().item(),
            cg_mapHV[1].mean().item(),
            cg_mapHV[1].std().item(),
        )
    )

    numOvflNets = routeforce.num_ovfl_nets()
    gr_wirelength, gr_numVias = calc_gr_wl_via(grdb, routeforce)
    gr_numShorts = estimate_num_shorts(routeforce, gpdb, cap_map, wire_dmd_map, via_dmd_map)

    ace_list, selected_rc = evaluate_routability(args, logger, cg_mapHV)
    rc_hor_mean, rc_ver_mean = selected_rc.mean(dim=1).tolist()

    logger.info(
        "#OvflNets: %d (%.2f%%), GR WL: %d, GR #Vias: %d, #EstShorts: %d, RC Hor: %.3f, RC Ver: %.3f" % (
            numOvflNets,
            numOvflNets / data.num_nets * 100,
            gr_wirelength,
            gr_numVias,
            gr_numShorts,
            rc_hor_mean,
            rc_ver_mean,
        )
    )

    gr_metrics = (numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean)

    if visualize:
        title = "#OvflNets: %.2e, WL: %.2e, #Vias: %.2e\n#Shorts: %.2e, RC Hor: %.3f, RC Ver: %.3f " % (
            numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean
        )
        draw_cg_fig(args, input_mat, (data.design_name, iteration, "cg_mapAll"), title)
        draw_cg_fig(args, cg_mapHV[0], (data.design_name, iteration, "cg_mapH"), title)
        draw_cg_fig(args, cg_mapHV[1], (data.design_name, iteration, "cg_mapV"), title)
        # draw_cg_fig(args, route_gradmat[0], (data.design_name, iteration, "route_gradmatX"), title)
        # draw_cg_fig(args, route_gradmat[1], (data.design_name, iteration, "route_gradmatY"), title)

    if report_gr_metrics_only:
        return gr_metrics

    return grdb, routeforce, input_mat, cg_mapHV, map_raw, map_2d, route_gradmat, gr_metrics


def conn_route_force(
    num_conn_nodes, input_mat, wire_dmd_map2d, via_dmd_map2d, cap_map2d,
    route_gradmat, routeforce, args, data
):
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    max_n_grid = max(input_mat.shape[0], input_mat.shape[1])
    mask_map = (input_mat > 0).float()
    # mask_map = torch.logical_or(cg_mapH > 1.5, cg_mapV > 1.5).float()
    dist_weights = torch.ones(max_n_grid + 2, device=device)
    wirelength_weights = torch.ones(max_n_grid + 2, device=device)
    wirelength_weights[20:] = 0
    unit_wire_cost = 1.0
    unit_via_cost = 1.0
    grad_weight = -1.0

    conn_route_grad: torch.Tensor = routeforce.route_grad(
        mask_map,
        wire_dmd_map2d,
        via_dmd_map2d,
        cap_map2d,
        dist_weights,
        wirelength_weights,
        route_gradmat,
        data.node2pin_list,
        data.node2pin_list_end,
        grad_weight,
        unit_wire_cost,
        unit_via_cost,
        num_conn_nodes,
    )

    return conn_route_grad


def cell_congestion_force(
    data, mov_node_pos, mov_node_size, expand_ratio, route_gradmat, routeforce, lhs, rhs, grad_weight=1.0
):
    # NOTE: grad_weight == 1.0, push cell to congested area
    num_bin_x, num_bin_y = route_gradmat.shape[1], route_gradmat.shape[2]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width

    filler_weight = torch.ones(rhs - lhs, device=mov_node_pos.device)

    filler_route_grad = routeforce.filler_route_grad(
        mov_node_pos[lhs:rhs],
        mov_node_size[lhs:rhs],
        filler_weight,
        expand_ratio[lhs:rhs],
        route_gradmat,
        grad_weight,
        unit_len_x,
        unit_len_y,
        num_bin_x,
        num_bin_y,
        rhs - lhs
    )

    return filler_route_grad


def filler_pseudo_wire_force(data, ps, mov_node_pos, mov_node_size, routeforce, cg_mapAll, lhs, rhs):
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=7, sigma=2)
    cg_mapAll_blurred: torch.Tensor = blurrer(cg_mapAll.unsqueeze(0)).squeeze(0)
    meanKrnl = 11
    cg_mapMean = torch.nn.functional.avg_pool2d(cg_mapAll.unsqueeze(0), meanKrnl, 1, padding=meanKrnl // 2).squeeze(0)
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width

    pseudo_pin_pos = (cg_mapMean == torch.max(cg_mapMean)).nonzero()[0].float() + 0.5
    scale = torch.tensor([unit_len_x, unit_len_y], device=mov_node_pos.device)
    pseudo_pin_pos.mul_(scale)
    pseudo_pin_pos = pseudo_pin_pos.repeat(rhs - lhs, 1)
    pseudo_pin_pos.add_(torch.randn_like(pseudo_pin_pos) * scale * 5)

    pseudo_grad = routeforce.pseudo_grad(mov_node_pos[lhs:rhs], pseudo_pin_pos, ps.wa_coeff)

    return pseudo_grad


def route_inflation(
    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
    constraint_fn=None, skip_m1_route=True, use_weighted_inflation=True, hv_same_ratio=True,
    min_area_inc=0.01, decrease_target_density=False, **kwargs
):
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_connected_index
    _, filler_lhs = data.movable_connected_index
    filler_rhs = mov_node_pos.shape[0]
    num_fillers = filler_rhs - filler_lhs

    if route_cache.first_run:
        # TODO: check args.target_density should be changed or not?
        route_cache.original_mov_node_size = mov_node_size.clone()
        # NOTE: size_real denotes the node size before node expand
        # these size_real are internally maintained by route_cache 
        route_cache.mov_node_size_real = data.mov_node_size_real.clone()
        route_cache.original_mov_node_size_real = data.mov_node_size_real.clone()
        mov_node_size_real = route_cache.mov_node_size_real

        mov_conn_size = mov_node_size_real[mov_lhs:filler_lhs, ...]
        filler_size = mov_node_size_real[filler_lhs:filler_rhs, ...]

        route_cache.first_run = False
        tmp_area = torch.sum(torch.prod(mov_node_size_real, 1)).item()
        route_cache.target_area = tmp_area
        route_cache.placeable_area = tmp_area / args.target_density
        route_cache.whitespace_area = route_cache.placeable_area - torch.sum(torch.prod(mov_conn_size, 1)).item()
        route_cache.original_num_fillers = filler_rhs - filler_lhs
        route_cache.original_filler_area_total = torch.sum(torch.prod(filler_size, 1)).item()
        route_cache.original_pin_rel_cpos = data.pin_rel_cpos.clone()
        route_cache.original_target_density = copy.deepcopy(args.target_density)

    ori_mov_node_size = route_cache.original_mov_node_size
    ori_mov_node_size_real = route_cache.original_mov_node_size_real
    mov_node_size_real = route_cache.mov_node_size_real

    # 1) check remain space
    last_mov_area = torch.prod(mov_node_size_real[mov_lhs:filler_lhs], 1)
    last_mov_area_total = last_mov_area.sum().item()
    max_inc_area_total = min(0.1 * route_cache.whitespace_area, route_cache.placeable_area - last_mov_area_total) # TODO: tune
    if max_inc_area_total <= 0:
        logger.warning("No space to inflate. Terminate inflation.")
        ps.use_cell_inflate = False  # not inflation anymore
        return None
    
    # 2) run GR to get congestion map
    output = run_gr_and_fft_main(
        args, logger, data, rawdb, gpdb, ps, mov_node_pos, 
        constraint_fn=constraint_fn, skip_m1_route=skip_m1_route,
        **kwargs
    )
    grdb, routeforce, input_mat, cg_mapHV, _, _, route_gradmat, gr_metrics = output

    num_bin_x, num_bin_y = input_mat.shape[0], input_mat.shape[1]
    unit_len_x, unit_len_y = routeforce.gcell_steps()
    unit_len_x /= data.site_width
    unit_len_y /= data.site_width

    # 3) get next step inflation ratio
    # TODO: tune parameters
    mov_node_weight = torch.ones(mov_node_size.shape[0], device=mov_node_size.device)
    if hv_same_ratio:
        inflate_mat: torch.Tensor = torch.stack((input_mat + 1, input_mat + 1)).contiguous().pow_(2)
    else:
        inflate_mat: torch.Tensor = (cg_mapHV + 1).permute(0, 2, 1).contiguous()
    inflate_mat.clamp_(min=1.0, max=2.0)

    # NOTE: 1) If use_weighted_inflation == False, use max congestion as inflation ratio.
    #       2) We use mov_node_size instead of mov_node_size_real to cover more GR Grids.
    this_mov_conn_inflate_ratio = routeforce.inflate_ratio(
        mov_node_pos[mov_lhs:filler_lhs],
        mov_node_size[mov_lhs:filler_lhs],
        mov_node_weight[mov_lhs:filler_lhs],
        torch.ones_like(mov_node_weight[mov_lhs:filler_lhs]),  # use expand ratio == 1
        inflate_mat,
        1.0, unit_len_x, unit_len_y, num_bin_x, num_bin_y, use_weighted_inflation
    )
    if hv_same_ratio:
        this_mov_conn_inflate_ratio.sqrt_()
    else:
        this_mov_conn_inflate_ratio.sqrt_()

    # 4.1) if the remaining space is not enough, scale down the movable cell inflation ratio
    expect_new_mov_area = torch.prod(this_mov_conn_inflate_ratio * mov_node_size_real[mov_lhs:filler_lhs], 1)
    inc_mov_area = expect_new_mov_area - last_mov_area
    inc_mov_area_total = inc_mov_area.sum().item()
    inc_area_scale = max_inc_area_total / inc_mov_area_total
    if inc_area_scale < 1:
        logger.warning("Not enough space to inflate. Scale down.")
        inc_mov_area *= inc_area_scale
        inc_mov_area_total *= inc_area_scale
        new_mov_area = inc_mov_area + last_mov_area
        size_scale = (new_mov_area / expect_new_mov_area).sqrt_().unsqueeze_(1)
        this_mov_conn_inflate_ratio *= size_scale
    else:
        new_mov_area = expect_new_mov_area
    # 4.2) update total mov inflation ratio
    new_mov_node_size_real: torch.Tensor = mov_node_size_real.clone()
    new_mov_node_size_real[mov_lhs:filler_lhs].mul_(this_mov_conn_inflate_ratio)

    if inc_mov_area_total / last_mov_area_total < min_area_inc:
        logger.warning(
            "Too small relative area increment (%.4f < %.4f). Early terminate cell inflation." % (
                inc_mov_area_total / last_mov_area_total, min_area_inc
        ))
        ps.use_cell_inflate = False  # not inflation anymore
        return gr_metrics, None, None

    # 5) update total filler inflation ratio
    new_mov_area_total = last_mov_area_total + inc_mov_area_total
    new_filler_area_total = 0.0
    last_filler_area_total = torch.sum(torch.prod(mov_node_size_real[filler_lhs:filler_rhs], 1)).item()
    filler_scale = 0.0
    if new_mov_area_total + last_filler_area_total > route_cache.target_area:
        new_filler_area_total = max(route_cache.target_area - new_mov_area_total, 0)
        if decrease_target_density and new_filler_area_total / route_cache.placeable_area > 0.2:
            # remove some pre-inserted fillers / FloatMov nodes to decrease the target density
            new_target_density = max(0.8, 0.85 * 1.0)
            new_target_area = new_target_density * route_cache.placeable_area
            new_filler_area_total = max(new_target_area - new_mov_area_total, 0)
            filler_scale = new_filler_area_total / route_cache.original_filler_area_total
            original_num_fillers = route_cache.original_num_fillers
            num_remain_cells = filler_lhs + math.ceil(filler_scale * original_num_fillers)

            route_cache.target_area = new_target_area
            new_mov_node_size_real = new_mov_node_size_real[:num_remain_cells]
            mov_node_size_real = mov_node_size_real[:num_remain_cells]
            logger.warning("Remove nodes to reduce target density from %.4f to %.4f. \
                        This step may remove some FloatMov. #Nodes from %d to %d" % 
                        (old_target_density, args.target_density, filler_rhs, num_remain_cells))
        elif route_cache.original_filler_area_total > 0:
            filler_scale = math.sqrt(new_filler_area_total / route_cache.original_filler_area_total)
            new_mov_node_size_real[filler_lhs:filler_rhs] = filler_scale * ori_mov_node_size_real[filler_lhs:filler_rhs]
    else:
        new_filler_area_total = last_filler_area_total

    # pin rel cpos should be scaled by real inflate_ratio
    inflate_ratio = new_mov_node_size_real / mov_node_size_real
    new_pin_rel_cpos = routeforce.inflate_pin_rel_cpos(
        inflate_ratio,
        route_cache.original_pin_rel_cpos,
        data.pin_id2node_id,
        mov_rhs - filler_lhs
    )

    mov_node_size_real.copy_(new_mov_node_size_real)
    data.pin_rel_cpos.copy_(new_pin_rel_cpos)
    old_target_density = copy.deepcopy(args.target_density)
    args.target_density = (new_mov_area_total + new_filler_area_total) / route_cache.placeable_area
    logger.info("Update target density from %.4f to %.4f" % (old_target_density, args.target_density))

    logger.info("Relative area | increment: %.4f, mov: %.4f, filler: %.4f, all_cells: %.4f" % (
        inc_mov_area_total / last_mov_area_total,
        new_mov_area_total / last_mov_area_total,
        new_filler_area_total / last_filler_area_total if last_filler_area_total > 1e-5 else 0,
        (new_mov_area_total + new_filler_area_total) / (last_mov_area_total + last_filler_area_total)
    ))
    logger.info("Inflation Rate | movable: avgX/maxX %.4f/%.4f avgY/maxY %.4f/%.4f, filler: %.4f" % (
        inflate_ratio[mov_lhs:filler_lhs, 0].mean().item(), 
        inflate_ratio[mov_lhs:filler_lhs, 0].max().item(),
        inflate_ratio[mov_lhs:filler_lhs, 1].mean().item(), 
        inflate_ratio[mov_lhs:filler_lhs, 1].max().item(),
        filler_scale
    ))

    if args.target_density * route_cache.placeable_area - torch.sum(torch.prod(mov_node_size_real, 1)).item() < 0:
        logger.warning("Please check inflation...")
        logger.info("new_total_mov_area: %.1f ori_total_mov_area: %.1f placeable_area: %.1f target_density: %.2f placeable_area * target_density: %.1f" % (
            torch.sum(torch.prod(mov_node_size_real, 1)).item(),
            torch.sum(torch.prod(ori_mov_node_size, 1)).item(),
            route_cache.placeable_area,
            args.target_density,
            args.target_density * route_cache.placeable_area
        ))
    
    route_cache.mov_node_size_real = mov_node_size_real
    data.mov_node_area = torch.prod(mov_node_size_real, 1).unsqueeze_(1)

    mov_node_size = mov_node_size_real
    if args.clamp_node:
        mov_node_area = torch.prod(mov_node_size, 1)
        clamp_mov_node_size = mov_node_size.clamp(min=data.unit_len * math.sqrt(2))
        clamp_mov_node_area = torch.prod(clamp_mov_node_size, 1)
        # update
        expand_ratio = mov_node_area / clamp_mov_node_area
        mov_node_size = clamp_mov_node_size

    ps.use_cell_inflate = True
    return gr_metrics, mov_node_size, expand_ratio


def route_inflation_roll_back(args, logger, data, mov_node_size):
    if not route_cache.first_run:
        mov_lhs, mov_rhs = data.movable_index
        mov_node_size[mov_lhs:mov_rhs].copy_(route_cache.original_mov_node_size[mov_lhs:mov_rhs])
        data.pin_rel_cpos.copy_(route_cache.original_pin_rel_cpos)
        args.target_density = route_cache.original_target_density
    route_cache.reset()
