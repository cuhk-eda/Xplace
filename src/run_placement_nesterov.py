from utils import *
from src import *
from functools import partial
from FNO import FNO2d

def get_trunc_node_pos_fn(mov_node_size, data):
    node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4 
    node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4
    def trunc_node_pos_fn(x):
        x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
        return x
    return trunc_node_pos_fn

def run_placement_main_nesterov(args, logger):
    total_start = time.time()
    data, data_nn, rawdb, gpdb = load_dataset(args, logger)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    assert args.use_eplace_nesterov
    logger.info("Use Nesterov optimizer!")
    if args.scale_design:
        logger.warning("Eplace's nesterov optimizer cannot support normalized die. Disable scale_design.")
        args.scale_design = False
    data = data.to(device)
    data = data.preprocess()
    logger.info(data)
    logger.info(data.node_type_indices)
    # args.num_bin_x = args.num_bin_y = 2 ** math.ceil(math.log2(max(data.die_info).item() // 25))

    init_density_map = get_init_density_map(data, args, logger)
    data.init_filler()
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos, mov_node_size, expand_ratio = data.get_mov_node_info()
    mov_node_pos = mov_node_pos.requires_grad_(True)

    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)

    conn_fix_node_pos = data.node_pos.new_empty(0, 2)
    if data.fixed_connected_index[0] < data.fixed_connected_index[1]:
        lhs, rhs = data.fixed_connected_index
        conn_fix_node_pos = data.node_pos[lhs:rhs, ...]
    conn_fix_node_pos = conn_fix_node_pos.detach()

    def overflow_fn(mov_density_map):
        overflow_sum = ((mov_density_map - args.target_density) * data.bin_area).clamp_(min=0.0).sum()
        return overflow_sum / data.total_mov_area_without_filler
    overflow_helper = (mov_lhs, mov_rhs, overflow_fn)

    ps = ParamScheduler(data, args, logger)
    density_map_layer = ElectronicDensityLayer(
        unit_len=data.unit_len,
        num_bin_x=data.num_bin_x,
        num_bin_y=data.num_bin_y,
        device=device,
        overflow_helper=overflow_helper,
        sorted_maps=data.sorted_maps,
        expand_ratio=expand_ratio,
        deterministic=args.deterministic,
    ).to(device)

    # assert args.nn_size == 256
    model_path = args.model_path
    width, neck, modes = [int(i) for i in model_path.split("/")[-1].split("_")[1].split("x")]
    
    nn_bin = args.nn_bin
    args_nn = copy.deepcopy(args)
    args_nn.clamp_node = args.nn_expand   
    args_nn.num_bin_y = nn_bin
    args_nn.num_bin_x = nn_bin
    data_nn.__args__ = args_nn
    data_nn.__num_bin_x__ = args_nn.num_bin_x
    data_nn.__num_bin_y__ = args_nn.num_bin_y
    data_nn = data_nn.to(device)
    data_nn = data_nn.preprocess()
    init_density_map_nn = get_init_density_map(data_nn, args_nn, logger)
    data_nn.init_filler()
    if data.filler_size != None:
        data_nn.filler_size = data.filler_size.clone()
    else:
        data_nn.filler_size = None
    data_nn.__num_fillers__ = data.__num_fillers__
    _, mov_node_size_nn, expand_ratio_nn = data_nn.get_mov_node_info()
    def overflow_fn_nn(mov_density_map):
        overflow_sum = ((mov_density_map - args.target_density) * data_nn.bin_area).clamp_(min=0.0).sum()
        return overflow_sum / data.total_mov_area_without_filler
    overflow_helper_nn = (mov_lhs, mov_rhs, overflow_fn_nn)
    density_map_layer_nn = ElectronicDensityLayer(
        unit_len=data_nn.unit_len,
        num_bin_x=nn_bin,
        num_bin_y=nn_bin,
        device=device,
        overflow_helper=overflow_helper_nn,
        sorted_maps=data.sorted_maps,   #FIXME: !!!
        expand_ratio=expand_ratio_nn,
        deterministic=args.deterministic,
        scale_w_k=False,
    ).to(device)
    with torch.no_grad():
        model = FNO2d(modes, modes, width, neck).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        torch.cuda.synchronize()
        model(torch.randn((1, nn_bin, nn_bin, 1), device=device)) # warmup
    # fix_lhs, fix_rhs = data.fixed_index
    # info = (0, 0, data.design_name + "_fix")
    # fix_node_pos = data.node_pos[fix_lhs:fix_rhs, ...]
    # fix_node_size = data.node_size[fix_lhs:fix_rhs, ...]
    # draw_fig_with_cairo(
    #     None, None, fix_node_pos, fix_node_size, None, None, data, info, args
    # )

    obj_and_grad_fn = partial(
        calc_obj_and_grad_nn,
        constraint_fn=trunc_node_pos_fn,
        mov_node_size=mov_node_size,
        mov_node_size_nn=mov_node_size_nn,
        init_density_map=init_density_map,
        init_density_map_nn=init_density_map_nn,
        density_map_layer=density_map_layer,
        density_map_layer_nn=density_map_layer_nn,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
        model=model,
    )

    evaluator_fn = partial(
        fast_evaluator,
        constraint_fn=trunc_node_pos_fn,
        mov_node_size=mov_node_size,
        init_density_map=init_density_map,
        density_map_layer=density_map_layer,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
    )
    optimizer = NesterovOptimizer([mov_node_pos], lr=0)

    # initialization
    init_params(
        mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
        density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer, 
        ps, data, args
    )
    # init learnig rate
    init_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr.item()

    enable_nns = []
    nn_weights = []
    force_ratios = []

    torch.cuda.synchronize(device)
    gp_start_time = time.time()
    logger.info("start gp")

    terminate_signal = False
    for iteration in range(args.inner_iter):
        # optimizer.zero_grad() # zero grad inside obj_and_grad_fn
        obj = optimizer.step(obj_and_grad_fn)
        hpwl, overflow = evaluator_fn(mov_node_pos)
        # for nn tuning
        nn_weights.append(ps.nn_sigma.item() if type(ps.nn_sigma) == torch.Tensor else ps.nn_sigma)
        enable_nns.append((ps.weighted_weight < args.ps_end).item() and ps.iter > args.ps_end_iter)
        force_ratios.append(ps.force_ratio.item() if type(ps.force_ratio) == torch.Tensor else ps.force_ratio)
        # update parameters
        ps.step(hpwl, overflow, mov_node_pos, data)
        if ps.need_to_early_stop():
            terminate_signal = True

        if iteration % args.log_freq == 0 or iteration == args.inner_iter - 1 or ps.rerun_route:
            log_str = (
                "iter: %d | masked_hpwl: %.2E overflow: %.4f obj: %.4E "
                "density_weight: %.4E wa_coeff: %.4E"
                % (
                    iteration,
                    hpwl,
                    overflow,
                    obj,
                    ps.density_weight,
                    ps.wa_coeff,
                )
            )
            logger.info(log_str)
            if args.draw_placement:
                info = (iteration, hpwl, data.design_name)
                node_pos_to_draw = mov_node_pos[mov_lhs:mov_rhs, ...].clone()
                node_size_to_draw = data.node_size[mov_lhs:mov_rhs, ...].clone()
                node_pos_to_draw = torch.cat(
                    [node_pos_to_draw, data.node_pos[mov_rhs:, ...].clone()], dim=0
                )
                node_size_to_draw = torch.cat(
                    [node_size_to_draw, data.node_size[mov_rhs:, ...].clone()], dim=0
                )
                if args.use_filler:
                    node_pos_to_draw = torch.cat(
                        [node_pos_to_draw, mov_node_pos[mov_rhs:, ...].clone()], dim=0
                    )
                    node_size_filler_to_draw = data.filler_size[:(mov_node_pos.shape[0] - mov_rhs), ...]
                    node_size_to_draw = torch.cat(
                        [node_size_to_draw, node_size_filler_to_draw], dim=0
                    )
                draw_fig_with_cairo_cpp(
                    node_pos_to_draw, node_size_to_draw, data, info, args
                )

        if terminate_signal:
            break

    # Save best solution
    best_res = ps.get_best_solution()
    if best_res[0] is not None:
        best_sol, hpwl, overflow = best_res
        # fillers are unused from now, we don't copy there data
        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol[mov_lhs:mov_rhs])
    if ps.enable_route:
        route_inflation_roll_back(args, logger, data, mov_node_size)
        ps.rerun_route = True
        gr_metrics = run_gr_and_fft_main(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos, constraint_fn=trunc_node_pos_fn, 
            skip_m1_route=True, report_gr_metrics_only=True
        )
        ps.rerun_route = False
        ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
        best_sol_gr = ps.get_best_gr_sol()
        mov_node_pos[mov_lhs:mov_rhs].data.copy_(best_sol_gr[mov_lhs:mov_rhs])

    node_pos = mov_node_pos[mov_lhs:mov_rhs]
    node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
    torch.cuda.synchronize(device)
    gp_end_time = time.time()
    gp_time = gp_end_time - gp_start_time
    gp_per_iter = gp_time / (iteration + 1)
    logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f GP Time: %.4fs perIterTime: %.6fs" % 
        (iteration, hpwl, overflow, gp_time, gp_time / (iteration + 1))
    )

    plt.figure()
    plt.plot(np.linspace(1, len(enable_nns), len(enable_nns)), enable_nns, label="use_nn")
    plt.plot(np.linspace(1, len(nn_weights), len(nn_weights)), nn_weights, label="sigma")
    plt.plot(np.linspace(1, len(ps.recorder.weighted_weight), len(ps.recorder.weighted_weight)), ps.recorder.weighted_weight, label="omega")
    plt.plot(np.linspace(1, len(force_ratios), len(force_ratios)), force_ratios, label="force_ratio(r)")
    plt.legend()
    plt.savefig(os.path.join(args.result_dir, args.exp_id, "%s_nn_tuned.png"%(args.design_name)))
    plt.close()

    # Eval
    hpwl, overflow = evaluate_placement(
        node_pos, density_map_layer, init_density_map, data, args
    )
    hpwl, overflow = hpwl.item(), overflow.item()
    info = (iteration + 1, hpwl, data.design_name)
    if args.draw_placement:
        draw_fig_with_cairo_cpp(node_pos, data.node_size, data, info, args)
    logger.info("After GP, best solution eval, exact HPWL: %.4E exact Overflow: %.4f" % (hpwl, overflow))
    ps.visualize(args, logger)
    gp_hpwl = hpwl
    gp_time = gp_end_time - gp_start_time
    iteration += 1 # increase 1 For DP drawing

    # detail placement
    node_pos, dp_hpwl, top5overflow, lg_time, dp_time = detail_placement_main(
        node_pos, gpdb, rawdb, ps, data, args, logger
    )
    iteration += 1

    route_metrics = None
    if ps.enable_route and args.final_route_eval:
        logger.info("Final routing evalution by GGR...")
        route_metrics = run_gr_and_fft(
            args, logger, data, rawdb, gpdb, ps, 
            report_gr_metrics_only=True,
            skip_m1_route=True, given_gr_params={
                "rrrIters": 1,
                "route_guide": os.path.join(args.result_dir, args.exp_id, args.output_dir, "%s_%s.guide" %(args.output_prefix, args.design_name)),
            }
        )

    if args.load_from_raw:
        del gpdb, rawdb

    place_time = time.time() - total_start
    logger.info("GP Time: %.4f LG Time: %.4f DP Time: %.4f Total Place Time: %.4f" % (
        gp_time, lg_time, dp_time, place_time))
    place_metrics = (dp_hpwl, gp_hpwl, top5overflow, overflow, gp_time, dp_time + lg_time, gp_per_iter, place_time)

    return place_metrics, route_metrics