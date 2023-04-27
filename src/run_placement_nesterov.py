from utils import *
from src import *
from functools import partial

def get_trunc_node_pos_fn(mov_node_size, data):
    node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4 
    node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4
    def trunc_node_pos_fn(x):
        x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
        return x
    return trunc_node_pos_fn

def run_placement_main_nesterov(args, logger):
    total_start = time.time()
    data, rawdb, gpdb = load_dataset(args, logger)
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

    init_density_map = get_init_density_map(rawdb, gpdb, data, args, logger)
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

    # fix_lhs, fix_rhs = data.fixed_index
    # info = (0, 0, data.design_name + "_fix")
    # fix_node_pos = data.node_pos[fix_lhs:fix_rhs, ...]
    # fix_node_size = data.node_size[fix_lhs:fix_rhs, ...]
    # draw_fig_with_cairo(
    #     None, None, fix_node_pos, fix_node_size, None, None, data, info, args
    # )
    def calc_route_force(mov_node_pos, mov_node_size, expand_ratio, constraint_fn):
        return get_route_force(
            args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
            constraint_fn=constraint_fn
        )

    obj_and_grad_fn = partial(
        calc_obj_and_grad,
        constraint_fn=trunc_node_pos_fn,
        route_fn=calc_route_force,
        mov_node_size=mov_node_size,
        expand_ratio=expand_ratio,
        init_density_map=init_density_map,
        density_map_layer=density_map_layer,
        conn_fix_node_pos=conn_fix_node_pos,
        ps=ps,
        data=data,
        args=args,
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
        ps, data, args, route_fn=calc_route_force
    )
    # init learnig rate
    init_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr.item()

    torch.cuda.synchronize(device)
    gp_start_time = time.time()
    logger.info("start gp")

    # def trace_handler(prof):
    #     print(prof.key_averages().table(
    #         sort_by="self_cuda_time_total", row_limit=-1))
    #     prof.export_chrome_trace("test_trace_" + str(prof.step_num) + ".json")
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ], schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=2,
    #         active=2),
    #     on_trace_ready=trace_handler
    #     ) as p:
    #         for iter in range(6):
    #             # optimizer.zero_grad()
    #             obj = optimizer.step(obj_and_grad_fn)
    #             hpwl, overflow = evaluator_fn(mov_node_pos)
    #             # update parameters
    #             ps.step(hpwl, overflow, mov_node_pos, data)
    #             if ps.need_to_early_stop():
    #                 break
    #             p.step()
    # exit(0)
    terminate_signal = False
    for iteration in range(args.inner_iter):
        # optimizer.zero_grad() # zero grad inside obj_and_grad_fn
        obj = optimizer.step(obj_and_grad_fn)
        hpwl, overflow = evaluator_fn(mov_node_pos)
        # update parameters
        ps.step(hpwl, overflow, mov_node_pos, data)
        if ps.need_to_early_stop():
            terminate_signal = True

        if ps.use_cell_inflate and ps.curr_optimizer_cnt < ps.max_route_opt and terminate_signal:
            terminate_signal = False  # reset signal
            ps.start_route_opt = True
            ps.curr_optimizer_cnt += 1
            best_res = ps.get_best_solution()
            if best_res[0] is not None:
                best_sol, hpwl, overflow = best_res
                mov_node_pos.data.copy_(best_sol)

        if ps.use_route_force:
            if ps.iter > 100 and ps.enable_route:
                if ps.recorder.overflow[-1] < 0.2 and ps.recorder.overflow[-2] >= 0.2 and not ps.start_route_opt:
                    ps.start_route_opt = True
                    ps.curr_optimizer_cnt += 1
                # if ps.recorder.overflow[-2] < 0.2 and ps.recorder.overflow[-1] >= 0.2 and ps.start_route_opt:
                #     ps.start_route_opt = False
                #     ps.curr_optimizer_cnt += 1

        # if ps.use_cell_inflate and ps.curr_optimizer_cnt < ps.max_route_opt:
        #     if ps.iter > 100 and ps.enable_route:
        #         if ps.recorder.overflow[-1] < 0.15 and ps.recorder.overflow[-2] >= 0.15 and not ps.start_route_opt:
        #             ps.start_route_opt = True
        #             ps.curr_optimizer_cnt += 1
        #         if ps.recorder.overflow[-2] < 0.15 and ps.recorder.overflow[-1] >= 0.15 and ps.start_route_opt:
        #             ps.start_route_opt = False

        if ps.start_route_opt and ps.enable_route:
            if (ps.iter % args.route_freq == 0 and ps.use_route_force) or \
                  (ps.curr_optimizer_cnt != ps.prev_optimizer_cnt and ps.curr_optimizer_cnt <= ps.max_route_opt):
                ps.rerun_route = True
            else:
                ps.rerun_route = False
        else:
            ps.rerun_route = False

        if ps.rerun_route:
            new_mov_node_size, new_expand_ratio = None, None
            if ps.use_cell_inflate:
                output = route_inflation(
                    args, logger, data, rawdb, gpdb, ps, mov_node_pos, mov_node_size, expand_ratio,
                    constraint_fn=trunc_node_pos_fn,
                )  # ps.use_cell_inflate is updated in route_inflation
                if not ps.use_cell_inflate:
                    logger.info("Stop cell inflation...")
                if output is not None:
                    gr_metrics, new_mov_node_size, new_expand_ratio = output
                    ps.push_gr_sol(gr_metrics, hpwl, overflow, mov_node_pos)
            route_fn=None
            if ps.use_route_force:
                route_fn=calc_route_force
            ps.prev_optimizer_cnt = ps.curr_optimizer_cnt
            if ps.use_cell_inflate or ps.use_route_force:
                logger.info("Reset optimizer...")
                if new_mov_node_size is not None:
                    # remove some fillers, we should update the size the pos
                    mov_node_size = new_mov_node_size
                    mov_node_pos = mov_node_pos[:new_mov_node_size.shape[0]].detach().clone()
                    mov_node_pos = mov_node_pos.requires_grad_(True)
                    # update expand ratio and precondition relevant data
                    expand_ratio = new_expand_ratio  # already update in route_inflation()
                    data.mov_node_to_num_pins = data.mov_node_to_num_pins[:new_mov_node_size.shape[0]]
                    data.mov_node_area = data.mov_node_area  # already update in route_inflation()
                    # update partial function correspondingly
                    trunc_node_pos_fn = get_trunc_node_pos_fn(mov_node_size, data)
                    obj_and_grad_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    obj_and_grad_fn.keywords["mov_node_size"] = mov_node_size
                    obj_and_grad_fn.keywords["expand_ratio"] = expand_ratio
                    evaluator_fn.keywords["constraint_fn"] = trunc_node_pos_fn
                    evaluator_fn.keywords["mov_node_size"] = mov_node_size
                    density_map_layer.expand_ratio = expand_ratio
                # reset nesterov optimizer
                optimizer = NesterovOptimizer([mov_node_pos], lr=0)
                init_params(
                    mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
                    density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer,
                    ps, data, args, route_fn=route_fn
                )
                cur_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = cur_lr.item()
                logger.info(
                    "Route Iter: %d | lr: %.2E density_weight: %.2E route_weight: %.2E "
                    "congest_weight: %.2E pseudo_weight: %.2E " 
                    % (
                        ps.curr_optimizer_cnt - 1,
                        cur_lr.item(),
                        ps.density_weight,
                        ps.route_weight,
                        ps.congest_weight,
                        ps.pseudo_weight,
                    )
                )
                ps.reset_best_sol()

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