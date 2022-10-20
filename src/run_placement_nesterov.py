from utils import *
from src import *
from functools import partial
from FNO import FNO2d

def run_placement_main_nesterov(args, logger):
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

    init_density_map = get_init_density_map(data, args, logger)
    data.init_filler()
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos, mov_node_size, expand_ratio = data.get_mov_node_info()

    mov_node_pos = mov_node_pos.requires_grad_(True)
    node_pos_lb = mov_node_size / 2 + data.die_ll + 1e-4 
    node_pos_ub = data.die_ur - mov_node_size / 2 + data.die_ll - 1e-4
    def trunc_node_pos_fn(x):
        x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
        return x

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
    ).to(device)

    # assert args.nn_size == 256
    model_path = args.model_path
    width, neck, modes = [int(i) for i in model_path.split("/")[-1].split("_")[1].split("x")]
    
    nn_bin = args.nn_bin
    args_tmp = copy.deepcopy(args)
    args_tmp.clamp_node = args.nn_expand   
    args_tmp.num_bin_y = nn_bin
    args_tmp.num_bin_x = nn_bin
    data_nn, _, _ = load_dataset(args_tmp, logger)
    data_nn = data_nn.to(device)
    data_nn = data_nn.preprocess()
    init_density_map_nn = get_init_density_map(data_nn, args_tmp, logger)
    data_nn.init_filler()
    if data.filler_size != None:
        data_nn.filler_size = data.filler_size.clone()
    else: data_nn.filler_size=None
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
        scale_w_k=False,
    ).to(device)
    with torch.no_grad():
        model = FNO2d(modes, modes, width, neck).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

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
    optimizer = NesterovOptimizer(
        [mov_node_pos],
        lr=0,
    )

    # initialization
    init_params(
        mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
        density_map_layer, mov_node_size, init_density_map, optimizer, ps, data, args
    )
    # init learnig rate
    init_lr = estimate_initial_learning_rate(obj_and_grad_fn, trunc_node_pos_fn, mov_node_pos, args.lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = init_lr.item()

    enable_nns = []
    nn_weights = []
    force_ratios = []

    torch.cuda.synchronize()
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
            break
        if iteration % args.log_freq == 0 or iteration == args.inner_iter - 1:
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
                node_size_to_draw = mov_node_size[mov_lhs:mov_rhs, ...].clone()
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
                    node_size_to_draw = torch.cat(
                        [node_size_to_draw, mov_node_size[mov_rhs:, ...].clone()], dim=0
                    )
                draw_fig_with_cairo_cpp(
                    node_pos_to_draw, node_size_to_draw, data, info, args
                )

    # Save best solution
    best_res = ps.get_best_solution()
    if best_res[0] is not None:
        best_sol, hpwl, overflow = best_res
        mov_node_pos.data.copy_(best_sol)
    node_pos = mov_node_pos[mov_lhs:mov_rhs]
    node_pos = torch.cat([node_pos, data.node_pos[mov_rhs:]], dim=0)
    torch.cuda.synchronize()
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
    iteration += 1 # increase 1 For DP drawing

    # Write placement
    if args.write_placement and args.load_from_raw:
        res_root = os.path.join(args.result_dir, args.exp_id)
        gp_prefix = os.path.join(res_root, args.output_dir, "%s_%s_gp" %(args.output_prefix, args.design_name))
        if not os.path.exists(os.path.dirname(gp_prefix)):
            os.makedirs(os.path.dirname(gp_prefix))
        start_write_time = time.time()
        if data.dataset_format == "lefdef":
            exact_node_pos = torch.round(node_pos * data.die_scale + data.die_shift).cpu()
            gpdb.apply_node_pos(exact_node_pos)
            gpdb.write_placement(gp_prefix)
        elif data.dataset_format == "bookshelf":
            logger.info("Use python to generate .pl file")
            data.write_pl(node_pos, gp_prefix)
        else:
            raise NotImplementedError("Dataset format %s unsupported" % data.dataset_format)
        logger.info("Write global placement in %s. Time: %.4f" % (gp_prefix, time.time() - start_write_time))

    dp_start_time = None
    dp_end_time = None
    dp_hpwl = -1
    top5overflow = -1
    if args.detail_placement and args.load_from_raw:
        # TODO: too ugly...
        post_fix = None
        if data.dataset_format == "lefdef":
            post_fix = ".def"
        elif data.dataset_format == "bookshelf":
            post_fix = ".pl"
        gp_out_file = gp_prefix + post_fix
        if args.dp_engine == "ntuplace3":
            dp_out_file = gp_out_file.replace("_gp%s" % post_fix, "")

            dp_engine = "./thirdparty/placers/ntuplace3/ntuplace3"
            aux_input = data.dataset_path["aux"]
            target_density_cmd = ""
            if args.target_density < 1.0:
                target_density_cmd = " -util %f" % (args.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal" % (
                dp_engine, aux_input, gp_out_file, target_density_cmd, dp_out_file)
            logger.info(cmd)
            # os.system(cmd)
            dp_start_time = time.time()
            output = os.popen(cmd).read()
            dp_end_time = time.time()
            dp_hpwl = float(output.split("========\n         HPWL=")[1].split("Time")[0].strip())
            dp_out_file = dp_out_file + ".ntup%s" % post_fix
        elif args.dp_engine == "rippledp":
            dp_out_file = gp_out_file.replace("_gp", "")

            dp_engine = "./thirdparty/placers/ripple/bin/placer"
            aux_input = data.dataset_path["aux"]
            MLLMaxDensity = int(round(args.target_density * 1000.0))
            cmd = "%s -flow dac2016 -bookshelf ispd2005 -aux %s -pl %s -MLLMaxDensity %s -cpu %s -output %s" % (
                dp_engine, aux_input, gp_out_file, MLLMaxDensity, args.num_threads, dp_out_file)
            dp_start_time = time.time()
            os.system(cmd)
            dp_end_time = time.time()
        elif args.dp_engine == 'ntuplace_4dr':
            dp_out_file = gp_out_file.replace(".gp.def", "")
            dp_engine = "./thirdparty/placers/ntuplace4dr/ntuplace4dr_binary/placer"
            cmd = dp_engine
            tech_lef = data.dataset_path["tech_lef"]
            cell_lef = data.dataset_path["cell_lef"]
            cmd += " -tech_lef %s" % tech_lef
            cmd += " -cell_lef %s" % cell_lef
            benchmark_dir = os.path.dirname(tech_lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (benchmark_dir)
            cmd += " -noglobal; "
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logger.info("%s" % (cmd))
            dp_start_time = time.time()
            output = os.popen(cmd).read()
            dp_end_time = time.time()
            # consider site_width
            dp_hpwl = float(output.split("=======\n         HPWL=")[1].split("Time")[0].strip())
            top5overflow = float(output.split("[CONG] Top 5 Overflow")[1].split("\n")[0].strip())
        else:
            raise NotImplementedError("DP Engine %s unsupported" % args.dp_engine)
        logger.info("External detailed placement takes %.2f seconds" %
                        (dp_end_time - dp_start_time))
        logger.info("After DP, HPWL: %.4E" % dp_hpwl)
        logger.info("Write detail placement in %s" % dp_out_file)

        del gpdb, rawdb
        # logger.info("Evaluating detail placement result...")
        # data, rawdb, gpdb = load_dataset(args, logger, dp_out_file)
        # data = data.to(device).preprocess()
        # hpwl, overflow = evaluate_placement(
        #     data.node_pos, density_map_layer, init_density_map, data, args
        # )
        # hpwl, overflow = hpwl.item(), overflow.item()
        # info = (iteration + 1, hpwl, data.design_name)
        # draw_fig_with_cairo_cpp(data.node_pos, data.node_size, data, info, args)
        # logger.info("After DP, HPWL: %.4E Overflow: %.4f" % (hpwl, overflow))

    gp_time = gp_end_time - gp_start_time
    dp_time = dp_end_time - dp_start_time if dp_end_time is not None else 0.0
    logger.info("GP Time: %.4f DP Time: %.4f" % (gp_time, dp_time))

    return dp_hpwl, gp_hpwl, top5overflow, overflow, gp_time, dp_time, gp_per_iter