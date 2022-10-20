from utils import *
from src import *
import torch.optim

def run_placement_main_adam(args, logger):
    data, rawdb, gpdb = load_dataset(args, logger)
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    data = data.to(device)
    gp_start_time = time.time()
    logger.info("start gp")
    data = data.preprocess()
    logger.info(data)
    logger.info(data.node_type_indices)

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

    # optimizer = torch.optim.SGD(
    #     [mov_node_pos],
    #     lr=args.lr,
    #     momentum=0.9,
    #     nesterov=True,
    # )
    optimizer = torch.optim.Adam(
        [mov_node_pos],
        lr=args.lr,
    )
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

    init_params(mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
        density_map_layer, mov_node_size, init_density_map, optimizer, ps, data, args)

    # fix_lhs, fix_rhs = data.fixed_index
    # info = (0, 0, data.design_name + "_fix")
    # fix_node_pos = data.node_pos[fix_lhs:fix_rhs, ...]
    # fix_node_size = data.node_size[fix_lhs:fix_rhs, ...]
    # draw_fig_with_cairo(
    #     None, None, fix_node_pos, fix_node_size, None, None, data, info, args
    # )

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
    #             if mov_node_pos.grad is not None:
    #                 mov_node_pos.grad.zero_()
    #             hpwl, overflow, mov_node_pos = fast_optimization(
    #                 mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
    #                 density_map_layer, mov_node_size, init_density_map, ps, data, args
    #             )
    #             optimizer.step()
    #             ps.step(hpwl, overflow, mov_node_pos, data)
    #             p.step()
    # exit(0)
    for iteration in range(args.inner_iter):
        optimizer.zero_grad()
        hpwl, overflow, mov_node_pos = fast_optimization(
            mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
            density_map_layer, mov_node_size, init_density_map, ps, data, args
        )
        optimizer.step()
        # update parameters
        ps.step(hpwl, overflow, mov_node_pos, data)
        if ps.need_to_early_stop():
            break
        if iteration % args.log_freq == 0 or iteration == args.inner_iter - 1:
            log_str = (
                "iter: %d | masked_hpwl: %.2E overflow: %.4f "
                "density_weight: %.4E wa_coeff: %.4E"
                % (
                    iteration,
                    hpwl.item(),
                    overflow.item(),
                    ps.density_weight,
                    ps.wa_coeff,
                )
            )

            logger.info(log_str)
            if args.draw_placement:
                info = (iteration, hpwl, data.design_name)
                # draw_fig_new(conn_node_pos.detach(), info, args)

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
    gp_end_time = time.time()
    gp_time = gp_end_time - gp_start_time
    logger.info("GP Stop! #Iters %d masked_hpwl: %.4E overflow: %.4f GP Time: %.4fs perIterTime: %.6fs" % 
        (iteration, hpwl, overflow, gp_time, gp_time / (iteration + 1))
    )

    # Eval
    hpwl, overflow = evaluate_placement(
        node_pos, density_map_layer, init_density_map, data, args
    )
    hpwl, overflow = hpwl.item(), overflow.item()
    info = (iteration + 1, hpwl, data.design_name)
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
        exact_node_pos = torch.round(node_pos * data.die_scale + data.die_shift).cpu()
        gpdb.apply_node_pos(exact_node_pos)
        gpdb.write_placement(gp_prefix)
        logger.info("Write global placement in %s" % gp_prefix)

    dp_start_time = time.time()
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
            output = os.popen(cmd).read()
            dp_hpwl = float(output.split("========\n         HPWL=")[1].split("Time")[0].strip())
            dp_out_file = dp_out_file + ".ntup%s" % post_fix
        elif args.dp_engine == "rippledp":
            dp_out_file = gp_out_file.replace("_gp", "")

            dp_engine = "./thirdparty/placers/ripple/bin/placer"
            aux_input = data.dataset_path["aux"]
            MLLMaxDensity = int(round(args.target_density * 1000.0))
            cmd = "%s -flow dac2016 -bookshelf ispd2005 -aux %s -pl %s -MLLMaxDensity %s -cpu %s -output %s" % (
                dp_engine, aux_input, gp_out_file, MLLMaxDensity, args.num_threads, dp_out_file)
            os.system(cmd)
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
            output = os.popen(cmd).read()
            # consider site_width
            dp_hpwl = float(output.split("=======\n         HPWL=")[1].split("Time")[0].strip())
            top5overflow = float(output.split("[CONG] Top 5 Overflow")[1].split("\n")[0].strip())
        else:
            raise NotImplementedError("DP Engine %s unsupported" % args.dp_engine)
        logger.info("External detailed placement takes %.2f seconds" %
                        (time.time() - dp_start_time))
        logger.info("Write detail placement in %s" % dp_out_file)

        dp_end_time = time.time()

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

    return dp_hpwl, gp_hpwl, top5overflow, overflow, gp_time, dp_time
