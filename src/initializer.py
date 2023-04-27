import torch
from .database import PlaceData
from cpp_to_py import density_map_cuda
from .core import WAWirelengthLossAndHPWL
from .calculator import calc_grad


def get_init_density_map(rawdb, gpdb, data: PlaceData, args, logger):
    lhs, rhs = data.fixed_index
    device = data.node_size.get_device()
    dtype = data.node_size.dtype
    zeros_density_map = torch.zeros(
        (data.num_bin_x, data.num_bin_y), device=device, dtype=dtype,
    )
    if lhs == rhs:
        data.init_density_map = zeros_density_map
        return zeros_density_map
    # get fix nodes which are located inside die
    node_pos = data.node_pos[lhs:rhs]
    node_size = data.node_size[lhs:rhs]
    node_weight = node_size.new_ones(node_size.shape[0])
    init_density_map = density_map_cuda.forward_naive(
        node_pos, node_size, node_weight, data.unit_len, zeros_density_map,
        data.num_bin_x, data.num_bin_y, node_pos.shape[0], -1.0, -1.0, 1e-4, False,
        args.deterministic
    )
    init_density_map = init_density_map.contiguous()
    if (init_density_map > 1).sum() > 0:
        logger.warning("Some bins in init_density_map are overflow. Clamp them.")
    if (init_density_map < 0).sum() > 0:
        logger.error("init_density_map has negative value. Please check.")
    if args.use_route_force or args.use_cell_inflate:
        # consider snet as plaement blkg in density map to resolve M2 Vertical 
        # SNet pin access problem
        if gpdb is not None and gpdb.m1direction() == 0:
            # TODO: only include snet density when util is small
            snet_lpos, snet_size, snet_layer = gpdb.snet_info_tensor()

            snet_lpos = snet_lpos.to(device)
            snet_size = snet_size.to(device)
            snet_layer = snet_layer.to(device)
            m2_mask = snet_layer == 1
            snet_lpos = snet_lpos[m2_mask, :]
            snet_size = snet_size[m2_mask, :]
            snet_lpos -= data.die_shift
            snet_lpos /= data.die_scale
            snet_size /= data.die_scale

            snet_pos = snet_lpos + snet_size / 2
            snet_weight = snet_size.new_ones(snet_size.shape[0])
            snet_density_map = density_map_cuda.forward_naive(
                snet_pos, snet_size, snet_weight, data.unit_len, zeros_density_map,
                data.num_bin_x, data.num_bin_y, snet_pos.shape[0], -1.0, -1.0, 1e-4, False,
                args.deterministic
            )
            init_density_map += snet_density_map.contiguous()
    init_density_map.clamp_(min=0.0, max=1.0).mul_(args.target_density)
    if args.use_route_force or args.use_cell_inflate:
        # inflate connected IOPins
        _, fix_rhs, _ = data.node_type_indices[2]
        _, iopin_rhs, _ = data.node_type_indices[3]
        if fix_rhs != iopin_rhs:
            zeros_density_map = torch.zeros(
                (data.num_bin_x, data.num_bin_y), device=device, dtype=dtype,
            )
            iopin_pos = data.node_pos[fix_rhs:iopin_rhs]
            iopin_size = data.node_size[fix_rhs:iopin_rhs]
            iopin_weight = iopin_size.new_ones(iopin_size.shape[0])
            iopin_density_map = density_map_cuda.forward_naive(
                iopin_pos, iopin_size, iopin_weight, data.unit_len, zeros_density_map,
                data.num_bin_x, data.num_bin_y, iopin_pos.shape[0], -1.0, -1.0, 1e-4, False,
                args.deterministic
            )
            iopin_density_map = iopin_density_map.contiguous() * max(4 - 1, 0)
            init_density_map += iopin_density_map
    data.init_density_map = init_density_map
    return data.init_density_map


def init_params(
    mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
    density_map_layer, mov_node_size, expand_ratio, init_density_map, optimizer, 
    ps, data, args, route_fn=None
):  
    mov_node_pos = trunc_node_pos_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat(
        [conn_node_pos, conn_fix_node_pos], dim=0
    )
    wl_loss, hpwl = WAWirelengthLossAndHPWL.apply(
        conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
        data.node2pin_list, data.node2pin_list_end,
        data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
        ps.wa_coeff, data.hpwl_scale, args.deterministic
    )
    density_loss, overflow = density_map_layer(
        mov_node_pos, mov_node_size, init_density_map
    )
    wl_grad, density_grad = calc_grad(
        optimizer, mov_node_pos, wl_loss, density_loss
    )
    if not ps.rerun_route or route_fn is None:
        init_density_weight = (wl_grad.norm(p=1) / density_grad.norm(p=1)).detach()
        # init_density_weight = (wl_grad.norm(p=1) / grad_mat.norm(p=1)).detach()
        ps.set_init_param(init_density_weight, data, density_loss)
    else:
        _, filler_lhs = data.movable_connected_index
        filler_rhs = mov_node_pos.shape[0]

        mov_route_grad, mov_congest_grad, mov_pseudo_grad = route_fn(
            mov_node_pos, mov_node_size, expand_ratio, trunc_node_pos_fn
        )
        if mov_pseudo_grad is not None:
            wl_grad.add_(mov_pseudo_grad * args.pseudo_weight)
        init_density_weight = (wl_grad.norm(p=1) / density_grad.norm(p=1)).detach()
        init_route_weight = (density_grad.abs().max() / mov_route_grad.abs().max()).detach()
        # init_route_weight = (wl_grad[:filler_lhs].norm(p=1) / mov_route_grad[:filler_lhs].norm(p=1)).detach()
        init_congest_weight = (density_grad.abs().max() / mov_congest_grad.abs().max()).detach()
        # print("Weight den: %.4f route: %.4f congest: %.4f" % (init_density_weight, init_route_weight, init_congest_weight))
        ps.set_route_init_param(
            init_density_weight, init_route_weight, init_congest_weight, data, args
        )


# Nesterove learning rate initialization
def estimate_initial_learning_rate(obj_and_grad_fn, constraint_fn, x_k, lr):
    x_k = constraint_fn(x_k).clone().detach().requires_grad_(True)
    obj_k, g_k = obj_and_grad_fn(x_k)
    x_k_1 = (constraint_fn(x_k - lr * g_k)).clone().detach().requires_grad_(True)
    obj_k_1, g_k_1 = obj_and_grad_fn(x_k_1)
    return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)