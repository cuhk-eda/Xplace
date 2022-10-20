import torch
from .database import PlaceData
from cpp_to_py import density_map_cuda
from .core import WAWirelengthLossAndHPWL
from .calculator import calc_grad


def get_init_density_map(data: PlaceData, args, logger):
    lhs, rhs = data.fixed_index
    device = data.node_size.get_device()
    dtype = data.node_size.dtype
    zeros_density_map = torch.zeros(
        (data.num_bin_x, data.num_bin_y), device=device, dtype=dtype,
    )
    if lhs == rhs:
        return zeros_density_map
    # get fix nodes which are located inside die
    node_pos = data.node_pos[lhs:rhs]
    node_size = data.node_size[lhs:rhs]
    # if data.fix_node_in_bd_mask is not None:
    #     node_pos = node_pos[data.fix_node_in_bd_mask]
    #     node_size = node_size[data.fix_node_in_bd_mask]
    # if data.dummy_macro_pos is not None and data.dummy_macro_size is not None:
    #     node_pos = torch.cat([node_pos, data.dummy_macro_pos], dim=0)
    #     node_size = torch.cat([node_size, data.dummy_macro_size], dim=0)
    # if node_size.shape[0] == 0:
    #     return zeros_density_map
    node_weight = node_size.new_ones(node_size.shape[0])
    init_density_map = density_map_cuda.forward_naive(
        node_pos, node_size, node_weight, data.unit_len, zeros_density_map,
        data.num_bin_x, data.num_bin_y, node_pos.shape[0], -1.0, -1.0, 1e-4, False
    )
    init_density_map = init_density_map.contiguous()
    if (init_density_map > 1).sum() > 0:
        logger.warning("Some bins in init_density_map are overflow. Clamp them.")
    if (init_density_map < 0).sum() > 0:
        logger.error("init_density_map has negative value. Please check.")
    init_density_map.clamp_(min=0.0, max=1.0).mul_(args.target_density)
    data.init_density_map = init_density_map
    return data.init_density_map


def init_params(
    mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
    density_map_layer, mov_node_size, init_density_map, optimizer, ps, data, args
):  
    mov_node_pos = trunc_node_pos_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat(
        [conn_node_pos, conn_fix_node_pos], dim=0
    )
    wl_loss, hpwl = WAWirelengthLossAndHPWL.apply(
        conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos, 
        data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
        ps.wa_coeff, data.hpwl_scale
    )
    density_loss, overflow = density_map_layer(
        mov_node_pos, mov_node_size, init_density_map
    )
    wl_grad, density_grad = calc_grad(
        optimizer, mov_node_pos, wl_loss, density_loss
    )
    init_density_weight = (wl_grad.norm(p=1) / density_grad.norm(p=1)).detach()
    # init_density_weight = (wl_grad.norm(p=1) / grad_mat.norm(p=1)).detach()
    ps.set_init_param(init_density_weight, data, density_loss)


# Nesterove learning rate initialization
def estimate_initial_learning_rate(obj_and_grad_fn, constraint_fn, x_k, lr):
    x_k = constraint_fn(x_k).clone().detach().requires_grad_(True)
    obj_k, g_k = obj_and_grad_fn(x_k)
    x_k_1 = (constraint_fn(x_k - lr * g_k)).clone().detach().requires_grad_(True)
    obj_k_1, g_k_1 = obj_and_grad_fn(x_k_1)
    return (x_k - x_k_1).norm(p=2) / (g_k - g_k_1).norm(p=2)