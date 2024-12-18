import torch
from .database import PlaceData
from .core import masked_scale_hpwl
from cpp_to_py import hpwl_cuda, density_map_cuda

def get_hpwl(data, pos):
    # CUDA only
    hpwl = hpwl_cuda.hpwl(pos, data.hyperedge_list, data.hyperedge_list_end)
    return (
        torch.round(hpwl * (data.die_scale / data.site_width)).sum(axis=1).unsqueeze(1)
    )

def get_obj_hpwl(node_pos, data: PlaceData, args):
    mov_lhs, mov_rhs = data.movable_index
    fix_lhs, fix_rhs = data.fixed_connected_index
    conn_node_pos = torch.cat([
        node_pos[mov_lhs:mov_rhs], node_pos[fix_lhs:fix_rhs]
    ], dim=0)
    with torch.no_grad():
        pin_pos = hpwl_cuda.node_pos_to_pin_pos(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos
        )
        hpwl = torch.sum(get_hpwl(data, pin_pos.detach()))
    return hpwl

def get_obj_overflow(node_pos, init_density_map, ps, data: PlaceData, args):
    mov_lhs, mov_rhs = data.movable_index
    node_pos = node_pos[mov_lhs:mov_rhs]
    node_size = data.node_size[mov_lhs:mov_rhs]
    if ps.zero_macro_grad:
        node_pos = node_pos[
            torch.logical_not(data.is_mov_macro[mov_lhs:mov_rhs])
        ].contiguous()
        node_size = node_size[
            torch.logical_not(data.is_mov_macro[mov_lhs:mov_rhs])
        ].contiguous()
    node_weight = node_size.new_ones(node_pos.shape[0])

    if init_density_map is None:
        init_density_map = node_pos.new_zeros(data.num_bin_x, data.num_bin_y)
    aux_mat = init_density_map.clone()
    density_map = density_map_cuda.forward_naive(
        node_pos, node_size, node_weight, data.unit_len, aux_mat, data.num_bin_x, 
        data.num_bin_y, node_pos.shape[0], -1.0, -1.0, 1e-4, False, args.deterministic,
    )

    with torch.no_grad():
        overflow_sum = ((density_map - args.target_density) * data.bin_area).clamp_(min=0.0).sum()
        overflow = overflow_sum / data.total_mov_area_without_filler
    return overflow

def evaluate_placement(node_pos, init_density_map, ps, data: PlaceData, args):
    # NOTE: since some nets are masked in global placement, hpwl may 
    # underestimate, this function return the exact value of hpwl
    # Original overflow calculation uses the clamp node size (expand ratio), 
    # this function uses the exact node size to evaluate the overflow
    hpwl = get_obj_hpwl(node_pos, data, args)
    overflow = get_obj_overflow(node_pos, init_density_map, ps, data, args)
    return hpwl, overflow

def fast_evaluator(
    mov_node_pos,
    constraint_fn=None,
    mov_node_size=None,
    init_density_map=None,
    density_map_layer=None,
    conn_fix_node_pos=None,
    ps=None,
    data=None,
    args=None,
):
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos = constraint_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat([conn_node_pos, conn_fix_node_pos], dim=0)
    masked_hpwl = masked_scale_hpwl(
        conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos, 
        data.hyperedge_list, data.hyperedge_list_end, data.net_mask, data.hpwl_scale
    )
    overflow = density_map_layer.direct_calc_overflow(
        mov_node_pos, mov_node_size, init_density_map
    )
    return masked_hpwl, overflow