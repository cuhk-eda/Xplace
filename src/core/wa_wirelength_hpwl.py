import torch
from cpp_to_py import wa_wirelength_hpwl_cuda

class HPWLCache:
    def __init__(self) -> None:
        self.masked_scale_partial_hpwl = None

hpwl_cache = HPWLCache()


def merged_wl_loss_grad(
    node_pos,
    pin_id2node_id,
    pin_rel_cpos,
    node2pin_list,
    node2pin_list_end,
    hyperedge_list,
    hyperedge_list_end,
    net_mask,
    hpwl_scale,
    gamma,
    deterministic,
    cache_hpwl=True,
):
    (
        partial_wa_wl,
        node_grad,
        partial_hpwl,
    ) = wa_wirelength_hpwl_cuda.merged_forward_backward_with_masked_scale_hpwl(
        node_pos,
        pin_id2node_id,
        pin_rel_cpos,
        node2pin_list,
        node2pin_list_end,
        hyperedge_list,
        hyperedge_list_end,
        net_mask,
        hpwl_scale,
        gamma,
        deterministic,
    )
    if cache_hpwl:
        hpwl_cache.masked_scale_partial_hpwl = partial_hpwl
    return torch.sum(partial_wa_wl), node_grad


def masked_scale_hpwl(
    node_pos,
    pin_id2node_id,
    pin_rel_cpos,
    hyperedge_list,
    hyperedge_list_end,
    net_mask,
    hpwl_scale,
):  
    if hpwl_cache.masked_scale_partial_hpwl is None:
        return wa_wirelength_hpwl_cuda.masked_scale_hpwl_sum(
            node_pos,
            pin_id2node_id,
            pin_rel_cpos,
            hyperedge_list,
            hyperedge_list_end,
            net_mask,
            hpwl_scale,
        )
    else:
        return torch.sum(hpwl_cache.masked_scale_partial_hpwl)
