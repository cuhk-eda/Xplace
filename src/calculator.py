import torch
from .param_scheduler import ParamScheduler
from .core import merged_wl_loss_grad, WAWirelengthLoss, WAWirelengthLossAndHPWL


def calc_loss(wl_loss, density_loss, ps, args):
    if args.loss_type == "weighted_sum":
        loss = (wl_loss + ps.density_weight * density_loss) / (1 + ps.density_weight)
    elif args.loss_type == "direct":
        loss = wl_loss + ps.density_weight * density_loss
    else:
        raise NotImplementedError("Loss type not defined")
    return loss


def apply_precond(mov_node_pos: torch.Tensor, ps: ParamScheduler, args):
    if not args.use_precond:
        return
    mov_node_pos.grad /= ps.precond_weight
    return mov_node_pos.grad

# For nesterov
def calc_obj_and_grad(
    mov_node_pos,
    constraint_fn=None,
    mov_node_size=None,
    init_density_map=None,
    density_map_layer=None,
    conn_fix_node_pos=None,
    ps=None,
    data=None,
    args=None,
    merged_forward_backward=True,
):
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos = constraint_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat([conn_node_pos, conn_fix_node_pos], dim=0)
    if merged_forward_backward:
        if mov_node_pos.grad is not None:
            mov_node_pos.grad.zero_()
        else:
            mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()
        wl_loss, conn_node_grad_by_wl = merged_wl_loss_grad(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos, 
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
            data.hpwl_scale, ps.wa_coeff
        )
        mov_node_pos.grad[mov_lhs:mov_rhs] = conn_node_grad_by_wl[mov_lhs:mov_rhs]
        if ps.enable_sample_force:
            if ps.iter > 3 and ps.iter % 20 == 0:
                # ps.iter > 3 for warmup
                density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                    mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
                )
                ps.force_ratio = (
                    ps.density_weight * node_grad_by_density[mov_lhs:mov_rhs].norm(p=1) / 
                    conn_node_grad_by_wl[mov_lhs:mov_rhs].norm(p=1)
                ).clamp_(max=10)
                mov_node_pos.grad += node_grad_by_density * ps.density_weight
            else:
                density_loss = 0.0
            if (ps.iter > 3 and ps.recorder.force_ratio[-1] > 1e-2) or ps.iter > 100:
                # no longer enable sampling back
                ps.enable_sample_force = False
        else:
            density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
            )
            mov_node_pos.grad += node_grad_by_density * ps.density_weight
        grad = apply_precond(mov_node_pos, ps, args)
        loss = wl_loss + ps.density_weight * density_loss
    else:
        if mov_node_pos.grad is not None:
            mov_node_pos.grad.zero_()
        else:
            mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()
        wl_loss = WAWirelengthLoss.apply(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos, 
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask, ps.wa_coeff
        )
        density_loss, _ = density_map_layer(
            mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
        )
        loss = calc_loss(wl_loss, density_loss, ps, args)
        loss.backward()
        grad = apply_precond(mov_node_pos, ps, args)
    return loss, grad

# For Adam

def calc_grad(
    optimizer: torch.optim.Optimizer, mov_node_pos: torch.Tensor, wl_loss, density_loss
):
    optimizer.zero_grad()
    wl_loss.backward(retain_graph=True)
    wl_grad = mov_node_pos.grad.detach().clone()
    optimizer.zero_grad()
    density_loss.backward(retain_graph=True)
    density_grad = mov_node_pos.grad.detach().clone()
    optimizer.zero_grad()
    return wl_grad, density_grad


def fast_optimization(
    mov_node_pos, trunc_node_pos_fn, mov_lhs, mov_rhs, conn_fix_node_pos, 
    density_map_layer, mov_node_size, init_density_map, ps, data, args
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
    loss = calc_loss(wl_loss, density_loss, ps, args)
    loss.backward()
    apply_precond(mov_node_pos, ps, args)
    # calculate objective (hpwl, overflow)
    return hpwl.detach(), overflow.detach(), mov_node_pos