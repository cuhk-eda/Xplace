import torch
import math
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

def calc_obj_and_grad(
    mov_node_pos,
    constraint_fn=None,
    route_fn=None,
    mov_node_size=None,
    expand_ratio=None,
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

        if ps.use_route_force and ps.start_route_opt:
            mov_route_grad, mov_congest_grad, mov_pseudo_grad = route_fn(
                mov_node_pos, mov_node_size, expand_ratio, constraint_fn
            )
            mov_node_pos.grad += mov_route_grad * ps.route_weight
            mov_node_pos.grad += mov_congest_grad * ps.congest_weight
            mov_node_pos.grad += mov_pseudo_grad * ps.pseudo_weight

        wl_loss, conn_node_grad_by_wl = merged_wl_loss_grad(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
            data.hpwl_scale, ps.wa_coeff, args.deterministic
        )
        mov_node_pos.grad[mov_lhs:mov_rhs] += conn_node_grad_by_wl[mov_lhs:mov_rhs]
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
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask,
            ps.wa_coeff, args.deterministic
        )
        density_loss, _ = density_map_layer(
            mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
        )
        loss = calc_loss(wl_loss, density_loss, ps, args)
        loss.backward()
        grad = apply_precond(mov_node_pos, ps, args)
    return loss, grad

# For Xplace-NN nesterov
def calc_obj_and_grad_nn(
    mov_node_pos,
    constraint_fn=None,
    mov_node_size=None,
    mov_node_size_nn=None,
    init_density_map=None,
    init_density_map_nn=None,
    density_map_layer=None,
    density_map_layer_nn=None,
    conn_fix_node_pos=None,
    ps=None,
    data=None,
    args=None,
    merged_forward_backward=True,
    model=None,
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
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
            data.hpwl_scale, ps.wa_coeff, args.deterministic
        )
        mov_node_pos.grad[mov_lhs:mov_rhs] = conn_node_grad_by_wl[mov_lhs:mov_rhs]
        if ps.enable_sample_force:
            ps.use_nn = False
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
        elif ps.iter > 100 and ps.weighted_weight < args.ps_end and not ps.terminate_nn:
            # density_map_layer_nn: use nn_size model to calc nn_bin level grad (can use "ep" or "nn")
            # density_map_layer: use nn_size model to calc num_bin_x/y level grad (can use "ep" or "nn")
            ps.use_nn = True
            density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                mov_node_pos, mov_node_size, init_density_map, calc_overflow=False,
                grad_fn="ep",
                model=model,
                cache_grad_4norm=True,
                norm_grad=False,
            )
            _, _, node_grad_by_density_nn = density_map_layer_nn.merged_density_loss_grad(
                mov_node_pos, mov_node_size_nn, init_density_map_nn, calc_overflow=False,
                grad_fn="nn",
                model=model,
                cache_grad_4norm=False,
                norm_grad=True,
            )
            # compute smooth function (for nesterov)
            s_func = lambda a,x: 1 - 1 / (1 + torch.exp(-a * (x / args.ps_end-0.5)))
            t_func = lambda a,b,x: 1 / (1 + math.exp(-a * (x / b - 0.5)))
            a = s_func(5, ps.weighted_weight)
            b = t_func(5, 0.005, ps.force_ratio)
            sigma = max(a + b - 1, 0)

            # we observe that using NN mode too long would affect the quality and runtime,
            # here is a fast decay trick to detect and avoid that
            if ps.nn_dominate_iter is None and sigma > 0.5:
                ps.nn_dominate_iter = ps.iter
            if sigma > 0.5 and ps.iter - ps.nn_dominate_iter > ps.max_nn_dominate_iter:
                ps.fast_decay = True
            if ps.fast_decay:
                e_func = lambda a,b,x: 1 - math.exp(-a * (x - b))
                diff_iter = ps.iter - (ps.nn_dominate_iter + ps.max_nn_dominate_iter)
                sigma = max(sigma - e_func(0.1, 0, diff_iter), 0)
                if sigma < 1e-4:
                    # save runtime
                    ps.terminate_nn = True

            # update grad
            ps.nn_sigma = sigma
            node_grad_by_density = ((1 - sigma) * node_grad_by_density + sigma * node_grad_by_density_nn)
            ps.force_ratio = (
                ps.density_weight * node_grad_by_density[mov_lhs:mov_rhs].norm(p=1) / 
                conn_node_grad_by_wl[mov_lhs:mov_rhs].norm(p=1)
            ).clamp_(max=10)
            mov_node_pos.grad += node_grad_by_density * ps.density_weight
        else:
            ps.use_nn = False
            ps.nn_sigma = 0
            density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                mov_node_pos, mov_node_size, init_density_map, calc_overflow=False,
                grad_fn="ep",
                model=model,
            )
            mov_node_pos.grad += node_grad_by_density * ps.density_weight
            ps.force_ratio = (
                ps.density_weight * node_grad_by_density[mov_lhs:mov_rhs].norm(p=1) / 
                conn_node_grad_by_wl[mov_lhs:mov_rhs].norm(p=1)
            ).clamp_(max=10)
        grad = apply_precond(mov_node_pos, ps, args)
        loss = wl_loss + ps.density_weight * density_loss
    else:
        if mov_node_pos.grad is not None:
            mov_node_pos.grad.zero_()
        else:
            mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()
        wl_loss = WAWirelengthLoss.apply(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask,
            ps.wa_coeff, args.deterministic
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
    optimizer.zero_grad(set_to_none=False)
    wl_loss.backward(retain_graph=True)
    wl_grad = mov_node_pos.grad.detach().clone()
    optimizer.zero_grad(set_to_none=False)
    density_loss.backward(retain_graph=True)
    density_grad = mov_node_pos.grad.detach().clone()
    optimizer.zero_grad(set_to_none=False)
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
        data.node2pin_list, data.node2pin_list_end,
        data.hyperedge_list, data.hyperedge_list_end, data.net_mask, 
        ps.wa_coeff, data.hpwl_scale, args.deterministic
    )
    density_loss, overflow = density_map_layer(
        mov_node_pos, mov_node_size, init_density_map
    )
    loss = calc_loss(wl_loss, density_loss, ps, args)
    loss.backward()
    apply_precond(mov_node_pos, ps, args)
    # calculate objective (hpwl, overflow)
    return hpwl.detach(), overflow.detach(), mov_node_pos