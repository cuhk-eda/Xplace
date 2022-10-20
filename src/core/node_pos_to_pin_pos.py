import torch
from cpp_to_py import node_pos_to_pin_pos_cuda

class NodePosToPinPosFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_pos: torch.Tensor,
        pin_id2node_id: torch.Tensor,
        pin_rel_cpos: torch.Tensor,
    ):
        ctx.num_nodes = node_pos.shape[0]
        ctx.save_for_backward(pin_id2node_id)
        pin_pos = node_pos_to_pin_pos_cuda.forward(node_pos, pin_id2node_id, pin_rel_cpos)
        return pin_pos
    
    @staticmethod
    def backward(ctx, pos_grad: torch.Tensor):
        pin_id2node_id = ctx.saved_tensors[0]
        node_grad = torch.zeros((ctx.num_nodes, 2), dtype=pos_grad.dtype, device=pin_id2node_id.device)
        # TODO / NOTE: may cause non-deterministic
        node_grad.scatter_add_(0, pin_id2node_id.unsqueeze(1).expand(-1,2), pos_grad)
        return node_grad, None, None
