from cpp_to_py import hpwl_cuda
import torch

class HPWL(object):
    @staticmethod
    def hpwl(pos: torch.Tensor, hyperedge_list, hyperedge_list_end):
        hpwl = hpwl_cuda.hpwl(pos, hyperedge_list, hyperedge_list_end)
        return hpwl

def get_hpwl(batch, pos):
    # CUDA only
    y = HPWL.hpwl(pos, batch.hyperedge_list, batch.hyperedge_list_end)
    return (
        torch.round(y * (batch.die_scale / batch.site_width)).sum(axis=1).unsqueeze(1)
    )