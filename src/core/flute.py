from cpp_to_py import flute_cpp
import torch
import numpy as np

class Flute(object):
    num_threads = 1

    @staticmethod
    def register(num_threads_=1, POWVFILE="thirdparty/flute/POWV9.dat", POSTFILE="thirdparty/flute/POST9.dat"):
        Flute.read_lut(POWVFILE, POSTFILE)
        Flute.num_threads = num_threads_
        
    @staticmethod
    def read_lut(POWVFILE, POSTFILE):
        flute_cpp.read_lut(POWVFILE, POSTFILE)  # only need to execute read_lut once

    @staticmethod
    def flute_wl(xs: list, ys: list):
        assert len(xs) == len(ys)
        if len(xs) > 150:
            raise NotImplementedError("net size is too big, flute only supports 150-pin net")
        rsmt_wl = flute_cpp.flute_rsmt_wl(xs, ys)
        return rsmt_wl
    
    @staticmethod
    def flute_wl_tensor(pos):
        if not isinstance(pos, torch.Tensor):
            raise NotImplementedError("flute_wl_tensor() only supports torch tensor")
        assert pos.shape[1] == 2 # 2D wirelength
        if pos.shape[0] > 150:
            raise NotImplementedError("net size is too big, flute only supports 150-pin net")
        xs_ys = pos.t().tolist()
        return flute_cpp.flute_rsmt_wl(xs_ys[0], xs_ys[1]) 
    
    @staticmethod
    def flute_wl_ndarray(pos):
        if not isinstance(pos, np.ndarray):
            raise NotImplementedError("flute_wl_ndarray() only supports torch ndarray")
        assert pos.shape[1] == 2 # 2D wirelength
        if pos.shape[0] > 150:
            raise NotImplementedError("net size is too big, flute only supports 150-pin net")
        xs_ys = pos.T.tolist()
        return flute_cpp.flute_rsmt_wl(xs_ys[0], xs_ys[1]) 
    
    @staticmethod
    def flute_wl_mt(pos, hyperedge_list, hyperedge_list_end):
        pos_t = pos.t().contiguous().tolist()
        hyperedge_list_ = hyperedge_list.tolist()
        hyperedge_list_end_ = hyperedge_list_end.tolist()
        rsmt = flute_cpp.flute_rsmt_wl_mt(pos_t[0], pos_t[1], 
                        hyperedge_list_, hyperedge_list_end_, Flute.num_threads)
        rsmt = torch.FloatTensor(rsmt).to(pos.get_device())
        return rsmt


def get_flute_wl(batch, pos):
    raise NotImplementedError("Not tested. Please make sure its behavior is correct. (incl. die_scale and site_width)")
    # CPU only
    y = Flute.flute_wl_mt(pos, batch.hyperedge_list, batch.hyperedge_list_end)
    return y.unsqueeze(1)
