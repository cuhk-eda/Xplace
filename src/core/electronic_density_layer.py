import torch
from cpp_to_py import density_map_cuda
import numpy as np
import math
from .dct2_fft2 import dct2, idct2, idxst_idct, idct_idxst, dct2_fft2_cache
from .torch_dct import torch_dct_idct


class ElectronicDensityFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        node_pos: torch.Tensor,
        node_size: torch.Tensor,
        node_weight: torch.Tensor,
        expand_ratio: torch.Tensor,
        unit_len: torch.Tensor,
        init_density_map: torch.Tensor,
        num_bin_x: int,
        num_bin_y: int,
        num_nodes: int,
        fft_scale: tuple,
        overflow_helper: tuple,
        sorted_maps: tuple,
        calc_overflow: bool,
        deterministic: bool,
    ):
        ctx.constant_var = (num_bin_x, num_bin_y, num_nodes, deterministic)
        mov_lhs, mov_rhs, overflow_fn = overflow_helper
        mov_sorted_map, mov_conn_sorted_map, filler_sorted_map = sorted_maps
        # 1) Compute Density Map
        normalize_node_info = node_size.new_empty((num_nodes, 5)) # x_l, x_h, y_l, y_h, weight
        normalize_node_info = density_map_cuda.pre_normalize(
            node_pos, node_size, node_weight, expand_ratio, unit_len, normalize_node_info,
            num_bin_x, num_bin_y, num_nodes, 
        )
        if calc_overflow:
            aux_mat = init_density_map.clone()
            mov_density_map = density_map_cuda.forward(
                normalize_node_info[mov_lhs:mov_rhs], mov_conn_sorted_map, aux_mat, 
                num_bin_x, num_bin_y, mov_rhs - mov_lhs, deterministic
            )
            overflow = overflow_fn(mov_density_map)
            aux_mat2 = torch.zeros_like(init_density_map)
            if filler_sorted_map is not None:
                filler_density_map = density_map_cuda.forward(
                    normalize_node_info[mov_rhs:], filler_sorted_map, aux_mat2, 
                    num_bin_x, num_bin_y, num_nodes - (mov_rhs - mov_lhs), deterministic
                )
                density_map = mov_density_map + filler_density_map
            else:
                density_map = mov_density_map
        else:
            overflow = node_size.new_empty(0)
            aux_mat = init_density_map.clone()
            density_map = density_map_cuda.forward(
                normalize_node_info, mov_sorted_map, aux_mat, 
                num_bin_x, num_bin_y, num_nodes, deterministic
            )
        # density_map -= density_map.mean() # we don't need this one anymore
        
        if True:
            potential_scale, potential_coeff, force_x_scale, force_y_scale, force_x_coeff, force_y_coeff = fft_scale
            fft_coeff = dct2(density_map)
            force_x_map = idxst_idct(fft_coeff * force_x_scale)
            force_y_map = idct_idxst(fft_coeff * force_y_scale)
            potential_map = idct2(fft_coeff * potential_scale)
            grad_mat = torch.vstack(
                (force_x_map.unsqueeze(0), force_y_map.unsqueeze(0))
            ).contiguous()  # 2 x M x N
        else:
            # NOTE: in some of cases, torch version is faster
            grad_mat, potential_map = torch_dct_idct(density_map, fft_scale)

        energy = (potential_map * density_map).sum()
        ctx.save_for_backward(normalize_node_info, mov_sorted_map, grad_mat)

        return energy, overflow

    @staticmethod
    def backward(ctx, energy_grad_out, overflow_grad_out):
        normalize_node_info, mov_sorted_map, grad_mat = ctx.saved_tensors
        num_bin_x, num_bin_y, num_nodes, deterministic = ctx.constant_var
        grad_mat = grad_mat * energy_grad_out
        grad_weight = -1.0 # Gradient descent
        node_grad = normalize_node_info.new_zeros((num_nodes, 2))
        node_grad = density_map_cuda.backward(
            normalize_node_info, grad_mat, mov_sorted_map, node_grad,
            grad_weight, num_bin_x, num_bin_y, num_nodes, deterministic
        )
        return (node_grad,) + (None,) * 13
    
def merged_density_loss_grad_main(
    node_pos: torch.Tensor,
    node_size: torch.Tensor,
    node_weight: torch.Tensor,
    expand_ratio: torch.Tensor,
    unit_len: torch.Tensor,
    init_density_map: torch.Tensor,
    num_bin_x: int,
    num_bin_y: int,
    num_nodes: int,
    fft_scale: tuple,
    overflow_helper: tuple,
    sorted_maps: tuple,
    calc_overflow: bool,
    deterministic: bool,
    cache_mov_density_map: bool = True,
):
    mov_lhs, mov_rhs, overflow_fn = overflow_helper
    mov_sorted_map, mov_conn_sorted_map, filler_sorted_map = sorted_maps
    # 1) Compute Density Map
    # Due to atomic add, density map calculation is non-deterministic
    #   please use deterministic mode
    normalize_node_info = node_size.new_empty((num_nodes, 5)) # x_l, x_h, y_l, y_h, weight
    normalize_node_info = density_map_cuda.pre_normalize(
        node_pos, node_size, node_weight, expand_ratio, unit_len, normalize_node_info,
        num_bin_x, num_bin_y, num_nodes
    )
    mov_density_map = init_density_map.clone()
    mov_density_map = density_map_cuda.forward(
        normalize_node_info[mov_lhs:mov_rhs], mov_conn_sorted_map, mov_density_map, 
        num_bin_x, num_bin_y, mov_rhs - mov_lhs, deterministic
    )
    if filler_sorted_map is not None:
        filler_density_map = torch.zeros_like(init_density_map)
        filler_density_map = density_map_cuda.forward(
            normalize_node_info[mov_rhs:], filler_sorted_map, filler_density_map, 
            num_bin_x, num_bin_y, num_nodes - (mov_rhs - mov_lhs), deterministic
        )
        density_map = filler_density_map
        density_map.add_(mov_density_map)
    else:
        density_map = mov_density_map
    if calc_overflow:
        overflow = overflow_fn(mov_density_map)
    else:
        overflow = None

    potential_scale, _, force_x_scale, force_y_scale, _, _ = fft_scale
    fft_coeff = dct2(density_map)
    force_x_map = idxst_idct(fft_coeff * force_x_scale)
    force_y_map = idct_idxst(fft_coeff * force_y_scale)
    potential_map = idct2(fft_coeff * potential_scale)
    grad_mat = torch.vstack(
        (force_x_map.unsqueeze(0), force_y_map.unsqueeze(0))
    ).contiguous()  # 2 x M x N

    energy = (potential_map * density_map).sum()

    grad_weight = -1.0 # Gradient descent
    node_grad = normalize_node_info.new_zeros((num_nodes, 2))
    node_grad = density_map_cuda.backward(
        normalize_node_info, grad_mat, mov_sorted_map, node_grad,
        grad_weight, num_bin_x, num_bin_y, num_nodes, deterministic
    )
    if not cache_mov_density_map:
        mov_density_map = None
    return energy, overflow, node_grad, mov_density_map


class ElectronicDensityLayer(torch.nn.Module):
    def __init__(
        self,
        unit_len=None,
        inference_mode=True,
        num_bin_x=100,
        num_bin_y=100,
        device=torch.device("cuda:0"),
        overflow_helper=None,
        expand_ratio=None,
        sorted_maps=None,
        scale_w_k=True,
        deterministic=False,
    ):
        super(ElectronicDensityLayer, self).__init__()
        self.num_bin_x = num_bin_x
        self.num_bin_y = num_bin_y
        self.deterministic = deterministic

        assert overflow_helper is not None
        self.overflow_helper = overflow_helper

        self.expand_ratio = expand_ratio
        self.sorted_maps = sorted_maps

        self.inference_mode = inference_mode
        self.has_setup_inference_var = False

        if unit_len is None:
            self.unit_len = torch.tensor([1.0 / num_bin_x, 1.0 / num_bin_y], device=device)
        else:
            self.unit_len = unit_len
        
        self.min_node_w = self.unit_len[0].item() * math.sqrt(2)
        self.min_node_h = self.unit_len[1].item() * math.sqrt(2)

        # Pre-compute some constants in FFT computation
        w_j = torch.arange(num_bin_x, device=device).float().mul(2 * np.pi / num_bin_x).reshape(num_bin_x, 1)
        w_k = torch.arange(num_bin_y, device=device).float().mul(2 * np.pi / num_bin_y).reshape(1, num_bin_y)
        # scale_w_k because the aspect ratio of a bin may not be 1 
        # NOTE: we will not scale down w_k in NN since it may distrub the training 
        if scale_w_k:
            w_k.mul_(self.unit_len[0] / self.unit_len[1])
        wj2_plus_wk2 = w_j.pow(2) + w_k.pow(2)
        wj2_plus_wk2[0, 0] = 1.0

        potential_scale = 1.0 / wj2_plus_wk2
        potential_scale[0, 0] = 0.0

        force_x_scale = w_j * potential_scale * 0.5
        force_y_scale = w_k * potential_scale * 0.5

        force_x_coeff = ((-1.0) ** torch.arange(num_bin_x, device=device)).unsqueeze(1)
        force_y_coeff = ((-1.0) ** torch.arange(num_bin_y, device=device)).unsqueeze(0)

        dct_scalar = 1 / (num_bin_x * num_bin_y)
        idct_scalar = num_bin_x * num_bin_y * 4

        potential_coeff = 1.0
        # potential_scale *= dct_scalar
        # force_x_scale *= dct_scalar
        # force_y_scale *= dct_scalar
        # force_x_coeff *= idct_scalar
        # force_y_coeff *= idct_scalar
        # potential_coeff = idct_scalar

        self.fft_scale = (potential_scale, potential_coeff, force_x_scale, force_y_scale, force_x_coeff, force_y_coeff)

        # Cache
        self.use_cache_mov_density_map = True
        self.mov_density_map = None
        # reset global var
        dct2_fft2_cache.reset()

    def preprocess_inference_var(self, node_pos, node_size, node_weight):
        # precompute all common variables for faster inferencing
        assert self.inference_mode
        assert not self.has_setup_inference_var
        if node_weight is None:
            # generate all ones node_weight if not given node_weight
            node_weight = torch.ones(
                node_size.shape[0],
                device=node_size.get_device(),
                dtype=node_size.dtype,
            ).detach()
        self.cache_node_weight = node_weight

        self.has_setup_inference_var = True

    def get_cache_var(self, node_pos, node_size, node_weight):
        if self.inference_mode:
            if not self.has_setup_inference_var:
                with torch.no_grad():
                    self.preprocess_inference_var(
                        node_pos, node_size, node_weight
                    )
            node_weight = self.cache_node_weight

        if node_weight is None:
            node_weight = node_pos.new_ones(node_pos.shape[0]).detach()
        
        return node_weight

    def get_density_map_naive(
        self,
        node_pos,
        node_size,
        init_density_map=None,
    ):
        node_weight = node_size.new_ones(node_pos.shape[0])
        if init_density_map is None:
            node_pos.new_zeros(self.num_bin_x, self.num_bin_y)
        aux_mat = init_density_map.clone()
        num_nodes = node_pos.shape[0]
        density_map = density_map_cuda.forward_naive(
            node_pos,
            node_size,
            node_weight,
            self.unit_len, 
            aux_mat,
            self.num_bin_x,
            self.num_bin_y,
            num_nodes,
            -1.0,
            -1.0,
            1e-4,
            False,
            self.deterministic,
        )

        return density_map

    def direct_calc_overflow(
        self,
        node_pos,
        node_size,
        init_density_map,
        node_weight=None,
    ):
        mov_lhs, mov_rhs, overflow_fn = self.overflow_helper
        if self.use_cache_mov_density_map and self.mov_density_map is not None:
            mov_density_map = self.mov_density_map
            self.mov_density_map = None
        else:
            _, mov_conn_sorted_map, _ = self.sorted_maps
            num_mov_nodes = mov_rhs - mov_lhs
            node_weight = self.get_cache_var(node_pos, node_size, node_weight)

            normalize_node_info = node_size.new_zeros((num_mov_nodes, 5)) # x_l, x_h, y_l, y_h, weight
            normalize_node_info = density_map_cuda.pre_normalize(
                node_pos[mov_lhs:mov_rhs], node_size[mov_lhs:mov_rhs], node_weight[mov_lhs:mov_rhs], 
                self.expand_ratio[mov_lhs:mov_rhs], self.unit_len, normalize_node_info,
                self.num_bin_x, self.num_bin_y, num_mov_nodes
            )
            aux_mat = init_density_map.clone()
            mov_density_map = density_map_cuda.forward(
                normalize_node_info, mov_conn_sorted_map, aux_mat, 
                self.num_bin_x, self.num_bin_y, num_mov_nodes, self.deterministic,
            )
        overflow = overflow_fn(mov_density_map)
        return overflow

    def forward(
        self,
        node_pos,
        node_size,
        init_density_map,
        node_weight=None,
        calc_overflow=True
    ):
        node_weight = self.get_cache_var(
            node_pos, node_size, node_weight
        )
        num_nodes = node_pos.shape[0]
        energy, overflow = ElectronicDensityFunction.apply(
            node_pos, node_size, node_weight, self.expand_ratio, self.unit_len, 
            init_density_map, self.num_bin_x, self.num_bin_y, num_nodes, 
            self.fft_scale, self.overflow_helper, self.sorted_maps, calc_overflow,
            self.deterministic
        )
        return energy, overflow

    def merged_density_loss_grad(
        self,
        node_pos,
        node_size,
        init_density_map,
        node_weight=None,
        calc_overflow=True
    ):
        node_weight = self.get_cache_var(
            node_pos, node_size, node_weight
        )
        num_nodes = node_pos.shape[0]
        energy, overflow, node_grad, mov_density_map = merged_density_loss_grad_main(
            node_pos, node_size, node_weight, self.expand_ratio, self.unit_len, 
            init_density_map, self.num_bin_x, self.num_bin_y, num_nodes, 
            self.fft_scale, self.overflow_helper, self.sorted_maps, calc_overflow,
            self.deterministic, cache_mov_density_map=self.use_cache_mov_density_map,
        )
        if self.use_cache_mov_density_map:
            self.mov_density_map = mov_density_map
        return energy, overflow, node_grad 
