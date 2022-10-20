import torch
import numpy as np
import torch.fft


def torch_dct_idct(density_map: torch.Tensor, fft_scale):
    potential_scale, potential_coeff, force_x_scale, force_y_scale, force_x_coeff, force_y_coeff = fft_scale
    fft_coeff = dct_2d(density_map)  # Real number, M x N
    fft_coeff = fft_coeff * 4 # to align with cuda dct implementation
    potential_map = idct_2d(fft_coeff * potential_scale).real * potential_coeff # M x N
    force_x_map = compute_electronic_force(fft_coeff, force_x_scale, force_x_coeff, dim=0)
    force_y_map = compute_electronic_force(fft_coeff, force_y_scale, force_y_coeff, dim=1)
    grad_mat = torch.vstack(
        (force_x_map.unsqueeze(0), force_y_map.unsqueeze(0))
    )  # 2 x M x N
    grad_mat = grad_mat.contiguous()
    return grad_mat, potential_map


class FFTBasisCache:
    def __init__(self) -> None:
        self.dct = {}
        self.idct = {}


fft_basis_cache = FFTBasisCache()


class DCTmtxCache:
    def __init__(self) -> None:
        self.dctA = {}
        self.idctA = {}


dct_matrix = DCTmtxCache()


def compute_electronic_force(x: torch.Tensor, scale, coeff, dim=0):
    # E_x: dim == 0, E_y: dim == 1
    assert len(x.shape) == 2
    assert dim in [0, 1]
    x = x * scale
    if dim == 0:
        x = torch.cat((x[:1, :] * 0, x[1:, :].flip([0])), dim=0)
    else:
        x = torch.cat((x[:, :1] * 0, x[:, 1:].flip([1])), dim=1)
    x = idct_2d(x).real
    x = x * coeff
    return x


# https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v, dim=1)

    if N not in fft_basis_cache.dct.keys():
        k = -torch.arange(N, device=x.device)[None, :] * np.pi / (2 * N)
        fft_basis_cache.dct[N] = torch.cos(k) + 1j * torch.sin(k)

    V = Vc * fft_basis_cache.dct[N]

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.real.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    if N not in fft_basis_cache.idct.keys():
        k = torch.arange(N, device=X.device)[None, :] * np.pi / (2 * N)
        fft_basis_cache.idct[N] = torch.cos(k) + 1j * torch.sin(k)

    V_t = X_v + 1j * torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V = V_t * fft_basis_cache.idct[N]

    v = torch.fft.ifft(V, dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


class LinearDCT(torch.nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(
        self,
        in_features,
        device=torch.device("cuda:0"),
        type="dct",
        norm=None,
        bias=False,
    ):
        self.type = type
        self.device = device
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N, device=self.device)
        if self.type == "dct":
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == "idct":
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!


def apply_linear_layer_2d(x, linear_dct0, linear_dct1):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    with torch.no_grad():
        X1 = linear_dct1(x)
        X2 = linear_dct0(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)


def apply_linear_weight_2d(x, W1, W2, scale=None):
    # Fastest one but has floating point precision error
    if scale is not None:
        x = x * scale
    X1 = x @ W2
    X2 = X1.transpose(-1, -2) @ W1
    return X2.transpose(-1, -2)


def get_linear_weight_2d(shape, device, type="dct"):
    assert len(shape) == 2
    n1, n2 = shape
    if type == "dct":
        fft_func = dct
    elif type == "idct":
        fft_func = idct
    I1 = torch.eye(n1, device=device)
    W1 = fft_func(I1).data
    I2 = torch.eye(n2, device=device)
    W2 = fft_func(I2).data
    return W1, W2


# LinearDCT
def L_dct(x):
    tensor_shape = x.shape
    if tensor_shape not in dct_matrix.dctA.keys():
        layer0, layer1 = get_linear_weight_2d(tensor_shape, x.device, type="dct")
        dct_matrix.dctA[tensor_shape] = [layer0, layer1]
    layer0, layer1 = dct_matrix.dctA[tensor_shape]
    return apply_linear_weight_2d(x, layer0, layer1)


def L_idct(x):
    tensor_shape = x.shape
    if tensor_shape not in dct_matrix.idctA.keys():
        layer0, layer1 = get_linear_weight_2d(tensor_shape, x.device, type="idct")
        dct_matrix.idctA[tensor_shape] = [layer0, layer1]
    layer0, layer1 = dct_matrix.idctA[tensor_shape]
    x = x.to(torch.cfloat)
    return apply_linear_weight_2d(x, layer0, layer1)
