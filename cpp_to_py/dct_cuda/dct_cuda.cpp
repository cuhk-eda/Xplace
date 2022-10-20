#include <torch/extension.h>

void dct2_fft2_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf);
void idct2_fft2_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf);
void idct_idxst_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf);
void idxst_idct_forward_cuda(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf);

void dct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    dct2_fft2_forward_cuda(x, expkM, expkN, out, buf);
}

void idct2_fft2_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    idct2_fft2_forward_cuda(x, expkM, expkN, out, buf);
}

void idct_idxst_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    idct_idxst_forward_cuda(x, expkM, expkN, out, buf);
}

void idxst_idct_forward(at::Tensor x, at::Tensor expkM, at::Tensor expkN, at::Tensor out, at::Tensor buf) {
    idxst_idct_forward_cuda(x, expkM, expkN, out, buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dct2_fft2", &dct2_fft2_forward, "DCT2 FFT2D (CUDA)");
    m.def("idct2_fft2", &idct2_fft2_forward, "IDCT2 FFT2D (CUDA)");
    m.def("idct_idxst", &idct_idxst_forward, "IDCT IDXST FFT2D (CUDA)");
    m.def("idxst_idct", &idxst_idct_forward, "IDXST IDCT FFT2D (CUDA)");
}
