#include <torch/extension.h>

torch::Tensor hpwl_cuda(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

torch::Tensor hpwl(torch::Tensor pos, torch::Tensor hyperedge_list, torch::Tensor hyperedge_list_end) {
    CHECK_INPUT(pos);
    CHECK_INPUT(hyperedge_list);
    CHECK_INPUT(hyperedge_list_end);

    return hpwl_cuda(pos, hyperedge_list, hyperedge_list_end);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("hpwl", &hpwl, "get HPWL wirelength from hyperedge"); }