#include "common/common.h"
#include "gpudp/db/dp_torch.h"

namespace dp {

void kReorderCUDA(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int K, int max_iters);
void globalSwapCUDA(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int max_iters, float displacement_region_ratio);
void independentSetMatchingCUDA(
    DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int set_size, int max_iters);

void kReorder(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int K, int max_iters) {
    kReorderCUDA(at_db, num_bins_x, num_bins_y, K, max_iters);
}

void globalSwap(DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int max_iters, float displacement_region_ratio) {
    globalSwapCUDA(at_db, num_bins_x, num_bins_y, batch_size, max_iters, displacement_region_ratio);
}

void independentSetMatching(
    DPTorchRawDB& at_db, int num_bins_x, int num_bins_y, int batch_size, int set_size, int max_iters) {
    independentSetMatchingCUDA(at_db, num_bins_x, num_bins_y, batch_size, set_size, max_iters);
}

}  // namespace dp