#pragma once

#include <vector>
#include "common/lib/Lut.h"
#include "common/lib/Timing.h"

using std::vector;

namespace gt {

template <typename T>
__device__ int lower_bound(T *arr, int size, T val) {
    int l = 0, r = size - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (arr[m] < val)
            l = m + 1;
        else
            r = m;
    }
    return l;
}

template <typename T>
__device__ float interpolate(T x1, T x2, T y1, T y2, T x) {
    if (x1 == x2) return y1;
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
}

class GPULutAllocator {
public:
    // LUT attributes
    int num_luts_in_timing = 6;
    int num_luts;
    int x_size = 0, y_size = 0, table_size = 0;

    int *num_x, *num_y, *num_table;
    float *x_array, *y_array, *table_array;
    size_t *x_offset, *y_offset, *table_offset;
    bool *allocated;

    int *d_num_x, *d_num_y, *d_num_table;
    float *d_x_array, *d_y_array, *d_table_array;
    size_t *d_x_offset, *d_y_offset, *d_table_offset;
    bool *d_allocated;

    // Timing attributes
    int num_timings;
    int *timing_sense;
    int *lut_template_var;
    bool *is_rising_edge_triggered, *is_falling_edge_triggered, *is_constraint;

    int *d_timing_sense;
    int *d_lut_template_var;
    bool *d_is_rising_edge_triggered, *d_is_falling_edge_triggered, *d_is_constraint;

public:
    GPULutAllocator() = default;
    __host__ __forceinline__ void AllocateBatch(vector<TimingArc *> timings) {
        auto check_lut = [&](Lut *lut) {
            if (!lut) return;
            if (lut->set_) {
                x_size += lut->indices1.size();
                y_size += lut->indices2.size();
                table_size += lut->table.size();
            }
        };
        num_timings = timings.size();
        is_rising_edge_triggered = new bool[num_timings];
        is_falling_edge_triggered = new bool[num_timings];
        is_constraint = new bool[num_timings];
        timing_sense = new int[num_timings];
        num_timings = 0;
        for (auto timing_ptr : timings) {
            auto &timing = *timing_ptr;
            check_lut(timing.cell_delay_[0]);
            check_lut(timing.cell_delay_[1]);
            check_lut(timing.transition_[0]);
            check_lut(timing.transition_[1]);
            check_lut(timing.constraint_[0]);
            check_lut(timing.constraint_[1]);
            is_rising_edge_triggered[num_timings] = timing.is_rising_edge_triggered();
            is_falling_edge_triggered[num_timings] = timing.is_falling_edge_triggered();
            is_constraint[num_timings] = timing.is_constraint();
            if (timing.timing_sense_ != TimingSense::unknown)
                timing_sense[num_timings] = static_cast<int>(timing.timing_sense_);
            else
                timing_sense[num_timings] = -1;
            num_timings++;
        }

        num_luts = num_luts_in_timing * timings.size();  // delay/transitions/constraints * rise/fall
        x_array = new float[x_size];
        y_array = new float[y_size];
        table_array = new float[table_size];
        num_x = new int[num_luts];
        num_y = new int[num_luts];
        num_table = new int[num_luts];
        x_offset = new size_t[num_luts + 1];
        y_offset = new size_t[num_luts + 1];
        table_offset = new size_t[num_luts + 1];
        lut_template_var = new int[num_luts * 2];  // 0:capacitance/1:transition/2:constraint_transition/3:related_transition/4:input_transition
        allocated = new bool[num_luts];
        x_offset[0] = 0;
        y_offset[0] = 0;
        table_offset[0] = 0;

        num_luts = 0;
        auto insert_lut = [&](Lut *lut) {
            if (lut->set_) {
                num_x[num_luts] = lut->indices1.size();
                num_y[num_luts] = lut->indices2.size();
                num_table[num_luts] = lut->table.size();
                x_offset[num_luts + 1] = x_offset[num_luts] + num_x[num_luts];
                y_offset[num_luts + 1] = y_offset[num_luts] + num_y[num_luts];
                table_offset[num_luts + 1] = table_offset[num_luts] + num_table[num_luts];
                memcpy(x_array + x_offset[num_luts], lut->indices1.data(), lut->indices1.size() * sizeof(float));
                memcpy(y_array + y_offset[num_luts], lut->indices2.data(), lut->indices2.size() * sizeof(float));
                memcpy(table_array + table_offset[num_luts], lut->table.data(), lut->table.size() * sizeof(float));

                if (lut->lut_template) {
                    if (lut->lut_template->variable1)
                        lut_template_var[num_luts * 2] = static_cast<int>(lut->lut_template->variable1.value());
                    else
                        lut_template_var[num_luts * 2] = -1;
                    if (lut->lut_template->variable2)
                        lut_template_var[num_luts * 2 + 1] = static_cast<int>(lut->lut_template->variable2.value());
                    else
                        lut_template_var[num_luts * 2 + 1] = -1;
                } else {
                    lut_template_var[num_luts * 2] = -1;
                    lut_template_var[num_luts * 2 + 1] = -1;
                }
                allocated[num_luts] = true;
            } else {
                num_x[num_luts] = 0;
                num_y[num_luts] = 0;
                num_table[num_luts] = 0;
                x_offset[num_luts + 1] = x_offset[num_luts];
                y_offset[num_luts + 1] = y_offset[num_luts];
                table_offset[num_luts + 1] = table_offset[num_luts];
                lut_template_var[num_luts * 2] = -1;      // var1
                lut_template_var[num_luts * 2 + 1] = -1;  // var2
                allocated[num_luts] = false;
            }
            num_luts++;
        };
        for (auto timing_ptr : timings) {
            auto &timing = *timing_ptr;
            insert_lut(timing.cell_delay_[0]);
            insert_lut(timing.cell_delay_[1]);
            insert_lut(timing.transition_[0]);
            insert_lut(timing.transition_[1]);
            insert_lut(timing.constraint_[0]);
            insert_lut(timing.constraint_[1]);
        }
    }
    __host__ __forceinline__ void CopyToGPU() {
        cudaMalloc(&d_x_array, x_size * sizeof(float));
        cudaMalloc(&d_y_array, y_size * sizeof(float));
        cudaMalloc(&d_table_array, table_size * sizeof(float));
        cudaMalloc(&d_num_x, num_luts * sizeof(int));
        cudaMalloc(&d_num_y, num_luts * sizeof(int));
        cudaMalloc(&d_num_table, num_luts * sizeof(int));
        cudaMalloc(&d_x_offset, (num_luts + 1) * sizeof(size_t));
        cudaMalloc(&d_y_offset, (num_luts + 1) * sizeof(size_t));
        cudaMalloc(&d_table_offset, (num_luts + 1) * sizeof(size_t));
        cudaMalloc(&d_allocated, num_luts * sizeof(bool));
        cudaMalloc(&d_is_rising_edge_triggered, num_timings * sizeof(bool));
        cudaMalloc(&d_is_falling_edge_triggered, num_timings * sizeof(bool));
        cudaMalloc(&d_is_constraint, num_timings * sizeof(bool));
        cudaMalloc(&d_timing_sense, num_timings * sizeof(int));
        cudaMalloc(&d_lut_template_var, 2 * num_luts * sizeof(int));

        cudaMemcpy(d_x_array, x_array, x_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_array, y_array, y_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table_array, table_array, table_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_num_x, num_x, num_luts * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_num_y, num_y, num_luts * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_num_table, num_table, num_luts * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x_offset, x_offset, (num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_offset, y_offset, (num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_table_offset, table_offset, (num_luts + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_allocated, allocated, num_luts * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_is_rising_edge_triggered, is_rising_edge_triggered, num_timings * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_is_falling_edge_triggered, is_falling_edge_triggered, num_timings * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_is_constraint, is_constraint, num_timings * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_timing_sense, timing_sense, num_timings * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_template_var, lut_template_var, 2 * num_luts * sizeof(int), cudaMemcpyHostToDevice);
    }
    __host__ __forceinline__ void CopyToGPU(GPULutAllocator *d_gpuluts) {
        cudaMemcpy(&(d_gpuluts->d_num_x), &d_num_x, sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_num_y), &d_num_y, sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_num_table), &d_num_table, sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_x_array), &d_x_array, sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_y_array), &d_y_array, sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_table_array), &d_table_array, sizeof(float *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_x_offset), &d_x_offset, sizeof(size_t *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_y_offset), &d_y_offset, sizeof(size_t *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_table_offset), &d_table_offset, sizeof(size_t *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_allocated), &d_allocated, sizeof(bool *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_is_rising_edge_triggered), &d_is_rising_edge_triggered, sizeof(bool *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_is_falling_edge_triggered), &d_is_falling_edge_triggered, sizeof(bool *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_timing_sense), &d_timing_sense, sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_gpuluts->d_lut_template_var), &d_lut_template_var, sizeof(int *), cudaMemcpyHostToDevice);
    }

    __device__ __forceinline__ bool is_input_transition_defined(int timing_id, int irf) {
        if (d_is_rising_edge_triggered[timing_id] && irf != 0) return false;
        if (d_is_falling_edge_triggered[timing_id] && irf != 1) return false;
        return true;
    }

    __device__ __forceinline__ bool is_transition_defined(int timing_id, int irf, int orf) {
        if (!is_input_transition_defined(timing_id, irf)) return false;
        int sense = d_timing_sense[timing_id];

        if (sense != -1) {
            switch (sense) {
                case 1:
                    if (irf != orf) return false;
                    break;
                case 2:
                    if (irf == orf) return false;
                    break;
                default:
                    break;
            }
        }

        return true;
    }

    __device__ __forceinline__ float lut(int in_timing_lut, float x, float y) {
        if (d_num_x[in_timing_lut] < 1 || d_num_y[in_timing_lut] < 1) {
            // return std::nullopt;
            return nanf("");
        }

        if (d_num_table[in_timing_lut] == 1) {
            // return std::nullopt;
            return d_table_array[d_table_offset[in_timing_lut]];
        }
        int x_idx[2], y_idx[2];

        x_idx[1] = lower_bound<float>(d_x_array + d_x_offset[in_timing_lut], d_num_x[in_timing_lut], x);
        y_idx[1] = lower_bound<float>(d_y_array + d_y_offset[in_timing_lut], d_num_y[in_timing_lut], y);

        x_idx[1] = max(1, min(d_num_x[in_timing_lut] - 1, x_idx[1]));
        y_idx[1] = max(1, min(d_num_y[in_timing_lut] - 1, y_idx[1]));
        x_idx[0] = x_idx[1] - 1;
        y_idx[0] = y_idx[1] - 1;
        if (d_num_x[in_timing_lut] == 1) x_idx[1] = 0;
        if (d_num_y[in_timing_lut] == 1) y_idx[1] = 0;

        // interpolation
        float numeric[2];
        numeric[0] = interpolate<float>(d_x_array[d_x_offset[in_timing_lut] + x_idx[0]],
                                        d_x_array[d_x_offset[in_timing_lut] + x_idx[1]],
                                        d_table_array[d_table_offset[in_timing_lut] + x_idx[0] * d_num_y[in_timing_lut] + y_idx[0]],
                                        d_table_array[d_table_offset[in_timing_lut] + x_idx[1] * d_num_y[in_timing_lut] + y_idx[0]],
                                        x);
        numeric[1] = interpolate<float>(d_x_array[d_x_offset[in_timing_lut] + x_idx[0]],
                                        d_x_array[d_x_offset[in_timing_lut] + x_idx[1]],
                                        d_table_array[d_table_offset[in_timing_lut] + x_idx[0] * d_num_y[in_timing_lut] + y_idx[1]],
                                        d_table_array[d_table_offset[in_timing_lut] + x_idx[1] * d_num_y[in_timing_lut] + y_idx[1]],
                                        x);

        return interpolate<float>(d_y_array[d_y_offset[in_timing_lut] + y_idx[0]], d_y_array[d_y_offset[in_timing_lut] + y_idx[1]], numeric[0], numeric[1], y);
    }

    __device__ __forceinline__ float query(int timing_id, int irf, int orf, float slew_or_related, float load_or_constraint, int type) {  // 0:cell/1:trans/3:constraint
        if (!is_transition_defined(timing_id, irf, orf)) {
            // return std::nullopt;
            return nanf("");
        }

        int in_timing_lut = num_luts_in_timing * timing_id + orf + type * 2;
        in_timing_lut = d_allocated[in_timing_lut] ? in_timing_lut : -1;

        if (in_timing_lut == -1) {
            // return std::nullopt;
            return nanf("");
        }

        float val1{0.0f}, val2{0.0f};

        if (type == 0 || type == 1) {
            switch (d_lut_template_var[in_timing_lut * 2]) {
                case 0:  // LutVar::TOTAL_OUTPUT_NET_CAPACITANCE
                    if (d_lut_template_var[in_timing_lut * 2 + 1] != -1) {
                        assert(d_lut_template_var[in_timing_lut * 2 + 1] == 1);  // LutVar::INPUT_NET_TRANSITION
                    }
                    val1 = load_or_constraint;
                    val2 = slew_or_related;
                    break;
                case 1:  // LutVar::INPUT_NET_TRANSITION
                    if (d_lut_template_var[in_timing_lut * 2 + 1] != -1) {
                        assert(d_lut_template_var[in_timing_lut * 2 + 1] == 0);  // LutVar::TOTAL_OUTPUT_NET_CAPACITANCE
                    }
                    val1 = slew_or_related;
                    val2 = load_or_constraint;
                    break;
                default:
                    // printf("Invalid lut template variable\n");
                    break;
            }
        } else if (type == 2) {
            switch (d_lut_template_var[in_timing_lut * 2]) {
                case 2:  // LutVar::CONSTRAINED_PIN_TRANSITION
                    if (d_lut_template_var[in_timing_lut * 2 + 1] != -1) {
                        assert(d_lut_template_var[in_timing_lut * 2 + 1] == 3);  // LutVar::RELATED_PIN_TRANSITION
                    }
                    val1 = load_or_constraint;
                    val2 = slew_or_related;
                    break;
                case 3:  // LutVar::RELATED_PIN_TRANSITION
                    if (d_lut_template_var[in_timing_lut * 2 + 1] != -1) {
                        assert(d_lut_template_var[in_timing_lut * 2 + 1] == 2);  // LutVar::CONSTRAINED_PIN_TRANSITION
                    }
                    val1 = slew_or_related;
                    val2 = load_or_constraint;
                    break;
                default:
                    // printf("Invalid lut template variable\n");
                    break;
            }
        }

        return lut(in_timing_lut, val1, val2);
    }

    __host__ __forceinline__ void freeMem() {
        if (allocated) {
            logger.info("destruct gputiming");
            delete[] num_x;
            delete[] num_y;
            delete[] num_table;
            delete[] x_array;
            delete[] y_array;
            delete[] table_array;
            delete[] x_offset;
            delete[] y_offset;
            delete[] table_offset;
            delete[] allocated;
            delete[] is_rising_edge_triggered;
            delete[] is_falling_edge_triggered;
            delete[] is_constraint;
            delete[] timing_sense;
            cudaFree(d_num_x);
            cudaFree(d_num_y);
            cudaFree(d_num_table);
            cudaFree(d_x_array);
            cudaFree(d_y_array);
            cudaFree(d_table_array);
            cudaFree(d_x_offset);
            cudaFree(d_y_offset);
            cudaFree(d_table_offset);
            cudaFree(d_allocated);
            cudaFree(d_is_rising_edge_triggered);
            cudaFree(d_is_falling_edge_triggered);
            cudaFree(d_is_constraint);
            cudaFree(d_timing_sense);
        }
    }

    __host__ ~GPULutAllocator() { freeMem(); }
};

}  // namespace gt