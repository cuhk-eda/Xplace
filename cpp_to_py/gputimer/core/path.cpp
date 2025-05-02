
#include "GPUTimer.h"
#include "gputimer/db/GTDatabase.h"

using std::setw;
using std::string;
using std::endl;
using std::cout;
using std::tuple;

namespace gt {

// ------------------------------------------------------------------------------------------------------------------------
// Report timing paths
//
tuple<vector<int64_t>, vector<float>, vector<float>> GPUTimer::report_path(int ep_idx, int el, bool verbose) {
    if (!pin_slacks.numel()) {
        pin_slacks = torch::zeros_like(timing_raw_db.pinAT, torch::dtype(torch::kFloat32).device(timing_raw_db.pinAT.device()));
        auto s1 = timing_raw_db.pinAT - timing_raw_db.pinRAT;
        auto s2 = timing_raw_db.pinRAT - timing_raw_db.pinAT;
        pin_slacks.index({"...", torch::indexing::Slice(0, 2)}).data().copy_(s1.index({"...", torch::indexing::Slice(0, 2)}));
        pin_slacks.index({"...", torch::indexing::Slice(2, 4)}).data().copy_(s2.index({"...", torch::indexing::Slice(2, 4)}));
    }
    int worst_ep_idx;
    int worst_ep_i;
    if (ep_idx == -1) {
        auto ep_slacks = torch::nan_to_num(pin_slacks.index_select(0, timing_raw_db.endpoints_id), FLT_MAX);
        if (el != -1) ep_slacks = ep_slacks.index({"...", torch::indexing::Slice(2 * el, 2 * (el + 1))});

        auto [slack_elw, order] = torch::min(ep_slacks, 1);
        auto worst_ep = torch::argmin(slack_elw).item<int>();
        worst_ep_idx = timing_raw_db.endpoints_id[worst_ep].item<int>();
        worst_ep_i = torch::argmin(ep_slacks[worst_ep]).item<int>();
    } else {
        worst_ep_idx = ep_idx;
        worst_ep_i = torch::argmin(pin_slacks[worst_ep_idx]).item<int>();
    }
    worst_ep_i = el == -1 ? worst_ep_i : worst_ep_i + 2 * el;

    int cur = worst_ep_idx;
    int to_i = worst_ep_i;
    vector<long> path;
    vector<float> path_at;
    vector<float> path_rat;
    vector<float> to_delay;
    vector<float> path_slack;

    while (cur != -1) {
        int prev = timing_raw_db.at_prefix_pin[cur][to_i].item<int>();
        int arc_id = timing_raw_db.at_prefix_arc[cur][to_i].item<int>();
        int from_i = timing_raw_db.at_prefix_attr[cur][to_i].item<int>();
        int from_el = from_i >> 1;
        int to_el = to_i >> 1;

        int arc_i = (from_i << 1) + (to_i & 0b1);
        float at = 0;
        float rat = 0;
        float delay = 0;
        float slack = 0;
        if (prev != -1) {
            at = timing_raw_db.pinAT[cur][to_i].item<float>();
            rat = timing_raw_db.pinRAT[cur][to_i].item<float>();
            delay = timing_raw_db.arcDelay[arc_id][arc_i].item<float>();
            slack = pin_slacks[cur][to_i].item<float>();
        }
        path.push_back(cur);
        path_at.push_back(at);
        path_rat.push_back(rat);
        to_delay.push_back(delay);
        path_slack.push_back(slack);

        cur = prev;
        to_i = from_i;
    }

    if (verbose) {
        cout << std::fixed << std::setprecision(3);
        cout << '\n' << std::setw(10) << "Type" << std::setw(10) << "Delay" << std::setw(10) << "AT" << std::setw(10) << "RAT" << std::setw(10) << "Slack" << std::setw(10) << "Pin" << '\n';
        
        for (int i = path.size() - 1; i >= 0; i--) {
            int cur = path[i];
            cout << setw(10) << "pin " << setw(10) << to_delay[i] << setw(10) << path_at[i] << setw(10) << path_rat[i] << setw(10) << path_slack[i];
            std::fill_n(std::ostream_iterator<char>(cout), 3, ' ');
            cout << gtdb.pin_names[cur] << '\n';
        }
    }

    return {path, path_at, to_delay};
}

vector<vector<int64_t>> GPUTimer::report_K_path(int K, bool verbose) {
    if (!pin_slacks.numel()) {
        pin_slacks = torch::zeros_like(timing_raw_db.pinAT, torch::dtype(torch::kFloat32).device(timing_raw_db.pinAT.device()));
        auto s1 = timing_raw_db.pinAT - timing_raw_db.pinRAT;
        auto s2 = timing_raw_db.pinRAT - timing_raw_db.pinAT;
        pin_slacks.index({"...", torch::indexing::Slice(0, 2)}).data().copy_(s1.index({"...", torch::indexing::Slice(0, 2)}));
        pin_slacks.index({"...", torch::indexing::Slice(2, 4)}).data().copy_(s2.index({"...", torch::indexing::Slice(2, 4)}));
    }
    auto [endpoints_id, tmp1] = torch::_unique(timing_raw_db.endpoints_id);
    auto ep_slacks = torch::nan_to_num(pin_slacks.index_select(0, endpoints_id));
    auto [ep_slack_elw, ep_i_indices] = torch::min(ep_slacks, 1);
    auto [ep_slack_elw_ordered, indices] = torch::sort(ep_slack_elw, false);

    indices = indices.contiguous();
    ep_i_indices = ep_i_indices.contiguous();
    endpoints_id = endpoints_id.contiguous();

    vector<vector<int64_t>> paths;

    for (int i = 0; i < K; i++) {
        auto [path, path_at, to_delay] = report_path(endpoints_id[indices[i]].item<int>(), false);
        paths.push_back(path);
    }
    return paths;
}

// ------------------------------------------------------------------------------------------------------------------------
// Report timing paths
//
tuple<torch::Tensor, torch::Tensor> explore_path(index_type* at_prefix_pin,
                                                 index_type* at_prefix_arc,
                                                 int* at_prefix_attr,
                                                 float* pinAT,
                                                 int* arc_types,
                                                 float* arcDelay,
                                                 torch::Tensor indices,
                                                 torch::Tensor ep_i_indices,
                                                 torch::Tensor endpoints_id,
                                                 int num_pins,
                                                 int K,
                                                 bool deterministic);

tuple<torch::Tensor, torch::Tensor> GPUTimer::report_criticality(int K, bool verbose, bool deterministic) {
    auto [endpoints_id, tmp1] = torch::_unique(timing_raw_db.endpoints_id);
    auto ep_slacks = torch::nan_to_num(pin_slacks.index_select(0, endpoints_id));
    auto [ep_slack_elw, ep_i_indices] = torch::min(ep_slacks, 1);
    auto [ep_slack_elw_ordered, indices] = torch::sort(ep_slack_elw, false);

    indices = indices.contiguous();
    ep_i_indices = ep_i_indices.contiguous();
    endpoints_id = endpoints_id.contiguous();

    auto [from_pin_delay, pin_visited] = explore_path(
        at_prefix_pin, at_prefix_arc, at_prefix_attr, pinAT, arc_types, arcDelay, indices, ep_i_indices, endpoints_id, num_pins, K, deterministic);

    return {from_pin_delay, pin_visited};
}

tuple<torch::Tensor, torch::Tensor> GPUTimer::report_criticality_threshold(float thrs, bool verbose, bool deterministic) {
    auto [endpoints_id, tmp1] = torch::_unique(timing_raw_db.endpoints_id);
    auto ep_slacks = torch::nan_to_num(pin_slacks.index_select(0, endpoints_id));
    auto [ep_slack_elw, ep_i_indices] = torch::min(ep_slacks, 1);
    auto [ep_slack_elw_ordered, indices] = torch::sort(ep_slack_elw, false);

    auto threshold = thrs * ep_slack_elw_ordered[0];
    int K = (ep_slack_elw_ordered < threshold).sum().item<int>();

    indices = indices.contiguous();
    ep_i_indices = ep_i_indices.contiguous();
    endpoints_id = endpoints_id.contiguous();

    auto [from_pin_delay, pin_visited] = explore_path(
        at_prefix_pin, at_prefix_arc, at_prefix_attr, pinAT, arc_types, arcDelay, indices, ep_i_indices, endpoints_id, num_pins, K, deterministic);

    return {from_pin_delay, pin_visited};
}


}