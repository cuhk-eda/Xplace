#include "Drawer.h"

bool DrawGlobalPlacement(
    const std::vector<double>& node_pos_x,
    const std::vector<double>& node_pos_y,
    const std::vector<double>& node_size_x,
    const std::vector<double>& node_size_y,
    const std::vector<std::string>& node_name,
    const std::tuple<double, double, double, double>& die_info,
    const std::tuple<double, double>& site_info,
    const std::tuple<double, double>& bin_size_info,
    const std::vector<std::tuple<index_type, index_type, std::string>>& node_types_indices,
    const std::vector<std::tuple<std::string, double, double, double, double>>& ele_type_to_rgba_vec,
    const std::string& filename,
    double width,
    double height,
    const std::vector<std::string>& draw_contents) {

    Drawer drawer(ele_type_to_rgba_vec, filename, width, height, draw_contents);
    bool status = drawer.run(node_pos_x,
               node_pos_y,
               node_size_x,
               node_size_y,
               node_name,
               die_info,
               site_info,
               bin_size_info,
               node_types_indices);
    return status;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("draw", &DrawGlobalPlacement, "Draw placement results"); }
