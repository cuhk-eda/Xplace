#include "global.h"
class Drawer {
private:
    std::unordered_map<std::string, std::tuple<double, double, double, double>> type_to_rgba;
    std::string format = "";
    std::string filename = "";
    double width = 0.0;
    double height = 0.0;
    std::unordered_set<std::string> contents;

public:
    Drawer(const std::vector<std::tuple<std::string, double, double, double, double>>& ele_type_to_rgba_vec,
           const std::string& filename_,
           double width_,
           double height_,
           const std::vector<std::string>& contents_);
    std::tuple<double, double, double, double> get_colors(const std::string& ele_type);
    bool run(const std::vector<double>& node_pos_x,
             const std::vector<double>& node_pos_y,
             const std::vector<double>& node_size_x,
             const std::vector<double>& node_size_y,
             const std::vector<std::string>& node_name,
             const std::tuple<double, double, double, double>& die_info,
             const std::tuple<double, double>& site_info,
             const std::tuple<double, double>& bin_size_info,
             std::vector<std::tuple<index_type, index_type, std::string>> node_types_indices);
};
