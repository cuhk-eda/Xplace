// Created by Lixin LIU on 2021/09/14
// Adapt from https://github.com/limbo018/DREAMPlace/blob/master/dreamplace/ops/draw_place/src/PlaceDrawer.h

#include "Drawer.h"

Drawer::Drawer(const std::vector<std::tuple<std::string, double, double, double, double>>& ele_type_to_rgba_vec,
               const std::string& filename_,
               double width_,
               double height_,
               const std::vector<std::string>& contents_) {
    filename = filename_;
    std::string format_ = filename_.substr(filename_.size() - 4);
    if (format_ == ".png") {
        format = "png";
    } else if (format_ == ".pdf") {
        format = "pdf";
    } else if (format_ == ".svg") {
        format = "svg";
    } else if (format_ == ".eps") {
        format = "eps";
    }
    width = width_;
    height = height_;

    for (auto content_ : contents_) {
        contents.emplace(content_);
    }

    // Customized value in type_to_rgba
    for (auto [ele_type, r, g, b, a] : ele_type_to_rgba_vec) {
        auto it = type_to_rgba.find(ele_type);
        if (it == type_to_rgba.end()) {
            type_to_rgba.emplace(ele_type, std::make_tuple(r, g, b, a));
        }
    }
    // Default value in type_to_rgba
    std::string ele_type;
    ele_type = "Background";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(1.0, 1.0, 1.0, 1.0));
    }
    ele_type = "Die";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(1.0, 1.0, 1.0, 1.0));
    }
    ele_type = "Bin";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.3, 0.3, 0.3, 1.0));
    }
    ele_type = "Grid";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.3, 0.3, 0.3, 1.0));
    }
    ele_type = "Mov";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.475, 0.706, 0.718, 0.8));
    }
    ele_type = "Fix";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.878, 0.365, 0.365, 0.8));
    }
    ele_type = "IOPin";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.878, 0.365, 0.365, 0.8));
    }
    ele_type = "Blkg";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.878, 0.365, 0.365, 0.8));
    }
    ele_type = "FloatIOPin";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.878, 0.365, 0.365, 0.8));
    }
    ele_type = "FloatFix";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.878, 0.365, 0.365, 0.8));
    }
    ele_type = "FloatMov";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.082, 0.176, 0.208, 0.8));
    }
    ele_type = "Filler";
    if (type_to_rgba.find(ele_type) == type_to_rgba.end()) {
        type_to_rgba.emplace(ele_type, std::make_tuple(0.082, 0.176, 0.208, 0.8));
    }
}

std::tuple<double, double, double, double> Drawer::get_colors(const std::string& ele_type) {
    auto it = type_to_rgba.find(ele_type);
    if (it != type_to_rgba.end()) {
        return it->second;
    }
    return std::make_tuple(0.0, 0.0, 0.0, 1.0);
}

bool Drawer::run(const std::vector<double>& node_pos_x,   // after die scale
                 const std::vector<double>& node_pos_y,   // after die scale
                 const std::vector<double>& node_size_x,  // after die scale
                 const std::vector<double>& node_size_y,  // after die scale
                 const std::vector<std::string>& node_name,
                 const std::tuple<double, double, double, double>& die_info,  // after die scale
                 const std::tuple<double, double>& site_info,
                 const std::tuple<double, double>& bin_size_info,  // after die scale
                 std::vector<std::tuple<index_type, index_type, std::string>> node_types_indices) {
    // init cairo
    cairo_surface_t* cs;
    if (format == "png") {
        cs = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    } else if (format == "pdf") {
        cs = cairo_pdf_surface_create(filename.c_str(), width, height);
    } else if (format == "svg") {
        cs = cairo_svg_surface_create(filename.c_str(), width, height);
    } else if (format == "eps") {
        cs = cairo_ps_surface_create(filename.c_str(), width, height);
    } else {
        std::cout << "Unsupported file format " << format << ". Please use png/pdf/svg/eps instead." << std::endl;
        return false;
    }

    auto [die_lx, die_hx, die_ly, die_hy] = die_info;
    auto [site_width, site_height] = site_info;
    auto [bin_width, bin_height] = bin_size_info;

    double w_ratio = width / (die_hx - die_lx);
    double h_ratio = height / (die_hy - die_ly);
    cairo_t* c;
    cairo_text_extents_t extents;
    cairo_matrix_t font_reflection_matrix;
    char buf[32];
    c = cairo_create(cs);
    cairo_save(c);
    cairo_translate(c, 0 - die_lx * w_ratio, height + die_ly * h_ratio);
    cairo_scale(c, w_ratio, -h_ratio);

    auto set_rgba = [&](std::string type) {
        auto [r, g, b, a] = get_colors(type);
        cairo_set_source_rgba(c, r, g, b, a);
    };
    double lineWidth = 0.0001 * std::min(die_hx - die_lx, die_hy - die_ly);
    // die
    cairo_rectangle(c, die_lx, die_ly, (die_hx - die_lx), (die_hy - die_ly));
    set_rgba("Die");
    cairo_fill(c);
    // boundary
    cairo_rectangle(c, die_lx, die_ly, (die_hx - die_lx), (die_hy - die_ly));
    cairo_set_line_width(c, lineWidth);
    cairo_set_source_rgb(c, 0.1, 0.1, 0.1);
    cairo_stroke(c);
    // bin / grid
    cairo_set_line_width(c, lineWidth);
    set_rgba("Bin");
    for (double bx = die_lx; bx < die_hx; bx += bin_width) {
        cairo_move_to(c, bx, die_ly);
        cairo_line_to(c, bx, die_hy);
        cairo_stroke(c);
    }
    for (double by = die_ly; by < die_hy; by += bin_height) {
        cairo_move_to(c, die_lx, by);
        cairo_line_to(c, die_hx, by);
        cairo_stroke(c);
    }

    cairo_set_line_width(c, lineWidth);
    cairo_select_font_face(c, "Sans", CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_NORMAL);

    if (contents.find("Nodes") != contents.end()) {
        int num_nodes = node_pos_x.size();
        if (std::get<1>(node_types_indices.back()) <= num_nodes) {
            // enable filler
            index_type last_idx = std::get<1>(node_types_indices.back());
            auto tmp = std::make_tuple(last_idx, num_nodes, "Filler");
            node_types_indices.emplace_back(tmp);
        }
        // draw mov last
        std::rotate(node_types_indices.begin(), node_types_indices.begin() + 1, node_types_indices.end());
        bool draw_node_text = false;
        if (contents.find("NodesText") != contents.end()) {
            draw_node_text = true;
        }
        bool draw_node_bd = false;
        if (contents.find("NodesBoundary") != contents.end()) {
            draw_node_bd = true;
        }
        for (auto [start_idx, end_idx, node_type] : node_types_indices) {
            // std::cout << node_type << std::endl;
            for (index_type i = start_idx; i < end_idx; i++) {
                if (i >= num_nodes) {
                    break;
                }
                double node_lx = node_pos_x[i] - node_size_x[i] / 2;
                double node_ly = node_pos_y[i] - node_size_y[i] / 2;
                cairo_rectangle(c, node_lx, node_ly, node_size_x[i], node_size_y[i]);
                set_rgba(node_type);
                cairo_fill(c);
                if (draw_node_bd) {
                    cairo_rectangle(c, node_lx, node_ly, node_size_x[i], node_size_y[i]);
                    cairo_set_source_rgb(c, 0.1, 0.1, 0.1);
                    cairo_stroke(c);
                }
                if (draw_node_text) {
                    cairo_matrix_t font_reflection_matrix;
                    sprintf(buf, "%s", node_name[i].c_str());
                    double rotate = 0;
                    double font_size = node_size_y[i] / 5;
                    if (node_size_x[i] < node_size_y[i]) {
                        rotate = 3.1415926 / 2;
                        font_size = node_size_x[i] / 5;
                    }
                    cairo_set_font_size(c, font_size);
                    cairo_set_source_rgb(c, 0.2, 0.2, 0.2);
                    cairo_get_font_matrix(c, &font_reflection_matrix);
                    font_reflection_matrix.yy = font_reflection_matrix.yy * -1 * w_ratio / h_ratio;
                    cairo_matrix_rotate(&font_reflection_matrix, rotate);
                    cairo_set_font_matrix(c, &font_reflection_matrix);
                    cairo_text_extents(c, buf, &extents);
                    cairo_move_to(c,
                                  (node_lx + node_size_x[i] / 2) - (extents.width / 2 + extents.x_bearing),
                                  (node_ly + node_size_y[i] / 2) - (extents.height / 2 + extents.y_bearing));
                    cairo_show_text(c, buf);
                }
            }
        }
    }

    cairo_restore(c);
    cairo_show_page(c);
    cairo_destroy(c);
    // destory cairo
    cairo_surface_flush(cs);
    if (format == "png") cairo_surface_write_to_png(cs, filename.c_str());
    cairo_surface_destroy(cs);
    return true;
}