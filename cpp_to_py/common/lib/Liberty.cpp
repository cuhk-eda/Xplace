

#include "Liberty.h"

namespace gt {

void CellLib::uncomment(std::vector<char>& buffer) {
    auto fsize = buffer.size() > 0 ? buffer.size() - 1 : 0;

    for (size_t i = 0; i < fsize; ++i) {
        // Block comment
        if (buffer[i] == '/' && buffer[i + 1] == '*') {
            buffer[i] = buffer[i + 1] = ' ';
            for (i = i + 2; i < fsize; buffer[i++] = ' ') {
                if (buffer[i] == '*' && buffer[i + 1] == '/') {
                    buffer[i] = buffer[i + 1] = ' ';
                    i = i + 1;
                    break;
                }
            }
        }

        // Line comment
        if (buffer[i] == '/' && buffer[i + 1] == '/') {
            buffer[i] = buffer[i + 1] = ' ';
            for (i = i + 2; i < fsize; ++i) {
                if (buffer[i] == '\n' || buffer[i] == '\r') {
                    break;
                } else
                    buffer[i] = ' ';
            }
        }

        // Pond comment
        if (buffer[i] == '#') {
            buffer[i] = ' ';
            for (i = i + 1; i < fsize; ++i) {
                if (buffer[i] == '\n' || buffer[i] == '\r') {
                    break;
                } else
                    buffer[i] = ' ';
            }
        }
    }
}

void CellLib::tokenize(const std::vector<char>& buf, std::vector<std::string_view>& tokens) {
    static std::string_view dels = "(),:;/#[]{}*\"\\";
    const char* beg = buf.data();
    const char* end = buf.data() + buf.size();

    const char* token{nullptr};
    size_t len{0};

    tokens.clear();

    for (const char* itr = beg; itr != end && *itr != 0; ++itr) {
        if (std::isspace(*itr) || (dels.find(*itr) != std::string_view::npos)) {
            if (len > 0) {  // Add the current token.
                tokens.push_back({token, len});
                token = nullptr;
                len = 0;
            }
            if (*itr == '(' || *itr == ')' || *itr == '{' || *itr == '}') {
                tokens.push_back({itr, 1});
            }
        } else {
            if (len == 0) {
                token = itr;
            }
            ++len;
        }
    }

    if (len > 0) {
        tokens.push_back({token, len});
    }
}

LibertyCell* CellLib::get_cell(const std::string& name) {
    if (auto itr = lib_cells_.find(name); itr == lib_cells_.end()) {
        return nullptr;
    } else {
        return itr->second;
    }
}

int LibertyCell::get_port(const std::string& name) {
    if (auto itr = ports_map_.find(name); itr == ports_map_.end()) {
        return -1;
    } else {
        return itr->second;
    }
}

EnumNameMap<DelayModel> delay_model_name_map = {{DelayModel::generic_cmos, "generic_cmos"},
                                                {DelayModel::table_lookup, "table_lookup"},
                                                {DelayModel::cmos2, "cmos2"},
                                                {DelayModel::piecewise_cmos, "piecewise_cmos"},
                                                {DelayModel::dcm, "dcm"},
                                                {DelayModel::polynomial, "polynomial"},
                                                {DelayModel::unknown, "unknown"}};
DelayModel findDelayModel(const std::string model_name) {
    return delay_model_name_map.find(model_name, DelayModel::unknown);
}

EnumNameMap<CellPortDirection> port_direction_name_map = {{CellPortDirection::input, "input"},
                                                          {CellPortDirection::output, "output"},
                                                          {CellPortDirection::inout, "inout"},
                                                          {CellPortDirection::internal, "internal"},
                                                          {CellPortDirection::unknown, "unknown"}};

CellPortDirection findPortDirection(const std::string dir_name) {
    return port_direction_name_map.find(dir_name, CellPortDirection::unknown);
}

}  // namespace gt