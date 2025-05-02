#pragma once

#include <map>
#include <string>
#include <string_view>

namespace gt {

using std::initializer_list;
using std::map;
using std::pair;
using std::string;
using std::string_view;

template <class ENUM>
class EnumNameMap {
public:
    EnumNameMap(initializer_list<pair<const ENUM, string>> enum_names);
    const string find(ENUM key) const;
    ENUM find(string name, ENUM unknown_key) const;
    void find(string name, ENUM &key, bool &exists) const;

private:
    map<ENUM, string> enum_map_;
    map<string, ENUM> name_map_;
};

template <class ENUM>
EnumNameMap<ENUM>::EnumNameMap(initializer_list<pair<const ENUM, string>> enum_names) : enum_map_(enum_names) {
    for (const auto &[key, name] : enum_map_) name_map_[name] = key;
}

template <class ENUM>
const string EnumNameMap<ENUM>::find(ENUM key) const {
    auto find_iter = enum_map_.find(key);
    if (find_iter != enum_map_.end())
        return find_iter->second;
    else
        return nullptr;
}

template <class ENUM>
void EnumNameMap<ENUM>::find(string name, ENUM &key, bool &exists) const {
    auto find_iter = name_map_.find(name);
    if (find_iter != name_map_.end()) {
        key = find_iter->second;
        exists = true;
    } else
        exists = false;
}

template <class ENUM>
ENUM EnumNameMap<ENUM>::find(string name, ENUM unknown_key) const {
    auto find_iter = name_map_.find(name);
    if (find_iter != name_map_.end())
        return find_iter->second;
    else
        return unknown_key;
}

}  // namespace gt
