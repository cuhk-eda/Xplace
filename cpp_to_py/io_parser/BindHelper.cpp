#include "BindHelper.h"

namespace Xplace {

void bindGPDatabase(pybind11::module& m) {
    pybind11::bind_vector<std::vector<bool>>(m, "BoolList");
    pybind11::bind_vector<std::vector<gp::coord_type>>(m, "CoordList");
    pybind11::bind_vector<std::vector<gp::index_type>>(m, "IndexList");

    pybind11::class_<gp::Basic>(m, "Basic")
        .def(pybind11::init<>())
        .def("id", &gp::Basic::getId)
        .def("name", &gp::Basic::getName)
        .def("__repr__", &gp::Basic::getName);

    pybind11::class_<gp::GPNode, gp::Basic>(m, "Node")
        .def(pybind11::init<>())
        .def("lx", &gp::GPNode::getLx)
        .def("ly", &gp::GPNode::getLy)
        .def("hx", &gp::GPNode::getHx)
        .def("hy", &gp::GPNode::getHy)
        .def("width", &gp::GPNode::getWidth)
        .def("height", &gp::GPNode::getHeight)
        .def("orient", &gp::GPNode::getOrient)
        .def("node_type", &gp::GPNode::getNodeType)
        .def("pinIds", &gp::GPNode::pins);
    pybind11::bind_vector<std::vector<gp::GPNode>>(m, "NodeList");

    pybind11::class_<gp::GPPin, gp::Basic>(m, "Pin")
        .def(pybind11::init<>())
        .def("rel_lx", &gp::GPPin::getRelLx)
        .def("rel_ly", &gp::GPPin::getRelLy)
        .def("width", &gp::GPPin::getWidth)
        .def("height", &gp::GPPin::getHeight)
        .def("direction", &gp::GPPin::getDirection)
        .def("pin_type", &gp::GPPin::getType)
        .def("nodeId", &gp::GPPin::getParNodeId)
        .def("netId", &gp::GPPin::getParNetId);
    pybind11::bind_vector<std::vector<gp::GPPin>>(m, "PinList");

    pybind11::class_<gp::GPNet, gp::Basic>(m, "Net").def(pybind11::init<>()).def("pinIds", &gp::GPNet::pins);
    pybind11::bind_vector<std::vector<gp::GPNet>>(m, "NetList");

    pybind11::class_<gp::GPDatabase, std::shared_ptr<gp::GPDatabase>>(m, "GPDatabase")
        .def(pybind11::init<std::shared_ptr<db::Database>>())
        .def("setup", &gp::GPDatabase::setup)
        .def("reset", &gp::GPDatabase::reset)
        .def("nodes", &gp::GPDatabase::getNodes)  // NOTE: using py::return_value_policy::reference is dangerous
        .def("pins", &gp::GPDatabase::getPins)
        .def("nets", &gp::GPDatabase::getNets)
        .def("dieInfo", &gp::GPDatabase::getDieInfo, py::return_value_policy::copy)  // dieLX, dieHX, dieLY, dieHY
        .def("coreInfo", &gp::GPDatabase::getCoreInfo, py::return_value_policy::copy)  // coreLX, coreHX, coreLY, coreHY
        .def("siteWidth", &gp::GPDatabase::getSiteWidth, py::return_value_policy::copy)
        .def("siteHeight", &gp::GPDatabase::getSiteHeight, py::return_value_policy::copy)
        .def("m1direction", &gp::GPDatabase::getM1Direction, py::return_value_policy::copy)
        .def("node_type_indices", &gp::GPDatabase::getNodeTypeIndices, py::return_value_policy::copy)
        .def("node_id2node_name", &gp::GPDatabase::getNodeId2NodeName, py::return_value_policy::copy)
        .def("node_lpos_tensor", &gp::GPDatabase::getNodeLPosTensor, py::return_value_policy::move)
        .def("node_cpos_tensor", &gp::GPDatabase::getNodeCPosTensor, py::return_value_policy::move)
        .def("node_size_tensor", &gp::GPDatabase::getNodeSizeTensor, py::return_value_policy::move)
        .def("pin_rel_lpos_tensor", &gp::GPDatabase::getPinRelLPosTensor, py::return_value_policy::move)
        .def("pin_rel_cpos_tensor", &gp::GPDatabase::getPinRelCPosTensor, py::return_value_policy::move)
        .def("pin_size_tensor", &gp::GPDatabase::getPinSizeTensor, py::return_value_policy::move)
        .def("pin_id2node_id_tensor", &gp::GPDatabase::getPinId2NodeIdTensor, py::return_value_policy::move)
        .def("pin_id2net_id_tensor", &gp::GPDatabase::getPinId2NetIdTensor, py::return_value_policy::move)
        .def("hyperedge_info_tensor", &gp::GPDatabase::getHyperedgeInfoTensor, py::return_value_policy::move)
        .def("node2pin_info_tensor", &gp::GPDatabase::getNode2PinInfoTensor, py::return_value_policy::move)
        .def("region_info_tensor", &gp::GPDatabase::getRegionInfoTensor, py::return_value_policy::move)
        .def("snet_info_tensor", &gp::GPDatabase::getSnetInfoTensor, py::return_value_policy::move)
        .def("apply_node_cpos", &gp::GPDatabase::applyNodeCPos)
        .def("apply_node_lpos", &gp::GPDatabase::applyNodeLPos)
        .def("write_placement", &gp::GPDatabase::writePlacement);
}

}  // namespace Xplace