#include "GPDatabase.h"

namespace gp {

GPDatabase::~GPDatabase() { logger.info("destruct gpdb"); }

void GPDatabase::addCellNode(index_type cell_id, std::string& node_type) {
    auto cell = database.cells[cell_id];
    nodes.emplace_back(GPNode());
    GPNode& node = nodes.back();
    node.setId(nodes.size() - 1);
    node.setName(cell->name());
    node.setLx(cell->lx());
    node.setLy(cell->ly());
    node.setWidth(cell->width());
    node.setHeight(cell->height());
    node.setOrient(cell->orient());
    node.setNodeType(node_type);
    node.setOriDBId(cell_id);

    node.setRegionId(static_cast<index_type>(cell->region->id));
    regions[node.getRegionId()].addNode(node.getId());

    if (cell->fixed() && cell->ctype()->nonRegularRects().size() > 0) {
        node.setIsPolygonShape(true);
    }
    cell->gpdb_id = nodes.size() - 1;
}

void GPDatabase::addIOPinNode(index_type iopin_id, std::string& node_type) {
    auto iopin = database.iopins[iopin_id];
    nodes.emplace_back(GPNode());
    GPNode& node = nodes.back();
    node.setId(nodes.size() - 1);
    node.setName(iopin->name);
    node.setLx(iopin->x);
    node.setLy(iopin->y);
    node.setWidth(iopin->width());
    node.setHeight(iopin->height());
    node.setOrient(iopin->orient());
    node.setNodeType(node_type);
    node.setOriDBId(iopin_id);

    iopin->gpdb_id = nodes.size() - 1;
}

void GPDatabase::addBlockageNode(index_type blkg_id, std::string& node_type) {
    auto& blockage = database.placeBlockages[blkg_id];
    std::string blockage_name = "Blockage_" + std::to_string(blkg_id);
    nodes.emplace_back(GPNode());
    GPNode& node = nodes.back();
    node.setId(nodes.size() - 1);
    node.setName(blockage_name);
    node.setLx(blockage.lx);
    node.setLy(blockage.ly);
    node.setWidth(blockage.w());
    node.setHeight(blockage.h());
    node.setOrient(-1);  // No orientation for placement blockage
    node.setNodeType(node_type);
    node.setOriDBId(blkg_id);
}

void GPDatabase::addNet(index_type dbnet_id) {
    auto dbnet = database.nets[dbnet_id];
    nets.emplace_back(GPNet());
    GPNet& net = nets.back();
    net.setId(nets.size() - 1);
    net.setName(dbnet->name);
    net.setOriDBId(dbnet_id);
    dbnet->gpdb_id = nets.size() - 1;
    for (auto dbpin : dbnet->pins) {
        assert_msg(dbpin->is_connected, "Pin is not connected!");
        if (dbpin->iopin != nullptr) {
            auto& node = nodes.at(dbpin->iopin->gpdb_id);
            addPin(dbpin, dbpin->iopin->type, node, net, true);
        } else {
            auto& node = nodes.at(dbpin->cell->gpdb_id);
            addPin(dbpin, dbpin->type, node, net, false);
        }
    }
}

void GPDatabase::addPin(db::Pin* dbpin, const db::PinType* pintype, GPNode& node, GPNet& net, bool isIOPin) {
    pins.emplace_back(GPPin());
    GPPin& pin = pins.back();
    pin.setId(pins.size() - 1);

    pin.setRelLx(pintype->boundLX);
    pin.setRelLy(pintype->boundLY);
    pin.setWidth(pintype->getW());
    pin.setHeight(pintype->getH());
    pin.setDirection(pintype->direction());
    pin.setType(pintype->type());
    pin.setParNodeId(node.getId());
    pin.setParNetId(net.getId());
    pin.setOriDBInfo({node.getOriDBId(), isIOPin ? -1 : dbpin->parentCellPinId, net.getOriDBId()});

    node.addPin(pin.getId());
    net.addPin(pin.getId());

    dbpin->gpdb_id = pins.size() - 1;
}

void GPDatabase::addRegion(index_type dbregion_id) {
    auto dbregion = database.regions[dbregion_id];
    regions.emplace_back(GPRegion());
    GPRegion& region = regions.back();
    region.setId(regions.size() - 1);
    region.setName(dbregion->name());
    region.setOriDBId(dbregion_id);
    assert_msg(static_cast<index_type>(dbregion->id) == dbregion_id,
               "Set Region ID incorrectly! Please check origin DB");
    assert_msg(dbregion_id == region.getId(), "Set Region ID incorrectly! Please check GPDatabase");
    region.setType(dbregion->type());
    for (auto& rect : dbregion->rects) {
        box_type box(rect.lx, rect.ly, rect.hx, rect.hy);
        region.addBox(box);
    }
}

point_type GPDatabase::getAbsolutePinPos(const GPPin& pin) const {
    coord_type absX, absY;
    auto& parNode = this->nodes.at(pin.getParNodeId());
    absX = pin.getRelLx() + parNode.getLx();
    absY = pin.getRelLy() + parNode.getLy();
    return std::make_pair(absX, absY);
}

void GPDatabase::setupNum() {
    dieInfo = std::make_tuple(database.dieLX, database.dieHX, database.dieLY, database.dieHY);
    coreInfo = std::make_tuple(database.coreLX, database.coreHX, database.coreLY, database.coreHY);
    siteW = static_cast<int>(database.siteW);
    siteH = database.siteH;

    num_nodes = database.cells.size() + database.iopins.size() + database.placeBlockages.size();
    num_nets = database.nets.size();
    num_pins = 0;
    for (auto& dbnet : database.nets) {
        num_pins += dbnet->pins.size();
    }
    num_regions = database.regions.size();

    nodes.reserve(num_nodes);
    pins.reserve(num_pins);
    nets.reserve(num_nets);

    pin_id2node_id.reserve(num_pins);
    pin_id2net_id.reserve(num_pins);
}

void GPDatabase::setupNodes() {
    // preprocess nodes in database
    std::vector<index_type> all_mov_ids;
    all_mov_ids.reserve(database.cells.size());
    std::vector<index_type> all_fix_ids;
    all_fix_ids.reserve(database.cells.size());
    std::vector<index_type> all_iopin_ids;
    all_iopin_ids.reserve(database.iopins.size());
    std::vector<index_type> all_float_iopin_ids;
    all_float_iopin_ids.reserve(database.iopins.size());
    std::vector<index_type> all_float_fix_ids;
    all_float_fix_ids.reserve(database.cells.size());
    std::vector<index_type> all_float_mov_ids;
    all_float_mov_ids.reserve(database.cells.size());

    for (index_type i = 0; i < static_cast<index_type>(database.cells.size()); i++) {
        auto cell = database.cells[i];
        if (cell->is_connected) {
            if (!cell->fixed()) {
                all_mov_ids.emplace_back(i);
            } else {
                all_fix_ids.emplace_back(i);
            }
        } else {
            if (!cell->fixed()) {
                all_float_mov_ids.emplace_back(i);
            } else {
                all_float_fix_ids.emplace_back(i);
            }
        }
    }

    for (index_type i = 0; i < static_cast<index_type>(database.iopins.size()); i++) {
        auto iopin = database.iopins[i];
        if (iopin->is_connected) {
            all_iopin_ids.emplace_back(i);
        } else {
            all_float_iopin_ids.emplace_back(i);
        }
    }

    std::string cur_node_type = "Mov";
    index_type stard_idx = nodes.size();
    for (index_type cell_id : all_mov_ids) {
        addCellNode(cell_id, cur_node_type);
    }
    index_type end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));

    cur_node_type = "FloatMov";
    stard_idx = nodes.size();
    for (index_type cell_id : all_float_mov_ids) {
        addCellNode(cell_id, cur_node_type);
    }
    end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));

    cur_node_type = "Fix";
    stard_idx = nodes.size();
    for (index_type cell_id : all_fix_ids) {
        addCellNode(cell_id, cur_node_type);
    }
    end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));

    cur_node_type = "IOPin";
    stard_idx = nodes.size();
    for (index_type iopin_id : all_iopin_ids) {
        addIOPinNode(iopin_id, cur_node_type);
    }
    end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));

    cur_node_type = "Blkg";
    stard_idx = nodes.size();
    for (index_type blkg_id = 0; blkg_id < static_cast<index_type>(database.placeBlockages.size()); blkg_id++) {
        addBlockageNode(blkg_id, cur_node_type);
    }
    end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));

    cur_node_type = "FloatIOPin";
    stard_idx = nodes.size();
    for (index_type iopin_id : all_float_iopin_ids) {
        addIOPinNode(iopin_id, cur_node_type);
    }
    end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));

    cur_node_type = "FloatFix";
    stard_idx = nodes.size();
    for (index_type cell_id : all_float_fix_ids) {
        addCellNode(cell_id, cur_node_type);
    }
    end_idx = nodes.size();
    node_types_indices.emplace_back(std::make_tuple(stard_idx, end_idx, cur_node_type));
}

void GPDatabase::setupNets() {
    // setup nets and their pins
    for (index_type dbnet_id = 0; dbnet_id < static_cast<index_type>(database.nets.size()); dbnet_id++) {
        addNet(dbnet_id);
    }
}

void GPDatabase::setupRegions() {
    // setup regions
    // Make sure this step done before setupNodes
    for (index_type dbregion_id = 0; dbregion_id < static_cast<index_type>(database.regions.size()); dbregion_id++) {
        addRegion(dbregion_id);
    }
}

void GPDatabase::setupIndexMap() {
    for (auto& pin : pins) {
        const GPNode& node = nodes.at(pin.getParNodeId());
        const GPNet& net = nets.at(pin.getParNetId());
        pin_id2node_id.emplace_back(node.getId());
        pin_id2net_id.emplace_back(net.getId());
    }
    for (auto& node : nodes) {
        node_id2node_name.emplace_back(node.getName());
    }
}

void GPDatabase::setupCheckVar() {
    assert_msg(nodes.size() == num_nodes, "Nodes size is (%d), it should be (%d)", nodes.size(), num_nodes);
    assert_msg(nets.size() == num_nets, "Nets size is (%d), it should be (%d)", nets.size(), num_nets);
    assert_msg(pins.size() == num_pins, "Pins size is (%d), it should be (%d)", pins.size(), num_pins);
    logger.verbose("Finish checking");

    // check whether nodes and pins are placed out of boundary or not
    // TODO change log_debug to log_warn
    // utils::BoxT<coord_type> die_box(database.dieLX, database.dieLY, database.dieHX, database.dieHY);
    // for (auto& node : nodes) {
    //     utils::BoxT<coord_type> node_box(node.getLx(), node.getLy(), node.getHx(), node.getHy());
    //     auto union_box = node_box.UnionWith(die_box);
    //     if (union_box != die_box) {
    //         log(LOG_WARN) << "Node " << node.getName() << "'s (lx,ly,hx,hy) = (" << node.getLx() << ", " <<
    //         node.getLy()
    //                       << ", " << node.getHx() << ", " << node.getHy() << ")"
    //                       << " is placed outside die boundary" << std::endl;
    //     }
    // }
    // for (auto& pin : pins_abs_info) {
    //     int index = &pin - &pins_abs_info[0];
    //     utils::BoxT<coord_type> pin_box(pin[0], pin[1], pin[0] + pin[2], pin[1] + pin[3]);  // lx, ly, hx, hy
    //     auto union_box = pin_box.UnionWith(die_box);
    //     if (union_box != die_box) {
    //         log(LOG_WARN) << "Node(" << nodes[pins.at(index).getParNodeId()].getName() << ")'s pin on (lx,ly,hx,hy) =
    //         ("
    //                       << pin[0] << ", " << pin[1] << ", " << pin[0] + pin[2] << ", " << pin[1] + pin[3] << ")"
    //                       << " is outside die boundary" << std::endl;
    //     }
    // }
}

bool GPDatabase::setup() {
    if (db::setting.random_place) {
        setup_random_place();
        logger.info("random place db done");
    }
    setupNum();
    setupRegions();
    setupNodes();
    setupNets();
    setupIndexMap();
    setupCheckVar();
    logger.info("Finish initializing global placement database");
    return true;
}

bool GPDatabase::reset() {
    num_nodes = 0;
    num_pins = 0;
    num_nets = 0;

    nodes.clear();
    pins.clear();
    nets.clear();

    node_types_indices.clear();

    pin_id2node_id.clear();
    pin_id2net_id.clear();

    nodes_id2pins_id.clear();
    nets_id2pins_id.clear();

    return true;
}

void GPDatabase::setup_random_place() {
    int max_pin_width = std::numeric_limits<int>::min();
    int max_pin_height = std::numeric_limits<int>::min();
    int max_cell_width = std::numeric_limits<int>::min();
    int max_cell_height = std::numeric_limits<int>::min();
    for (auto cell : database.cells) {
        if (!cell->fixed()) {
            max_cell_width = std::max(max_cell_width, cell->ctype()->width);
            max_cell_height = std::max(max_cell_height, cell->ctype()->height);
            for (auto pintype : cell->ctype()->pins) {
                max_pin_width = std::max(max_pin_width, int(pintype->getW()));
                max_pin_height = std::max(max_pin_height, int(pintype->getH()));
            }
        }
    }
    int dieLXLegalPlace = database.dieLX + max_pin_width + max_cell_width + 1;
    int dieHXLegalPlace = database.dieHX - max_pin_width - max_cell_width - 1;
    int dieLYLegalPlace = database.dieLY + max_pin_height + max_cell_height + 1;
    int dieHYLegalPlace = database.dieHY - max_pin_height - max_cell_height - 1;

    // Non-deterministic
    // std::random_device rdx;
    // std::random_device rdy;
    // std::mt19937 genX(rdx());
    // std::mt19937 genY(rdy());

    std::mt19937 genX(123);
    std::mt19937 genY(567);
    std::uniform_int_distribution<> xDis(dieLXLegalPlace, dieHXLegalPlace);
    std::uniform_int_distribution<> yDis(dieLYLegalPlace, dieHYLegalPlace);

    for (auto cell : database.cells) {
        if (!cell->fixed()) {
            int x = xDis(genX);
            int y = yDis(genY);
            cell->place(x, y);
        }
    }
}

// Torch Related
torch::Tensor GPDatabase::getNodeLPosTensor() {
    torch::Tensor node_lpos = torch::zeros({num_nodes, 2});
    auto node_lpos_a = node_lpos.accessor<coord_type, 2>();
    for (auto& node : nodes) {
        node_lpos_a[node.getId()][0] = node.getLx();
        node_lpos_a[node.getId()][1] = node.getLy();
    }
    return node_lpos;
}

torch::Tensor GPDatabase::getNodeCPosTensor() {
    torch::Tensor node_cpos = torch::zeros({num_nodes, 2});
    auto node_cpos_a = node_cpos.accessor<coord_type, 2>();
    for (auto& node : nodes) {
        node_cpos_a[node.getId()][0] = node.getLx() + node.getWidth() / 2;
        node_cpos_a[node.getId()][1] = node.getLy() + node.getHeight() / 2;
    }
    return node_cpos;
}

torch::Tensor GPDatabase::getNodeSizeTensor() {
    torch::Tensor node_size = torch::zeros({num_nodes, 2});
    auto node_size_a = node_size.accessor<coord_type, 2>();
    for (auto& node : nodes) {
        if (!node.getIsPolygonShape()) {
            node_size_a[node.getId()][0] = node.getWidth();
            node_size_a[node.getId()][1] = node.getHeight();
        } else {
            // ICCAD/DAC 2012 contain fixed polygon-shape nodes. We consider them as placement
            // blockages instead of fixed nodes. So we set their width and height as zeros to
            // avoid duplicated density calculation.
        }
    }
    return node_size;
}

torch::Tensor GPDatabase::getPinRelLPosTensor() {
    // pin_pos == (node_pos - node_size / 2) + (pin_rel_lpos + pin_size / 2)
    torch::Tensor pin_rel_lpos = torch::zeros({num_pins, 2});
    auto pin_rel_lpos_a = pin_rel_lpos.accessor<coord_type, 2>();
    for (auto& pin : pins) {
        pin_rel_lpos_a[pin.getId()][0] = pin.getRelLx();
        pin_rel_lpos_a[pin.getId()][1] = pin.getRelLy();
    }
    return pin_rel_lpos;
}

torch::Tensor GPDatabase::getPinRelCPosTensor() {
    // pin_pos == node_pos + pin_rel_cpos
    torch::Tensor pin_rel_cpos = torch::zeros({num_pins, 2});
    auto pin_rel_cpos_a = pin_rel_cpos.accessor<coord_type, 2>();
    for (auto& pin : pins) {
        auto& node = nodes[pin.getParNodeId()];
        pin_rel_cpos_a[pin.getId()][0] = pin.getRelLx() + pin.getWidth() / 2 - node.getWidth() / 2;
        pin_rel_cpos_a[pin.getId()][1] = pin.getRelLy() + pin.getHeight() / 2 - node.getHeight() / 2;
    }
    return pin_rel_cpos;
}

torch::Tensor GPDatabase::getPinSizeTensor() {
    torch::Tensor pin_size = torch::zeros({num_pins, 2});
    auto pin_size_a = pin_size.accessor<coord_type, 2>();
    for (auto& pin : pins) {
        pin_size_a[pin.getId()][0] = pin.getWidth();
        pin_size_a[pin.getId()][1] = pin.getHeight();
    }
    return pin_size;
}

torch::Tensor GPDatabase::getPinId2NodeIdTensor() {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    return torch::from_blob(pin_id2node_id.data(), {num_pins}, options).clone();
}

torch::Tensor GPDatabase::getPinId2NetIdTensor() {
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    return torch::from_blob(pin_id2net_id.data(), {num_pins}, options).clone();
}

std::vector<torch::Tensor> GPDatabase::getHyperedgeInfoTensor() {
    auto options = torch::TensorOptions().dtype(torch::kInt64);

    torch::Tensor hyperedge_index_helper = torch::zeros({num_pins}, options);
    torch::Tensor hyperedge_list = torch::zeros({num_pins}, options);
    torch::Tensor hyperedge_list_end = torch::zeros({num_nets}, options);

    auto hyperedge_index_helper_a = hyperedge_index_helper.accessor<index_type, 1>();
    auto hyperedge_list_a = hyperedge_list.accessor<index_type, 1>();
    auto hyperedge_list_end_a = hyperedge_list_end.accessor<index_type, 1>();

    index_type ptr = 0;
    index_type last_idx = 0;
    for (auto& net : nets) {
        index_type net_id = net.getId();
        for (auto pin_id : net.pins()) {
            hyperedge_list_a[ptr] = pin_id;
            hyperedge_index_helper_a[ptr] = net_id;
            ptr += 1;
        }
        last_idx += net.pins().size();
        hyperedge_list_end_a[net_id] = last_idx;
    }
    auto hyperedge_index = torch::cat({hyperedge_list.unsqueeze(0), hyperedge_index_helper.unsqueeze(0)}, 0);

    auto new_order_idx = torch::argsort(hyperedge_index.index({0}), 0, false);
    hyperedge_index = hyperedge_index.index({torch::indexing::Slice(), new_order_idx});

    return {hyperedge_index, hyperedge_list, hyperedge_list_end};
}

std::vector<torch::Tensor> GPDatabase::getNode2PinInfoTensor() {
    auto options = torch::TensorOptions().dtype(torch::kInt64);

    torch::Tensor node2pin_index_helper = torch::zeros({num_pins}, options);
    torch::Tensor node2pin_list = torch::zeros({num_pins}, options);
    torch::Tensor node2pin_list_end = torch::zeros({num_nodes}, options);

    auto node2pin_index_helper_a = node2pin_index_helper.accessor<index_type, 1>();
    auto node2pin_list_a = node2pin_list.accessor<index_type, 1>();
    auto node2pin_list_end_a = node2pin_list_end.accessor<index_type, 1>();

    index_type ptr = 0;
    index_type last_idx = 0;
    for (auto& node : nodes) {
        index_type node_id = node.getId();
        for (auto pin_id : node.pins()) {
            node2pin_list_a[ptr] = pin_id;
            node2pin_index_helper_a[ptr] = node_id;
            ptr += 1;
        }
        last_idx += node.pins().size();
        node2pin_list_end_a[node_id] = last_idx;
    }
    auto node2pin_index = torch::cat({node2pin_list.unsqueeze(0), node2pin_index_helper.unsqueeze(0)}, 0);

    auto new_order_idx = torch::argsort(node2pin_index.index({0}), 0, false);
    node2pin_index = node2pin_index.index({torch::indexing::Slice(), new_order_idx});

    return {node2pin_index, node2pin_list, node2pin_list_end};
}

std::vector<torch::Tensor> GPDatabase::getRegionInfoTensor() {
    auto options_int = torch::TensorOptions().dtype(torch::kInt64);

    unsigned num_boxes = 0;
    for (auto& region : regions) {
        num_boxes += region.boxes().size();
    }

    torch::Tensor node_id2region_id = torch::zeros({num_nodes}, options_int);
    torch::Tensor region_boxes = torch::zeros({num_boxes, 4});
    torch::Tensor region_boxes_end = torch::zeros({num_regions}, options_int);

    auto node_id2region_id_a = node_id2region_id.accessor<index_type, 1>();
    auto region_boxes_a = region_boxes.accessor<coord_type, 2>();
    auto region_boxes_end_a = region_boxes_end.accessor<index_type, 1>();

    for (auto& node : nodes) {
        index_type node_id = node.getId();
        index_type region_id = node.getRegionId();
        node_id2region_id_a[node_id] = region_id;
    }

    index_type ptr = 0;
    index_type last_idx = 0;
    for (auto& region : regions) {
        index_type region_id = region.getId();
        for (auto& box : region.boxes()) {
            region_boxes_a[ptr][0] = box.lx();
            region_boxes_a[ptr][1] = box.hx();
            region_boxes_a[ptr][2] = box.ly();
            region_boxes_a[ptr][3] = box.hy();
            ptr += 1;
        }
        last_idx += region.boxes().size();
        region_boxes_end_a[region_id] = last_idx;
    }

    return {node_id2region_id, region_boxes, region_boxes_end};
}

std::vector<torch::Tensor> GPDatabase::getSnetInfoTensor() {
    auto options_int = torch::TensorOptions().dtype(torch::kInt64);

    unsigned num_snetshapes = 0;
    for (size_t snetId = 0; snetId < database.snets.size(); snetId++) {
        db::SNet* snet = database.snets[snetId];
        for (size_t shapeIdx = 0; shapeIdx < snet->shapes.size(); shapeIdx++) {
            num_snetshapes++;
        }
    }

    torch::Tensor snet_lpos = torch::zeros({num_snetshapes, 2});
    torch::Tensor snet_size = torch::zeros({num_snetshapes, 2});
    torch::Tensor snet_layer = torch::zeros({num_snetshapes}, options_int);

    auto snet_lpos_a = snet_lpos.accessor<coord_type, 2>();
    auto snet_size_a = snet_size.accessor<coord_type, 2>();
    auto snet_layer_a = snet_layer.accessor<index_type, 1>();

    int ptr = 0;
    for (size_t snetId = 0; snetId < database.snets.size(); snetId++) {
        db::SNet* snet = database.snets[snetId];
        for (size_t shapeIdx = 0; shapeIdx < snet->shapes.size(); shapeIdx++) {
            auto& shape = snet->shapes[shapeIdx];
            snet_lpos_a[ptr][0] = shape.lx;
            snet_lpos_a[ptr][1] = shape.ly;
            snet_size_a[ptr][0] = shape.hx - shape.lx;
            snet_size_a[ptr][1] = shape.hy - shape.ly;
            snet_layer_a[ptr] = shape.layer.rIndex;
            ptr++;
        }
    }

    return {snet_lpos, snet_size, snet_layer};
}

void GPDatabase::applyOneNodeOrient(int node_id) {
    auto& node = nodes[node_id];
    int rowId;
    int numRows = database.rows.size();
    if (node.getLy() <= database.coreLY) {
        rowId = 0;
    } else if (node.getLy() >= database.coreHY) {
        rowId = numRows - 1;
    } else {
        rowId = std::lround((node.getLy() - database.coreLY) / (float)siteH);
        rowId = std::max(std::min(rowId, numRows - 1), 0);
    }
    auto row = database.rows[rowId];
    if (row->flip()) {
        node.setOrient(6);  // FS
    } else {
        node.setOrient(0);  // N
    }
}

void GPDatabase::applyNodeCPos(torch::Tensor node_cpos) {
    const float* node_cpos_ptr = node_cpos.data_ptr<float>();
    for (auto& node : nodes) {
        if (node.getNodeType() != "Mov" && node.getNodeType() != "FloatMov") {
            continue;
        }
        node.setLx(node_cpos_ptr[node.getId() * 2 + 0] - node.getWidth() / 2);
        node.setLy(node_cpos_ptr[node.getId() * 2 + 1] - node.getHeight() / 2);
        applyOneNodeOrient(node.getId());
        auto cell = database.cells[node.getOriDBId()];
        cell->place(std::lround(node.getLx()), std::lround(node.getLy()), node.getOrient());
    }
}

void GPDatabase::applyNodeLPos(torch::Tensor node_lpos) {
    const float* node_lpos_ptr = node_lpos.data_ptr<float>();
    for (auto& node : nodes) {
        if (node.getNodeType() != "Mov" && node.getNodeType() != "FloatMov") {
            continue;
        }
        node.setLx(node_lpos_ptr[node.getId() * 2 + 0]);
        node.setLy(node_lpos_ptr[node.getId() * 2 + 1]);
        applyOneNodeOrient(node.getId());
        auto cell = database.cells[node.getOriDBId()];
        cell->place(std::lround(node.getLx()), std::lround(node.getLy()), node.getOrient());
    }
}

void GPDatabase::writePlacement(const std::string& given_prefix) { database.save(given_prefix); }

}  // namespace gp
