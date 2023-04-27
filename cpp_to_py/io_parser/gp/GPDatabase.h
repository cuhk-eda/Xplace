#pragma once
#include "common/common.h"
#include "common/db/Database.h"

namespace gp {

using index_type = int64_t;
using coord_type = float;
using orient_type = int;  // 0:N, 1:W, 2:S, 3:E, 4:FN, 5:FW, 6:FS, 7:FE, -1:NONE
using point_type = std::pair<coord_type, coord_type>;
using box_type = utils::BoxT<coord_type>;

class Basic {
public:
    index_type getId() const { return id; }
    void setId(const index_type& i) { id = i; }
    std::string getName() const { return name; }
    void setName(const std::string& name_str) { name = name_str; }

protected:
    index_type id = std::numeric_limits<index_type>::max();
    std::string name = "";
};

class GPNode : public Basic {
public:
    void setLx(const coord_type& lx_) { lx = lx_; }
    const coord_type& getLx() const { return lx; }
    void setLy(const coord_type& ly_) { ly = ly_; }
    const coord_type& getLy() const { return ly; }

    coord_type getHx() const { return getLx() + getWidth(); }
    coord_type getHy() const { return getLy() + getHeight(); }

    void setWidth(const coord_type& width_) { width = width_; }
    const coord_type& getWidth() const { return width; }
    void setHeight(const coord_type& height_) { height = height_; }
    const coord_type& getHeight() const { return height; }
    void setOrient(const orient_type& orient_) { orient = orient_; }
    const orient_type& getOrient() const { return orient; }
    void setNodeType(const std::string& node_type_) { node_type = node_type_; }
    const std::string& getNodeType() const { return node_type; }
    void setOriDBId(const index_type& ori_db_id_) { ori_db_id = ori_db_id_; }
    const index_type& getOriDBId() const { return ori_db_id; }
    void setRegionId(const index_type& region_id_) { region_id = region_id_; }
    const index_type& getRegionId() const { return region_id; }

    const std::vector<index_type>& pins() const { return pins_id; }
    void addPin(index_type pin_id) { pins_id.emplace_back(pin_id); }

    void setIsPolygonShape(bool isPolygonShape_) { isPolygonShape = isPolygonShape_; }
    const bool getIsPolygonShape() const { return isPolygonShape; }

protected:
    coord_type lx = std::numeric_limits<coord_type>::max();
    coord_type ly = std::numeric_limits<coord_type>::max();
    coord_type width = 0;
    coord_type height = 0;
    orient_type orient = -1;
    std::string node_type = "";  // Mov, FloatMov, Fix, IOPin, Blkg, FloatIOPin, FloatFix
    index_type ori_db_id = std::numeric_limits<index_type>::max();
    index_type region_id = -1;  // no region, we should ignore fixed nodes' fence region
    std::vector<index_type> pins_id;
    bool isPolygonShape = false;
};

class GPPin : public Basic {
public:
    void setRelLx(const coord_type& rel_lx_) { rel_lx = rel_lx_; }
    const coord_type& getRelLx() const { return rel_lx; }
    void setRelLy(const coord_type& rel_ly_) { rel_ly = rel_ly_; }
    const coord_type& getRelLy() const { return rel_ly; }
    void setWidth(const coord_type& width_) { width = width_; }
    const coord_type& getWidth() const { return width; }
    void setHeight(const coord_type& height_) { height = height_; }
    const coord_type& getHeight() const { return height; }

    coord_type getRelHx() const { return getRelLx() + getWidth(); }
    coord_type getRelHy() const { return getRelLy() + getHeight(); }

    void setDirection(const char& direction_) { direction = direction_; }
    const char& getDirection() const { return direction; }
    void setType(const char& type_) { type = type_; }
    const char& getType() const { return type; }
    void setParNodeId(const index_type& parent_node_id_) { parent_node_id = parent_node_id_; }
    const index_type& getParNodeId() const { return parent_node_id; }
    void setParNetId(const index_type& parent_net_id_) { parent_net_id = parent_net_id_; }
    const index_type& getParNetId() const { return parent_net_id; }
    void setOriDBInfo(const std::tuple<index_type, index_type, index_type>& ori_db_info_) {
        ori_db_info = ori_db_info_;
    }
    const std::tuple<index_type, index_type, index_type>& getOriDBInfo() const { return ori_db_info; }

protected:
    coord_type rel_lx = std::numeric_limits<coord_type>::max();  // relative position from node_lx to pin_lx
    coord_type rel_ly = std::numeric_limits<coord_type>::max();  // relative position from node_ly to pin_ly
    coord_type width = 0;
    coord_type height = 0;
    // i: input, o:output
    char direction = 'x';
    // s: signal, c: clk, p: power, g: ground
    char type = 's';
    index_type parent_node_id = std::numeric_limits<index_type>::max();
    index_type parent_net_id = std::numeric_limits<index_type>::max();
    //    iopin: ori_db_info == {ori_db_parent_iopin_id, -1, ori_db_parent_net_id}
    // cell pin: ori_db_info == {ori_db_parent_cell_id, ori_db_parent_cell_pin_id, ori_db_parent_net_id}
    std::tuple<index_type, index_type, index_type> ori_db_info = {-1, -1, -1};
};

class GPNet : public Basic {
public:
    void setOriDBId(const index_type& ori_db_id_) { ori_db_id = ori_db_id_; }
    const index_type& getOriDBId() const { return ori_db_id; }
    const std::vector<index_type>& pins() const { return pins_id; }
    void addPin(index_type pin_id) { pins_id.emplace_back(pin_id); }

protected:
    index_type ori_db_id = std::numeric_limits<index_type>::max();
    std::vector<index_type> pins_id;
};

// TODO: fence region and its mapping to nodes
class GPRegion : public Basic {
public:
    void setType(const char& type_) { type = type_; }
    const char& getType() const { return type; }
    void setOriDBId(const index_type& ori_db_id_) { ori_db_id = ori_db_id_; }
    const index_type& getOriDBId() const { return ori_db_id; }

    const std::vector<index_type>& nodes() const { return nodes_id; }
    void addNode(index_type node_id) { nodes_id.emplace_back(node_id); }
    const std::vector<box_type>& boxes() const { return _boxes; }
    void addBox(box_type box) { _boxes.emplace_back(box); }

protected:
    // f: fence, g: guide
    char type = 'f';
    index_type ori_db_id = std::numeric_limits<index_type>::max();
    std::vector<index_type> nodes_id;
    std::vector<box_type> _boxes;
};

class GPDatabase {
protected:
    db::Database& database;

    std::tuple<int, int, int, int> dieInfo;   // dieLX, dieHX, dieLY, dieHY
    std::tuple<int, int, int, int> coreInfo;  // coreLX, coreHX, coreLY, coreHY
    int siteW;
    int siteH;

    unsigned int num_nodes;
    unsigned int num_pins;
    unsigned int num_nets;
    unsigned int num_regions;

    std::vector<GPNode> nodes;      // store all nodes Mov + FloatMov + Fix + IOPin + Blkg + FloatIOPin + FloatFix
    std::vector<GPPin> pins;        // store all pins
    std::vector<GPNet> nets;        // store all nets
    std::vector<GPRegion> regions;  // store all regions

    std::vector<std::tuple<index_type, index_type, std::string>> node_types_indices;  // (start_idx, end_idx, type)
    std::vector<std::string> node_id2node_name;

    std::vector<index_type> pin_id2node_id;  // pin_id to node_id mapping
    std::vector<index_type> pin_id2net_id;   // pin_id to net_id mapping

    std::vector<std::vector<index_type>> nodes_id2pins_id;
    std::vector<std::vector<index_type>> nets_id2pins_id;

public:
    GPDatabase(std::shared_ptr<db::Database> database_) : database(*database_) {}
    ~GPDatabase();
    point_type getAbsolutePinPos(const GPPin& pin) const;

    void addCellNode(index_type cell_id, std::string& node_type);
    void addIOPinNode(index_type iopin_id, std::string& node_type);
    void addBlockageNode(index_type blkg_id, std::string& node_type);
    void addNet(index_type dbnet_id);
    void addPin(db::Pin* dbpin, const db::PinType* pintype, GPNode& node, GPNet& net, bool isIOPin);
    void addRegion(index_type dbregion_id);

    void setupNum();
    void setupRegions();
    void setupNodes();
    void setupNets();
    void setupIndexMap();
    void setupCheckVar();
    bool setup();
    bool reset();

    void setup_random_place();

    const std::vector<GPNode>& getNodes() const { return nodes; }
    const std::vector<GPNet>& getNets() const { return nets; }
    const std::vector<GPPin>& getPins() const { return pins; }
    const std::vector<std::tuple<index_type, index_type, std::string>>& getNodeTypeIndices() const {
        return node_types_indices;
    }
    const std::vector<std::string>& getNodeId2NodeName() const { return node_id2node_name; }

    const std::tuple<int, int, int, int>& getDieInfo() const { return dieInfo; }
    const std::tuple<int, int, int, int>& getCoreInfo() const { return coreInfo; }
    const int getSiteWidth() const { return siteW; }
    const int getSiteHeight() const { return siteH; }
    const int getM1Direction() const { return database.getRLayer(0)->direction == 'v' ? 1 : 0; }

    // Torch related
    torch::Tensor getNodeLPosTensor();
    torch::Tensor getNodeCPosTensor();
    torch::Tensor getNodeSizeTensor();
    torch::Tensor getPinRelLPosTensor();  // pin_lx - node_lx
    torch::Tensor getPinRelCPosTensor();  // pin_cx - node_cx
    torch::Tensor getPinSizeTensor();
    torch::Tensor getPinId2NodeIdTensor();
    torch::Tensor getPinId2NetIdTensor();
    std::vector<torch::Tensor> getHyperedgeInfoTensor();  // hyperedge_index, hyperedge_list, hyperedge_list_end
    std::vector<torch::Tensor> getNode2PinInfoTensor();   // node2pin_index, node2pin_list, node2pin_list_end
    std::vector<torch::Tensor> getRegionInfoTensor();     // node_id2region_id, region_boxes, region_boxes_end
    std::vector<torch::Tensor> getSnetInfoTensor();       // snet_lpos, snet_size, snet_layer (0 for M1, 1 for M2, ...)
    void applyOneNodeOrient(int node_id);
    void applyNodeCPos(torch::Tensor node_cpos);
    void applyNodeLPos(torch::Tensor node_lpos);

    // Write Placement
    void writePlacement(const std::string& given_prefix = "");
};

}  // namespace gp