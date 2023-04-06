import torch
from cpp_to_py import io_parser
import os

class IOParser(object):
    def __init__(self) -> None:
        super().__init__()
        self.params = None

    def reset_params(self):
        io_parser.load_params({})

    def load_params(
        self,
        params: dict,
        verbose_log: bool = False,
        lite_mode: bool = False,
        random_place: bool = True,
        num_threads: int = 8,
    ):
        check_status = self.check_params(params, verbose_log, lite_mode, random_place, num_threads)
        if not check_status:
            raise ValueError(
                "Checking failure. Please check the validity of params: %s" % params
            )
        load_status = io_parser.load_params(self.params)
        if not load_status:
            raise ValueError(
                "Loading failure. Please check the validity of params: %s" % params
            )
        return load_status

    def check_params(
        self, params: dict, verbose_log: bool, lite_mode: bool, random_place: bool, num_threads: int = 8
    ) -> bool:
        if "def" not in params.keys() and "aux" not in params.keys():
            print("def or aux is not found!")
            return False
        if "def" in params.keys() and "aux" in params.keys():
            print("Can only support one input format!")
            return False
        if "def" in params.keys():
            if "lef" not in params.keys():
                if "cell_lef" not in params.keys() and "tech_lef" not in params.keys():
                    print("lef or (cell_lef and tech_lef) is not found")
                    return False
                if not os.path.exists(params["cell_lef"]):
                    print("cell_lef %s not exists." % params["cell_lef"])
                    return False
                if not os.path.exists(params["tech_lef"]):
                    print("tech_lef %s not exists." % params["tech_lef"])
                    return False
            else:
                if not os.path.exists(params["lef"]):
                    print("lef %s not exists." % params["lef"])
                    return False
            if "output" in params.keys():
                if "def" != params["output"].split(".")[-1]:
                    print("output format should be .def")
                    return False
            if not os.path.exists(params["def"]):
                print("def %s not exists." % params["def"])
                return False
        if "aux" in params.keys():
            if "pl" not in params.keys():
                print("pl is not found!")
                return False
            if "output" in params.keys():
                if "pl" != params["output"].split(".")[-1]:
                    print("output format should be .pl")
                    return False
            if not os.path.exists(params["aux"]):
                print("aux %s not exists." % params["aux"])
                return False
            if not os.path.exists(params["pl"]):
                print("pl %s not exists." % params["pl"])
                return False
        self.params = params

        if verbose_log:
            self.params["verbose_parser_log"] = True
        else:
            self.params["verbose_parser_log"] = False

        if lite_mode:
            self.params["lite_mode"] = True
        else:
            self.params["lite_mode"] = False

        if random_place:
            self.params["random_place"] = True
        else:
            self.params["random_place"] = False
        
        self.params["num_threads"] = num_threads

        return True

    def read(
        self,
        params: dict,
        verbose_log: bool = False,
        lite_mode: bool = False,
        random_place: bool = True,
        num_threads: int = 8,
        debug: bool = False,
    ):
        check_status = self.check_params(params, verbose_log, lite_mode, random_place, num_threads)
        if not check_status:
            raise ValueError(
                "Checking failure. Please check the validity of params: %s" % params
            )
        if debug:
            load_status = io_parser.load_params(self.params)
            if not load_status:
                raise ValueError(
                    "Loading failure. Please check the validity of params: %s" % params
                )
            rawdb = io_parser.create_database()
            rawdb.load()
            rawdb.setup()
            gpdb = io_parser.create_gpdatabase(rawdb)
            gpdb.setup()
        else:
            rawdb, gpdb = io_parser.start(self.params)
        # rawdb and gpdb are both c++ shared pointer
        return (rawdb, gpdb)

    def preprocess_design_info(self, gpdb):
        dieLX, dieHX, dieLY, dieHY = gpdb.coreInfo() # use coreInfo instead of dieInfo

        die_info = torch.tensor([dieLX, dieHX, dieLY, dieHY]).float()
        # die_shift = torch.tensor([dieLX, dieLY])
        # die_scale = torch.tensor([dieHX - dieLX, dieHY - dieLY])

        siteWidth = gpdb.siteWidth()
        siteHeight = gpdb.siteHeight()
        site_info = (float(siteWidth), float(siteHeight))

        node_pos = gpdb.node_cpos_tensor()
        node_lpos = gpdb.node_lpos_tensor()
        node_size = gpdb.node_size_tensor()
        pin_rel_cpos = gpdb.pin_rel_cpos_tensor()
        pin_rel_lpos = gpdb.pin_rel_lpos_tensor()
        pin_size = gpdb.pin_size_tensor()

        pin_id2node_id = gpdb.pin_id2node_id_tensor()
        (
            hyperedge_index,
            hyperedge_list,
            hyperedge_list_end,
        ) = gpdb.hyperedge_info_tensor()

        (
            node2pin_index,
            node2pin_list,
            node2pin_list_end,
        ) = gpdb.node2pin_info_tensor()

        node_id2region_id, region_boxes, region_boxes_end = gpdb.region_info_tensor()

        node_type_indices = gpdb.node_type_indices()
        node_id2node_name = gpdb.node_id2node_name()
        all_node_types = []
        mov_end_idx = None
        fix_end_idx = None
        connected_end_idx = None
        for start_idx, end_idx, type_name in node_type_indices:
            all_node_types.append(type_name)
            if "FloatMov" == type_name:
                mov_end_idx = end_idx
            if "FloatFix" == type_name:
                fix_end_idx = end_idx
            if "IOPin" == type_name:
                connected_end_idx = end_idx
        # Mov + FloatMov
        movable_index = (0, mov_end_idx)
        # Mov + FloatMov + Fix + IOPin
        connected_index = (0, connected_end_idx)
        # Fix + IOPin + Blkg + FloatIOPin + FloatFix
        fixed_index = (mov_end_idx, fix_end_idx)

        design_info = {
            "benchmark": self.params["benchmark"],
            "dataset_path": self.params,
            "node_type_indices": node_type_indices,
            "node_id2node_name": node_id2node_name,
            "movable_index": movable_index,
            "connected_index": connected_index,
            "fixed_index": fixed_index,
            "site_info": site_info,
            "die_info": die_info,
            "node_pos": node_pos.contiguous(),
            "node_lpos": node_lpos.contiguous(),
            "node_size": node_size.contiguous(),
            "pin_rel_cpos": pin_rel_cpos.contiguous(),
            "pin_rel_lpos": pin_rel_lpos.contiguous(),
            "pin_size": pin_size.contiguous(),
            "pin_id2node_id": pin_id2node_id.long().contiguous(),
            "hyperedge_index": hyperedge_index.long().contiguous(),
            "hyperedge_list": hyperedge_list.long().contiguous(),
            "hyperedge_list_end": hyperedge_list_end.long().contiguous(),
            "node2pin_index": node2pin_index.long().contiguous(),
            "node2pin_list": node2pin_list.long().contiguous(),
            "node2pin_list_end": node2pin_list_end.long().contiguous(),
            "node_id2region_id": node_id2region_id.long().contiguous(),
            "region_boxes": region_boxes.contiguous(),
            "region_boxes_end": region_boxes_end.long().contiguous(),
        }
        return design_info
