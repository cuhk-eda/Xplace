import torch
import collections
import copy
import math
from utils import *


def load_dataset(args, logger, placement=None):
    rawdb, gpdb = None, None
    if args.custom_path != "":
        params = get_custom_design_params(args)
    else:
        params = get_single_design_params(
            args.dataset_root, args.dataset, args.design_name, placement
        )
    parser = IOParser()
    if args.load_from_raw:
        logger.info("loading from original benchmark...")
        rawdb, gpdb = parser.read(
            params, verbose_log=False, lite_mode=True, random_place=False, num_threads=args.num_threads
        )
        design_info = parser.preprocess_design_info(gpdb)
    else:
        logger.info("loading from pt benchmark...")
        design_pt_path = "./data/cad/%s/%s.pt" % (args.dataset, args.design_name)
        parser.load_params(
            params, verbose_log=False, lite_mode=True, random_place=False, num_threads=args.num_threads
        )
        design_info = torch.load(design_pt_path)
        gpdb = None
    data = PlaceData(args, logger, **design_info)
    return data, rawdb, gpdb


def size_repr(key, item, indent=0):
    indent_str = " " * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f"{indent_str}{key}={out}"


class PlaceData(object):
    def __init__(
        self,
        args,
        logger,
        node_pos=None,
        node_lpos=None,
        node_size=None,
        pin_rel_cpos=None,
        pin_rel_lpos=None,
        pin_size=None,
        pin_id2node_id=None,
        hyperedge_index=None,
        hyperedge_list=None,
        hyperedge_list_end=None,
        node2pin_index=None,
        node2pin_list=None,
        node2pin_list_end=None,
        node_id2region_id=None,
        region_boxes=None,
        region_boxes_end=None,
        dataset_path=None,
        benchmark=None,
        die_info=None,
        site_info=None,
        node_type_indices=None,
        node_id2node_name=None,
        movable_index=None,
        connected_index=None,
        fixed_index=None,
        **kwargs,
    ):
        self.die_info = die_info  # lx, hx, ly, hy
        self.die_ur = None
        self.die_ll = None

        self.node_pos = node_pos
        self.node_lpos = node_lpos
        self.node_size = node_size

        self.pin_rel_cpos = pin_rel_cpos
        self.pin_rel_lpos = pin_rel_lpos
        self.pin_size = pin_size

        self.pin_id2node_id = pin_id2node_id

        self.hyperedge_index = hyperedge_index
        self.hyperedge_list = hyperedge_list
        self.hyperedge_list_end = hyperedge_list_end
        self.pin_id2net_id = hyperedge_index[1]

        self.node2pin_index = node2pin_index
        self.node2pin_list = node2pin_list
        self.node2pin_list_end = node2pin_list_end

        self.node_id2region_id = node_id2region_id
        self.region_boxes = region_boxes
        self.region_boxes_end = region_boxes_end

        dataset_format = ""
        if "aux" in dataset_path.keys():
            dataset_format = "bookshelf"
        elif "def" in dataset_path.keys():
            dataset_format = "lefdef"
        self.__dataset_format__ = dataset_format
        self.__dataset_path__ = dataset_path
        self.__design_name__ = benchmark + "/" + dataset_path["design_name"]

        self.__node_id2node_name__ = node_id2node_name

        # NOTE: we set float movable node as connected node for convenience purposes
        self.__node_type_indices__ = node_type_indices
        self.__movable_index__ = movable_index
        self.__movable_connected_index__ = (
            movable_index[0],
            self.node_type_indices[0][1],
        )
        self.__connected_index__ = connected_index
        self.__fixed_index__ = fixed_index
        self.__fixed_connected_index__ = (self.fixed_index[0], self.connected_index[1])
        self.__fixed_unconnected_index__ = (
            self.connected_index[1],
            self.fixed_index[1],
        )

        self.__site_width__ = site_info[0]
        self.__site_height__ = site_info[1]
        self.__row_height__ = site_info[1]  # the same as site height

        self.__ori_die_lx__ = die_info[0].item()
        self.__ori_die_hx__ = die_info[1].item()
        self.__ori_die_ly__ = die_info[2].item()
        self.__ori_die_hy__ = die_info[3].item()

        self.__num_nodes__ = node_pos.shape[0]
        self.__num_pins__ = pin_id2node_id.shape[0]
        self.__num_nets__ = hyperedge_list_end.shape[0]

        self.__num_bin_x__ = args.num_bin_x
        self.__num_bin_y__ = args.num_bin_y

        self.__clamp_node__ = args.clamp_node

        # fence region
        self.__num_regions__ = 1
        self.__enable_fence__ = False

        # Extra variable to handle corner cases
        self.fix_node_in_bd_mask = None
        self.dummy_macro_pos = None
        self.dummy_macro_size = None
        self.mov_node_size_real = None

        # filler
        self.filler_size = None

        self.__logger__ = logger
        self.__args__ = args

        for key, item in kwargs.items():
            self[key] = item

    @property
    def dataset_format(self):
        if hasattr(self, "__dataset_format__"):
            return self.__dataset_format__

    @property
    def dataset_path(self):
        if hasattr(self, "__dataset_path__"):
            return self.__dataset_path__

    @property
    def design_name(self):
        if hasattr(self, "__design_name__"):
            return self.__design_name__

    @property
    def node_id2node_name(self):
        if hasattr(self, "__node_id2node_name__"):
            return self.__node_id2node_name__

    @property
    def node_type_indices(self):
        if hasattr(self, "__node_type_indices__"):
            return self.__node_type_indices__

    @property
    def movable_index(self):
        if hasattr(self, "__movable_index__"):
            return self.__movable_index__

    @property
    def movable_connected_index(self):
        if hasattr(self, "__movable_connected_index__"):
            return self.__movable_connected_index__

    @property
    def connected_index(self):
        if hasattr(self, "__connected_index__"):
            return self.__connected_index__

    @property
    def fixed_index(self):
        if hasattr(self, "__fixed_index__"):
            return self.__fixed_index__

    @property
    def fixed_connected_index(self):
        if hasattr(self, "__fixed_connected_index__"):
            return self.__fixed_connected_index__

    @property
    def fixed_unconnected_index(self):
        if hasattr(self, "__fixed_unconnected_index__"):
            return self.__fixed_unconnected_index__

    @property
    def site_width(self):
        if hasattr(self, "__site_width__"):
            return self.__site_width__

    @property
    def site_height(self):
        if hasattr(self, "__site_height__"):
            return self.__site_height__

    @property
    def row_height(self):
        if hasattr(self, "__row_height__"):
            return self.__row_height__

    @property
    def ori_die_lx(self):
        if hasattr(self, "__ori_die_lx__"):
            return self.__ori_die_lx__

    @property
    def ori_die_hx(self):
        if hasattr(self, "__ori_die_hx__"):
            return self.__ori_die_hx__

    @property
    def ori_die_ly(self):
        if hasattr(self, "__ori_die_ly__"):
            return self.__ori_die_ly__

    @property
    def ori_die_hy(self):
        if hasattr(self, "__ori_die_hy__"):
            return self.__ori_die_hy__

    @property
    def die_shift(self):
        if hasattr(self, "__die_shift__"):
            return self.__die_shift__

    @property
    def die_scale(self):
        if hasattr(self, "__die_scale__"):
            return self.__die_scale__

    @property
    def num_nodes(self):
        if hasattr(self, "__num_nodes__"):
            return self.__num_nodes__

    @property
    def num_pins(self):
        if hasattr(self, "__num_pins__"):
            return self.__num_pins__

    @property
    def num_nets(self):
        if hasattr(self, "__num_nets__"):
            return self.__num_nets__

    @property
    def num_fillers(self):
        if hasattr(self, "__num_fillers__"):
            return self.__num_fillers__

    @property
    def num_bin_x(self):
        if hasattr(self, "__num_bin_x__"):
            return self.__num_bin_x__

    @property
    def num_bin_y(self):
        if hasattr(self, "__num_bin_y__"):
            return self.__num_bin_y__

    @property
    def clamp_node(self):
        if hasattr(self, "__clamp_node__"):
            return self.__clamp_node__

    @property
    def enable_fence(self):
        if hasattr(self, "__enable_fence__"):
            return self.__enable_fence__

    @property
    def num_regions(self):
        if hasattr(self, "__num_regions__"):
            return self.__num_regions__

    @property
    def total_mov_area_without_filler(self):
        if hasattr(self, "__total_mov_area_without_filler__"):
            return self.__total_mov_area_without_filler__

    @property
    def bin_area(self):
        if hasattr(self, "__bin_area__"):
            return self.__bin_area__

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        return data

    def to_dict(self):
        return {key: item for key, item in self}

    def to_namedtuple(self):
        keys = self.keys
        DataTuple = collections.namedtuple("DataTuple", keys)
        return DataTuple(*[self[key] for key in keys])

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def __delitem__(self, key):
        r"""Delete the data of the attribute :obj:`key`."""
        return delattr(self, key)

    @property
    def keys(self):
        r"""Returns all names of graph attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def cpu(self, *keys):
        r"""Copies all attributes :obj:`*keys` to CPU memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.cpu(), *keys)

    def cuda(self, device=None, non_blocking=False, *keys):
        r"""Copies all attributes :obj:`*keys` to CUDA memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(
            lambda x: x.cuda(device=device, non_blocking=non_blocking), *keys
        )

    def clone(self):
        return self.__class__.from_dict(
            {
                k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.__dict__.items()
            }
        )

    def pin_memory(self, *keys):
        r"""Copies all attributes :obj:`*keys` to pinned memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.pin_memory(), *keys)

    def record_stream(self, stream: torch.cuda.Stream, *keys):
        r"""Ensures that the tensor memory is not reused for another tensor
        until all current work queued on :obj:`stream` has been completed.
        If :obj:`*keys` is not given, this will be ensured for all present
        attributes."""

        def _record_stream(x):
            x.record_stream(stream)
            return x

        return self.apply(_record_stream, *keys)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({}, {})".format(cls, self.design_name, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}({}, \n{}\n)".format(cls, self.design_name, ",\n".join(info))

    def backup_ori_var(self):
        # backup original position and size
        self.__ori_die_info__ = self.die_info.clone().cpu().numpy()
        self.__ori_node_pos__ = self.node_pos.clone().cpu().numpy()
        self.__ori_node_lpos__ = self.node_lpos.clone().cpu().numpy()
        self.__ori_node_size__ = self.node_size.clone().cpu().numpy()
        self.__ori_pin_rel_cpos__ = self.pin_rel_cpos.clone().cpu().numpy()
        self.__ori_pin_rel_lpos__ = self.pin_rel_lpos.clone().cpu().numpy()
        self.__ori_pin_size__ = self.pin_size.clone().cpu().numpy()
        self.__ori_region_boxes__ = self.region_boxes.clone().cpu().numpy()
        dtype, device = self.die_info.dtype, self.die_info.device
        self.__die_shift__ = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
        self.__die_scale__ = torch.tensor([1.0, 1.0], dtype=dtype, device=device)
        return self

    def preshift(self):
        # shift die info to (0.0, hx, 0.0, hy)
        die_lx, _, die_ly, _ = self.die_info.tolist()
        die_shift = torch.tensor(
            [die_lx, die_ly], dtype=self.die_info.dtype, device=self.die_info.device,
        )
        self.die_info = (self.die_info.reshape(2, 2).t() - die_shift).t().reshape(-1)
        self.region_boxes = (
            (self.region_boxes.reshape(-1, 2, 2).permute(0, 2, 1) - die_shift)
            .permute(0, 2, 1)
            .reshape(-1, 4)
        )
        self.node_pos -= die_shift
        self.node_lpos -= die_shift
        self.__die_shift__ += die_shift
        return self

    def prescale_by_site_width(self):
        # inplace scaling
        self.die_info /= self.site_width
        self.region_boxes /= self.site_width
        self.node_pos /= self.site_width
        self.node_lpos /= self.site_width
        self.node_size /= self.site_width
        self.pin_rel_cpos /= self.site_width
        self.pin_rel_lpos /= self.site_width
        self.pin_size /= self.site_width
        self.__die_scale__ *= self.site_width
        return self

    def prescale(self):
        # scale die info to (0.0, 1.0, 0.0, 1.0)
        die_lx, die_hx, die_ly, die_hy = self.die_info.tolist()
        die_scale = torch.tensor(
            [die_hx - die_lx, die_hy - die_ly],
            dtype=self.die_info.dtype,
            device=self.die_info.device,
        )
        self.node_pos /= die_scale
        self.node_lpos /= die_scale
        self.node_size /= die_scale
        self.pin_rel_cpos /= die_scale
        self.pin_rel_lpos /= die_scale
        self.pin_size /= die_scale
        self.die_info = (self.die_info.reshape(2, 2).t() / die_scale).t().reshape(-1)
        self.region_boxes = (
            (self.region_boxes.reshape(-1, 2, 2).permute(0, 2, 1) / die_scale)
            .permute(0, 2, 1)
            .reshape(-1, 4)
        )
        self.__die_scale__ *= die_scale
        return self

    def pre_compute_var(self):
        args = self.__args__
        device = self.node_size.get_device()
        # die related
        lx, hx, ly, hy = self.die_info.tolist()
        self.unit_len = torch.tensor(
            [(hx - lx) / self.num_bin_x, (hy - ly) / self.num_bin_y], device=device
        )
        self.die_ur = self.die_info.reshape(2, 2).t()[1].clone()
        self.die_ll = self.die_info.reshape(2, 2).t()[0].clone()
        self.hpwl_scale = self.die_scale / self.site_width
        # node related
        self.node_area = torch.prod(self.node_size, 1).unsqueeze(1)
        self.node_to_num_pins = torch.zeros(self.num_nodes, device=device)
        v = torch.ones(self.pin_id2node_id.shape[0], device=device)
        self.node_to_num_pins.scatter_add_(0, self.pin_id2node_id, v).round_()
        self.node_to_num_pins.unsqueeze_(1)
        # net related
        start_idx = self.hyperedge_list_end.roll(1)
        start_idx[0] = 0
        self.net_to_num_pins = self.hyperedge_list_end - start_idx
        self.net_mask = torch.logical_and(
            self.net_to_num_pins <= args.ignore_net_degree, self.net_to_num_pins >= 2
        )  # 0: ignore, 1: consider in wirelength calculation
        # obj related
        mov_lhs, mov_rhs = self.movable_index
        mov_cell_area = torch.prod(self.node_size[mov_lhs:mov_rhs, ...], 1)
        self.__total_mov_area_without_filler__ = torch.sum(mov_cell_area).item()
        self.__bin_area__ = torch.prod(self.unit_len).item()
        return self

    def init_fence_region(self):
        offset = self.region_boxes_end.diff().tolist()
        regions = torch.split(self.region_boxes[1:], offset)
        self.regions = (
            self.region_boxes[0],
            *regions,
        )  # first region is the default region (core area)
        self.__num_regions__ = len(self.regions)
        self.__enable_fence__ = len(self.regions) > 1
        return self

    def compute_filler(self, args, logger):
        if self.enable_fence:
            return self.compute_filler_with_fence(args, logger)
        else:
            return self.compute_filler_without_fence(args, logger)

    def compute_filler_with_fence(self, args, logger):
        raise NotImplementedError("We haven't yet supported fence region.")

    def compute_filler_without_fence(self, args, logger):
        self.__num_fillers__ = 0
        if args.use_filler:
            mov_lhs, mov_rhs = self.movable_index
            mov_node_size = self.node_size[mov_lhs:mov_rhs, ...]
            die_area = torch.prod(self.die_ur - self.die_ll)
            # init_density_map already multiplies with args.target_density,
            # we need to divide it back
            ori_dmap = (self.init_density_map / args.target_density).sum()
            # init_density_map are all normalized to (0.0, 1.0)
            fixed_node_area = ori_dmap * self.bin_area
            placeable_area = die_area - fixed_node_area
            if True:
                mov_cell_area = torch.prod(mov_node_size, 1)
                num_movable_nodes = mov_rhs - mov_lhs
                mov_node_xsize_order = torch.argsort(mov_node_size[:, 0])
                filler_size_x = torch.mean(
                    mov_node_size[:, 0][
                        mov_node_xsize_order[
                            int(num_movable_nodes * 0.05) : int(
                                num_movable_nodes * 0.95
                            )
                        ]
                    ]
                )
                filler_size_y = self.site_height / self.die_scale[1]
                total_filler_area = max(
                    args.target_density * placeable_area - torch.sum(mov_cell_area),
                    0.0,
                )
                single_filler_size = torch.tensor(
                    [filler_size_x, filler_size_y],
                    device=mov_node_size.device,
                    dtype=mov_node_size.dtype,
                )
                self.__num_fillers__ = int(
                    torch.round(total_filler_area / (filler_size_x * filler_size_y))
                )
            else:
                mov_cell_area = torch.prod(mov_node_size, 1)
                total_filler_area = max(
                    args.target_density * placeable_area
                    - torch.sum(mov_cell_area).item(),
                    0.0,
                )
                single_filler_area = torch.mean(mov_cell_area)
                single_filler_size = single_filler_area.sqrt().repeat(2)
                self.__num_fillers__ = int(total_filler_area / single_filler_area)
            if self.num_fillers > 0:
                self.filler_size = single_filler_size.repeat(self.num_fillers, 1)
                logger.info(
                    "#Fillers: %d Filler size: (%.4e, %.4e)"
                    % (
                        self.num_fillers,
                        single_filler_size[0].item(),
                        single_filler_size[1].item(),
                    )
                )
            else:
                logger.warning(
                    "num_fillers[%d] is smaller or equal to 0. Please make sure target_density[%.2f]"
                    " is larger than movable cell utilization[%.2f]. use_filler is disable."
                    % (
                        self.num_fillers,
                        args.target_density,
                        torch.sum(torch.prod(mov_node_size, 1)),
                    )
                )
                args.use_filler = False

            die_area, placeable_area = die_area.item(), placeable_area.item()
            fixed_node_area, mov_cell_area = fixed_node_area.item(), torch.sum(mov_cell_area).item()
            total_filler_area = float(total_filler_area)
            logger.info(
                "DieArea: %.3E FixArea: %.3E (%.1f%%) PlaceableArea: %.3E (%.1f%%) MovArea: %.3E (%.1f%%) FillerArea: %.3E (%.1f%%)"
                % (
                    die_area,
                    fixed_node_area, fixed_node_area / die_area * 100,
                    placeable_area, placeable_area / die_area * 100,
                    mov_cell_area, mov_cell_area / die_area * 100,
                    total_filler_area, total_filler_area / die_area * 100,
                )
            )

        return self

    def compute_precond_var(self):
        mov_lhs, mov_rhs = self.movable_index
        self.mov_node_area = self.node_area[mov_lhs:mov_rhs]
        self.mov_node_to_num_pins = self.node_to_num_pins[mov_lhs:mov_rhs]
        if self.filler_size is not None:
            num_fillers = self.filler_size.shape[0]
            filler_area = torch.prod(self.filler_size, 1).unsqueeze(1)
            self.mov_node_area = torch.cat((self.mov_node_area, filler_area), dim=0)
            filler_to_num_pins = self.mov_node_to_num_pins.new_zeros((num_fillers, 1))
            assert filler_to_num_pins.shape == filler_area.shape
            self.mov_node_to_num_pins = torch.cat(
                (self.mov_node_to_num_pins, filler_to_num_pins), dim=0
            )
        return self

    def compute_sorted_node_map(self):
        _, mov_sorted_map = torch.sort(self.mov_node_area.flatten(), descending=True)
        mov_sorted_map = mov_sorted_map.contiguous()
        mov_conn_sorted_map = mov_sorted_map
        filler_sorted_map = None
        if self.filler_size is not None:
            mov_lhs, mov_rhs = self.movable_index
            _, mov_conn_sorted_map = torch.sort(
                self.mov_node_area[mov_lhs:mov_rhs].flatten(), descending=True
            )
            _, filler_sorted_map = torch.sort(
                self.mov_node_area[mov_rhs:].flatten(), descending=True
            )
            mov_conn_sorted_map = mov_conn_sorted_map.contiguous()
            filler_sorted_map = filler_sorted_map.contiguous()
        self.sorted_maps = (mov_sorted_map, mov_conn_sorted_map, filler_sorted_map)

    def logging_statistics(self):
        args = self.__args__
        logger = self.__logger__
        content = "\n===================\n"
        content += "#nodes = %d, #nets = %d, #pins = %d\n" % (
            self.num_nodes,
            self.num_nets,
            self.num_pins,
        )
        num_conmov_nodes = self.node_type_indices[0][1] - self.node_type_indices[0][0]
        num_fltmov_nodes = self.node_type_indices[1][1] - self.node_type_indices[1][0]
        num_confix_nodes = self.node_type_indices[2][1] - self.node_type_indices[2][0]
        num_fltfix_nodes = self.node_type_indices[6][1] - self.node_type_indices[6][0]
        num_coniopin = self.node_type_indices[3][1] - self.node_type_indices[3][0]
        num_fltiopin = self.node_type_indices[5][1] - self.node_type_indices[5][0]
        num_blkg = self.node_type_indices[4][1] - self.node_type_indices[4][0]
        content += "#Mov = %d, #Fix = %d, #IOPin = %d, #Blkg = %d\n" % (
            num_conmov_nodes + num_fltmov_nodes,
            num_confix_nodes + num_fltfix_nodes,
            num_coniopin + num_fltiopin,
            num_blkg,
        )
        content += (
            "#ConnMov = %d, #FloatMov = %d, #ConnFix = %d, #FloatFix = %d, #ConnIOPin = %d, #FloatIOPin = %d\n"
            % (
                num_conmov_nodes,
                num_fltmov_nodes,
                num_confix_nodes,
                num_fltfix_nodes,
                num_coniopin,
                num_fltiopin,
            )
        )
        content += "Core Info " + str(self.die_info.tolist()) + "\n"
        content += "Site Width = %d, Row Height = %d\n" % (
            self.site_width,
            self.site_height,
        )
        content += "#Bins = (%d, %d), UnitLen = (%.5f, %.5f)\n" % (
            self.num_bin_x,
            self.num_bin_y,
            self.unit_len[0],
            self.unit_len[1],
        )
        content += "target density = %.2f\n" % (args.target_density)
        content += "==================="
        logger.info(content)
        return self

    def preprocess(self):
        args = self.__args__
        self.backup_ori_var()
        self.preshift()
        self.prescale_by_site_width()
        if args.scale_design:
            self.prescale()
        self.pre_compute_var()
        self.init_fence_region()
        self.logging_statistics()
        return self

    def init_filler(self):
        self.compute_filler(self.__args__, self.__logger__)
        self.compute_precond_var()
        self.compute_sorted_node_map()
        return self

    def get_mov_node_info(self, init_method="randn_center"):
        args = self.__args__
        mov_lhs, mov_rhs = self.movable_index
        mov_node_pos = self.node_pos[mov_lhs:mov_rhs, ...].clone()
        mov_node_size = self.node_size[mov_lhs:mov_rhs, ...].clone()

        if init_method == "randn_center":
            scale = (self.die_ur - self.die_ll) * 0.001
            loc = (self.die_ur + self.die_ll) * 0.5
            mov_node_pos = torch.randn_like(mov_node_pos) * scale + loc

        if self.num_fillers > 0:
            if self.enable_fence:
                raise NotImplementedError("We haven't yet supported fence region.")
            else:
                filler_pos = torch.rand(
                    (self.num_fillers, 2),
                    dtype=mov_node_size.dtype,
                    device=mov_node_size.device,
                )
                scale = self.die_ur - self.die_ll
                shift = self.die_ll
                filler_pos = filler_pos * scale + shift
            mov_node_pos = torch.cat([mov_node_pos, filler_pos], dim=0)
            mov_node_size = torch.cat([mov_node_size, self.filler_size], dim=0)

        if args.noise_ratio > 0:
            noise = torch.rand_like(mov_node_pos)
            noise.sub_(0.5).mul_(mov_node_size).mul_(args.noise_ratio)
            mov_node_pos += noise

        expand_ratio = mov_node_pos.new_ones((mov_node_pos.shape[0]))
        if self.clamp_node:
            self.mov_node_size_real = mov_node_size.clone() # before expanding
            mov_node_area = torch.prod(mov_node_size, 1)
            clamp_mov_node_size = mov_node_size.clamp(min=self.unit_len * math.sqrt(2))
            clamp_mov_node_area = torch.prod(clamp_mov_node_size, 1)
            # update
            expand_ratio = mov_node_area / clamp_mov_node_area
            mov_node_size = clamp_mov_node_size

        return mov_node_pos, mov_node_size, expand_ratio

    def write_pl(self, node_pos, gp_prefix):
        # support floating point based .pl file in global placement output
        pl_file = gp_prefix + ".pl"
        content = "UCLA pl 1.0\n"
        # use float here
        exact_node_pos = node_pos * self.die_scale + self.die_shift
        exact_node_size = torch.round(self.node_size * self.die_scale)
        tmp = (exact_node_size.div(self.site_width, rounding_mode="floor") == 1).bool()
        is_terminal_ni = torch.logical_and(tmp[:, 0], tmp[:, 1]).cpu()
        exact_node_lpos = (
            (exact_node_pos - exact_node_size / 2).div_(self.site_width).cpu()
        )
        for i in range(self.num_nodes):
            content += "\n%s %g %g : %s" % (
                self.node_id2node_name[i],
                exact_node_lpos[i, 0],
                exact_node_lpos[i, 1],
                "N"
            )
            if i >= self.fixed_index[0]:
                if is_terminal_ni[i]:
                    content += " /FIXED_NI"
                else:
                    content += " /FIXED"
        with open(pl_file, "w") as f:
            f.write(content)
