from utils import *
from src import *
from cpp_to_py import density_map_cuda
import torch.optim

import numpy as np
import matplotlib.pyplot as plt

class DummyLogger:
    def info(self, *args):
        return
    def warning(self, *args):
        return


class DummyArgs(object):
    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, value)


def generate_training_data_single(args, logger, device):
    data, rawdb, gpdb = load_dataset(args, logger)
    data = data.to(device)
    data = data.preprocess()
    init_density_map = get_init_density_map(data, args, logger)
    data.init_filler()
    mov_node_pos, mov_node_size, expand_ratio = data.get_mov_node_info()
    mov_node_size = mov_node_size.clone().cpu()
    init_density_map = init_density_map.clone().cpu()
    del data, rawdb, gpdb
    return mov_node_size, init_density_map


def get_fno_data(dataset, num_bin_x, num_bin_y, device):
    args = {
        "dataset": dataset,
        "design_name": None,
        "num_bin_x": num_bin_x,
        "num_bin_y": num_bin_y,
        "scale_design": True,
        "clamp_node": False,
        "load_from_raw": False,
        "ignore_net_degree": 100,
        "target_density": 1.0,
        "use_filler": True,
        "noise_ratio": 0.025,
    }
    logger = DummyLogger()
    args = setup_dataset_args(DummyArgs(args))
    mul_params = sorted(
        get_multiple_design_params(dataset), key=lambda params: params["design_name"]
    )
    for i, params in enumerate(mul_params):
        args.design_name = params["design_name"]
        data_path = "./data/FNO/%s/%s_%dx%d.pt" % (
                     dataset, params["design_name"], num_bin_x, num_bin_y)
        if os.path.exists(data_path):
            data = torch.load(data_path)
            mov_node_size = data["mov_node_size"]
            init_density_map = data["init_density_map"] 
        else:
            mov_node_size, init_density_map = generate_training_data_single(args, logger, device)
            if not os.path.exists(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            torch.save({
                "mov_node_size": mov_node_size,
                "init_density_map": init_density_map,
            }, data_path)
        yield mov_node_size, init_density_map, params["design_name"]


def random_transform_2d_tensor(tensor, hflip=False, vflip=False):
    if np.random.rand(1) > 0.5:
        tensor = tensor.transpose(1, 0)
    if np.random.rand(1) > 0.5 and hflip:
        tensor = tensor.flip(1)
    if np.random.rand(1) > 0.5 and vflip:
        tensor = tensor.flip(0)
    return tensor.contiguous()


class DataGenerator:
    def __init__(
        self, batch_size, mode, device, num_bin_x=512, num_bin_y=512, random_transform=False
    ):
        self.batch_size = batch_size
        self.mode = mode
        self.device = device
        self.num_bin_x = num_bin_x
        self.num_bin_y = num_bin_y
        self.dataset = "ispd2005"
        self.dtype = torch.float32
        self.random_transform = random_transform

        self.init_list = []
        self.cell_list = []
        for data in get_fno_data(self.dataset, num_bin_x, num_bin_y, device):
            mov_node_size, init_density_map, _ = data
            self.init_list.append(init_density_map.to(device))
            self.cell_list.append(mov_node_size.to(device))

        self.cell_list_len = len(self.cell_list)

        self.unit_len = torch.tensor(
            [1.0 / self.num_bin_x, 1.0 / self.num_bin_y], device=self.device, dtype=self.dtype
        )
        self.density_map_layer = ElectronicDensityLayer(
            unit_len=self.unit_len,
            num_bin_x=self.num_bin_x,
            num_bin_y=self.num_bin_y,
            device=self.device,
            overflow_helper=(None, None, None),
            sorted_maps=None,
        ).to(self.device)

    def generator(self, move_node=True, starter=0.5):
        ########################################
        #
        # mode bench: macro of benchmarks
        # mode rand: random macro
        ########################################

        batch_density_map = torch.zeros(
            (self.batch_size, self.num_bin_x, self.num_bin_y), device=self.device, dtype=self.dtype,
        )
        batch_potential_map = torch.zeros(
            (self.batch_size, self.num_bin_x, self.num_bin_y), device=self.device, dtype=self.dtype,
        )
        batch_grad_map = torch.zeros(
            (self.batch_size, 2, self.num_bin_x, self.num_bin_y), device=self.device, dtype=self.dtype,
        )

        cell_set = torch.randint(0, self.cell_list_len, (self.batch_size,))
        aggre_set = torch.randint(1, 20, (self.batch_size,))

        if self.mode == "rand":
            macro_nums = torch.randint(30, 60, (self.batch_size,))
            init_num = int(self.batch_size / 10)
            macro_nums = torch.randint(30, 60, (init_num,))
            batch_init_density_map = []
            for k in range(init_num):
                macro_pos = (0.5 * (abs(torch.randn(macro_nums[k], 2)) + 1 - \
                    abs(torch.randn(macro_nums[k], 2)))).clamp(0.1, 0.9).to(self.device)
                macro_size = (0.05 + torch.rand(macro_nums[k], 2) * 0.15).to(self.device)
                naive_mode = torch.ones(macro_nums[k], device=self.device, dtype=torch.bool)
                node_weight = torch.ones(macro_nums[k], device=self.device, dtype=self.dtype).detach()
                init_density_map = density_map_cuda.forward(
                    macro_pos,
                    macro_size,
                    node_weight,
                    naive_mode,
                    self.zeros_density_map,
                    self.unit_len,
                    self.num_bin_x,
                    self.num_bin_y,
                ).clamp(0, 1)
                batch_init_density_map.append(init_density_map)

        elif self.mode == "bench":
            init_num = len(self.init_list)
            batch_init_density_map = self.init_list

        for i in range(self.batch_size):
            j = int(torch.floor(torch.rand(1) * init_num).item())
            init_density_map = batch_init_density_map[j]
            # cells
            # cell_set = torch.randint(0,self.cell_list_len,(self.batch_size,))
            # mov_node_size = torch.tensor(self.cell_list[cell_set[i]]).to(self.device)
            # mov_node_size = torch.tensor(self.cell_list[j]).to(self.device)
            mov_node_size = self.cell_list[j]

            mov_node_nums = mov_node_size.shape[0]

            mov_node_pos = torch.zeros(mov_node_nums, 2)

            start = int(
                (torch.rand(1) * (1 - starter) + starter) * mov_node_nums
            )  # lower bound
            end = mov_node_nums - start
            mov_node_pos_unif = torch.rand(start, 2).clamp(0, 1)
            mov_node_pos_circle = torch.zeros(end, 2)

            splits = torch.zeros(aggre_set[i] + 2)
            splits[1:-1], _ = (torch.rand(aggre_set[i]) * end).int().sort()
            splits[-1] = end

            offsets = 0.1 + torch.rand(aggre_set[i] + 1, 2) * 0.8
            rs = torch.rand(aggre_set[i] + 1) * 0.05 + 0.05
            for j in range(len(rs)):
                r = rs[j]
                nums = int(splits[j + 1] - splits[j])
                # circle = torch.rand(nums, 1) / 2 * r

                circle = (
                    0.5 * (abs(torch.randn(nums, 1)) + 1 - abs(torch.randn(nums, 1)))
                ) * r

                angle = torch.rand(nums, 1) * 2 * math.pi
                x = circle * torch.cos(angle)
                y = circle * torch.sin(angle)
                mov_node_pos_circle[int(splits[j]) : int(splits[j + 1]), :] = \
                    torch.cat((x, y), dim=1) + offsets[j]

            mov_node_pos = torch.cat((mov_node_pos_unif, \
                mov_node_pos_circle), dim=0).clamp(0, 1).to(self.device)

            if move_node:
                return mov_node_pos, mov_node_size, init_density_map

            potential_map, density_map, grad_mat, _ = self.density_map_layer.generator(
                mov_node_pos, mov_node_size, init_density_map  # macro_only, init_coef
            )

            batch_density_map[i, :, :] = density_map
            batch_potential_map[i, :, :] = potential_map
            batch_grad_map[i, :, :, :] = grad_mat
        return batch_density_map, batch_potential_map, batch_grad_map

    def evolutioner(self, inner_iter=100, starter=0):
        
        batch_density_map = torch.zeros(
            (self.batch_size, self.num_bin_x, self.num_bin_y), device=self.device, dtype=self.dtype,
        )
        batch_potential_map = torch.zeros(
            (self.batch_size, self.num_bin_x, self.num_bin_y), device=self.device, dtype=self.dtype,
        )
        batch_grad_map = torch.zeros(
            (self.batch_size, 2, self.num_bin_x, self.num_bin_y), device=self.device, dtype=self.dtype,
        )
        outer_count = 0
        inner_count = 0
        mov_node_pos, mov_node_size, init_density_map = self.generator(True, starter)

        mov_node_pos = mov_node_pos.to(self.device)
        mov_node_pos = mov_node_pos.requires_grad_(True)
        node_pos_lb = mov_node_size / 2 + 1e-5
        node_pos_ub = 1 - mov_node_size / 2 - 1e-5

        def truc_node_pos(x):
            x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
            return x

        optimizer = torch.optim.Adam([mov_node_pos], 1e-3)

        _, mov_sorted_map = torch.sort(
            torch.prod(mov_node_size, 1).flatten(), descending=True
        )
        self.density_map_layer.sorted_maps = (mov_sorted_map, None, None)
        optimizer.zero_grad()
        torch.sum(mov_node_pos).backward()  # trick
        optimizer.zero_grad()

        while inner_count < inner_iter:
            for i in range(self.batch_size):

                optimizer.zero_grad()
                mov_node_pos = truc_node_pos(mov_node_pos)

                potential_map, density_map, grad_mat, node_grad = self.density_map_layer.generator(
                    mov_node_pos, mov_node_size, init_density_map, calc_node_grad=True # macro_only, init_coef
                )

                # plt.figure()
                # im = plt.imshow(density_map.squeeze(0).detach().cpu().numpy())
                # plt.colorbar(im)
                # plt.savefig("./figs_elec/_test.png")
                # plt.close()

                mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()
                mov_node_pos.grad.copy_(node_grad)

                batch_density_map[i, :, :].data.copy_(density_map)
                batch_potential_map[i, :, :].data.copy_(potential_map)
                batch_grad_map[i, :, :, :].data.copy_(grad_mat)

                # density_loss = torch.sum(potential_map)
                # density_loss.backward(retain_graph=True)

                optimizer.step()

                del density_map, potential_map, grad_mat
                # torch.cuda.empty_cache()

            yield batch_density_map, batch_potential_map, batch_grad_map
            inner_count += self.batch_size
            outer_count += self.batch_size
            starter = outer_count / 2000
            starter = min(starter, 1)
            if inner_count >= inner_iter:
                inner_count = 0

                del mov_node_pos, mov_node_size, init_density_map, node_pos_lb, node_pos_ub, mov_sorted_map
                # torch.cuda.empty_cache()

                mov_node_pos, mov_node_size, init_density_map = self.generator(True, starter)

                if self.random_transform:
                    init_density_map = random_transform_2d_tensor(init_density_map)

                mov_node_pos = mov_node_pos.to(self.device)
                mov_node_pos = mov_node_pos.requires_grad_(True)
                node_pos_lb = mov_node_size / 2 + 1e-5
                node_pos_ub = 1 - mov_node_size / 2 - 1e-5

                def truc_node_pos(x):
                    x.data.clamp_(min=node_pos_lb, max=node_pos_ub)
                    return x

                optimizer = torch.optim.Adam([mov_node_pos], 1e-3)
                optimizer.zero_grad()
                torch.sum(mov_node_pos).backward()  # trick
                optimizer.zero_grad()

                _, mov_sorted_map = torch.sort(
                    torch.prod(mov_node_size, 1).flatten(), descending=True
                )
                self.density_map_layer.sorted_maps = (mov_sorted_map, None, None)
