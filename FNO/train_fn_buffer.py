from .FN import *
from .utilities import *
from .Adam import *
from .data_generator import *
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, num_bin_x, num_bin_y, maxlen=10000, device=torch.device("cpu")):
        self.density_map_mem = torch.zeros((maxlen, num_bin_x, num_bin_y), dtype=torch.float32, device=device)
        self.grad_mat_mem = torch.zeros((maxlen, 2, num_bin_x, num_bin_y), dtype=torch.float32, device=device)
        self.maxlen = maxlen
        self.device = device
        self.ptr = 0

    def push(self, density_map, grad_mat):
        self.density_map_mem[self.ptr, ...] = density_map.to(self.device)
        self.grad_mat_mem[self.ptr, ...] = grad_mat.to(self.device)
        self.ptr += 1
        if self.ptr >= self.maxlen:
            self.ptr = 0

    def sample(self, batch_size):
        batch_idxs = np.random.randint(len(self), size=batch_size)
        return self.density_map_mem[batch_idxs], self.grad_mat_mem[batch_idxs]
        
    def __len__(self):
        return self.density_map_mem.shape[0]


def draw_2d_tensor(tensor, fig_path):
    plt.figure()
    im = plt.imshow(tensor.detach().cpu().numpy())
    plt.colorbar(im)
    plt.savefig(fig_path)
    plt.close()


def this_writer(x, y, o, pathnum, fig_folder, label="train"):
    draw_2d_tensor(x[0, :, :].squeeze(0), "%s/%d_%s_den.png" % (fig_folder, pathnum, label))
    draw_2d_tensor(y[0, :, :].squeeze(0), "%s/%d_%s_Ey_label.png" % (fig_folder, pathnum, label))
    draw_2d_tensor(o[0, :, :].squeeze(0), "%s/%d_%s_Ey.png" % (fig_folder, pathnum, label))


def train_FNO(args, logger):
    memory_saving_mode = False
    episode_len = args.inner_iter

    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    res_root = os.path.join(args.result_dir, args.exp_id)

    # traing data generator
    train_generator = DataGenerator(
        episode_len, args.macro_mode, device, args.num_x, args.num_y,
        random_transform=True, deterministic=args.deterministic
    )

    # model
    model = FNO2d(args.modes, args.modes, args.width, args.neck).to(device)
    model_name = "model_%sx%sx%s" % (args.width, args.neck, args.modes)
    # logger.info("#Params: %d" % count_params(model))
    criterion = LpLoss(size_average=False)
    if memory_saving_mode:
        buffer = ReplayBuffer(args.num_x, args.num_y, maxlen=args.ntrain, device=torch.device("cpu"))
    else:
        # faster but memory consuming
        buffer = ReplayBuffer(args.num_x, args.num_y, maxlen=args.ntrain, device=device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    pt_path = os.path.join(res_root, args.model_dir, "%s_epoch_%d.pt" % (model_name, 0))
    if not os.path.exists(os.path.dirname(pt_path)):
        os.makedirs(os.path.dirname(pt_path))
    # torch.save(model.state_dict(), pt_path)

    fig_folder = os.path.join(res_root, "figs")
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    train_evolutioner = train_generator.evolutioner()
    logger.info("Initializing buffer")
    for _ in tqdm(range(int(args.ntrain / episode_len))):
        den, _, grad = next(train_evolutioner)
        for sample_idx in range(den.shape[0]):
            buffer.push(den[sample_idx], grad[sample_idx])
    for ep in range(args.epochs):
        model.train()
        total_loss = 0
        t1 = time.time()
        sample_mean_visited = 4
        total_num_samples = args.ntrain * sample_mean_visited
        for it in tqdm(range(total_num_samples // args.batch_size)):
            x, y = buffer.sample(args.batch_size)
            x, y = x.to(device), y.to(device)

            x1 = x.unsqueeze(3)  # bs, s, s, 1
            y1 = y[:, 0, :, :] * args.scaler

            # x2 = x1.permute(0, 2, 1, 3).contiguous()
            # y2 = y[:, 1, :, :] * args.scaler

            optimizer.zero_grad()
            out1 = model(x1).squeeze(3)  # bs, s, s, 1
            # out2 = model(x2).squeeze(3).permute(0,2,1).contiguous()       #bs, s, s, 1

            loss = criterion(out1.view(args.batch_size, -1), y1.view(args.batch_size, -1))
            # loss = (criterion(out1.view(args.batch_size, -1), y1.view(args.batch_size, -1)) + \
            #        criterion(out2.view(args.batch_size, -1), y2.view(args.batch_size, -1))) / 2
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        # Visualize
        this_writer(x1, y1, out1, ep, fig_folder)

        # Re-init buffer
        for _ in tqdm(range(int(args.ntrain / episode_len))):
            den, _, grad = next(train_evolutioner)
            for sample_idx in range(den.shape[0]):
                buffer.push(den[sample_idx], grad[sample_idx])

        # Eval
        model.eval()
        x, y = buffer.sample(args.batch_size)
        x, y = x.to(device), y.to(device)

        x1 = x.unsqueeze(3)  # bs, s, s, 1
        y1 = y[:, 0, :, :] * args.scaler
        out1 = model(x1).squeeze(3)  # bs, s, s, 1
        eval_loss_1 = criterion(out1.view(args.batch_size, -1), y1.view(args.batch_size, -1))
        eval_loss_1 = eval_loss_1.item() / args.batch_size
        this_writer(x1, y1, out1, ep, fig_folder, label="eval1")

        x2 = x1.permute(0, 2, 1, 3).contiguous()
        y2 = y[:, 1, :, :] * args.scaler
        out2 = model(x2).squeeze(3).permute(0, 2, 1).contiguous()  #bs, s, s, 1
        eval_loss_2 = criterion(out2.view(args.batch_size, -1), y2.view(args.batch_size, -1))
        eval_loss_2 = eval_loss_2.item() / args.batch_size
        this_writer(x2, y2, out2, ep, fig_folder, label="eval2")
        del x2, y2, out2

        # Save model
        pt_path = os.path.join(
            res_root, args.model_dir, "%s_epoch_%d.pt" % (model_name, ep)
        )
        torch.save(model.state_dict(), pt_path)

        scheduler.step()
        # log
        train_avg_loss = total_loss / total_num_samples
        logger.info(
            "epoch: %d | train_loss: %.4f eval_loss_1: %.4f eval_loss_2: %.4f | time_used: %.2f"
            % (ep, train_avg_loss, eval_loss_1, eval_loss_2, time.time() - t1)
        )