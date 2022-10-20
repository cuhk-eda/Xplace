from .FN import *
from .utilities import *
from .Adam import *
from .data_generator import *
from tqdm import tqdm


def draw_2d_tensor(tensor, fig_path):
    plt.figure()
    im = plt.imshow(tensor.detach().cpu().numpy())
    plt.colorbar(im)
    plt.savefig(fig_path)
    plt.close()


def this_writer(x, y, o, pathnum, fig_folder):
    draw_2d_tensor(x[0, :, :].squeeze(0), "%s/train_%d_den.png" % (fig_folder, pathnum))
    draw_2d_tensor(y[0, :, :].squeeze(0), "%s/train_%d_Ey_label.png" % (fig_folder, pathnum))
    draw_2d_tensor(o[0, :, :].squeeze(0), "%s/train_%d_Ey.png" % (fig_folder, pathnum))


def train_FNO_straight(args, logger):
    episode_len = 20
    args.batch_size = 20
    device = torch.device(
        "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    res_root = os.path.join(args.result_dir, args.exp_id)

    # traing data generator
    train_generator = DataGenerator(
        episode_len, args.macro_mode, device, args.num_x, args.num_y
    )

    # model
    model = FNO2d(args.modes, args.modes, args.width, args.neck).to(device)
    model_name = "model_%sx%sx%s" % (args.width, args.neck, args.modes)
    # logger.info("#Params: %d" % count_params(model))
    criterion = LpLoss(size_average=False)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    pt_path = os.path.join(res_root, args.model_dir, "%s_epoch_%d.pt" % (model_name, 0))
    if not os.path.exists(os.path.dirname(pt_path)):
        os.makedirs(os.path.dirname(pt_path))
    torch.save(model.state_dict(), pt_path)

    fig_folder = os.path.join(res_root, "figs")
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    for ep in range(args.epochs):
        model.train()
        total_loss = 0
        t1 = time.time()

        train_evolutioner = train_generator.evolutioner()
        for _ in tqdm(range(int(args.ntrain / episode_len))):
            den, _, grad = next(train_evolutioner)
            x, y = den, grad

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

        this_writer(x1, y1, out1, ep, fig_folder)

        pt_path = os.path.join(
            res_root, args.model_dir, "%s_epoch_%d.pt" % (model_name, ep + 1)
        )
        torch.save(model.state_dict(), pt_path)

        train_avg_loss = total_loss / args.ntrain
        scheduler.step()

        logger.info(
            "epoch: %d | train_loss: %.4f | time_used: %.2f"
            % (ep, train_avg_loss, time.time() - t1)
        )
