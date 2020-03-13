import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from networks import Text2Mel, weight_init
from config import Config
from data import *
import argparse
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', dest='data_path', required=False, help='Path to the dataset')
parser.add_argument('--text', dest='text_path', required=False, help='Path to texts file')
parser.add_argument('--mel', dest='mel_path', required=False, help='Path to mel spectrograms')
parser.add_argument('-s', '--save', dest='save_dir', required=False, default='save', help='Where to save checkpoints')
parser.add_argument('-l', '--log', dest='log_dir', required=False, default='log', help='Where to save logs')
parser.add_argument('-r', '--restore', dest='restore_path', required=False, default=None,
                    help='Checkpoint to continue training from')
parser.add_argument('--batch_size', dest='batch_size', required=False, default=32, type=int)
parser.add_argument('--print_iter', dest='print_iter', required=False, default=100, type=int,
                    help='Print progress every x iterations')
parser.add_argument('--save_iter', dest='save_iter', required=False, default=10000, type=int,
                    help='Save checkpoint every x iterations')
parser.add_argument('--num_workers', dest="num_workers", required=False, default=8,
                    help="Number of processes to use for data loading")
parser.add_argument('--cc', dest="cc", action="store_true",
                    help="Set flag, if you do not want to use the current config file, instead of the config saved with"
                         " the checkpoint.")

def test():
    print(Config.sample_rate)

def plot_matrix(matrix, file):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix.detach().cpu())
    fig.colorbar(im)
    plt.title('{} Steps'.format(global_step))
    plt.savefig(file, format='png')
    plt.close(fig)


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.text_path:
        assert args.data_path is not None, "Data path not given"
        args.text_path = os.path.join(args.data_path, "lines.txt")
    if not args.mel_path:
        assert args.data_path is not None, "Data path not given"
        args.mel_path = os.path.join(args.data_path, "mel")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # When loading from checkpoint, first check for the config
    if args.restore_path is not None:
        print("Inspecting checkpoint: {}".format(args.restore_path))
        state = torch.load(args.restore_path, map_location=device)
        conf = state["config"]
        conflicts = False
        warning = "\nWARNING: Saved config does not match with current config file. Conflicts detected:"
        for key, value in conf:
            if getattr(Config, key) != value:
                conflicts = True
                warning += "\n      {}: '{}' vs. '{}'".format(key, value, getattr(Config, key))
        if conflicts:
            print(warning)
            if args.cc:
                print("Will use the current config file.\n")
            else:
                print("Will fall back to saved config. If you want to use the current config file, run with flag "
                      "'-cc'\n")
                Config.set_config(conf)

    # Tensorboard
    writer = SummaryWriter(args.log_dir)

    print("Loading Text2Mel...")
    net = Text2Mel().to(device)
    net.apply(weight_init)

    l1_criterion = nn.L1Loss().to(device)
    bd_criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    global_step = 0

    # Learning rate decay. Noam scheme
    warmup_steps = 4000.0
    def decay(_):
        step = global_step + 1
        return warmup_steps ** 0.5 * min(step * warmup_steps ** -1.5, step ** -0.5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)

    if args.restore_path is not None:
        print("Restoring from checkpoint: {}".format(args.restore_path))
        global_step = state["global_step"]
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        l1_criterion.load_state_dict(state["l1_criterion"])
        bd_criterion.load_state_dict(state["bd_criterion"])

    print("Loading dataset...")
    dataset = TTSDataset(args.text_path, args.mel_path, None, data_in_memory=True)
    batch_sampler = BucketBatchSampler(inputs=[d["text"] for d in dataset.data], batch_size=args.batch_size,
                                       bucket_boundaries=[i for i in range(1, Config.max_N - 1, 20)])
    data_loader = FastDataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                 num_workers=args.num_workers)

    # Prepare the W-Matrix for guided attention
    W = torch.zeros(1, Config.max_N, Config.max_T, dtype=torch.float32).to(device)
    for n in range(Config.max_N):
        for t in range(Config.max_T):
            # TODO: Is it "(.../2) * g^2" or "(...)/(2 * g^2)" ?
            W[0, n, t] = 1.0 - np.exp(-((n/float(Config.max_N) - t/float(Config.max_T))**2) / (2 * Config.g**2))

    print("Start training")
    while global_step < 100000000:
        for i, sample in enumerate(data_loader):
            mel = sample["mel"].to(device)
            text = sample["text"].to(device)

            optimizer.zero_grad()

            # When training on a wav of length T, we give the network almost all of the mel spectrogram except the last
            # part corresponding to time step T.
            # So network's job is to produce the last frame of the mel, given all the previous mel frames 1...T-1 as
            # input to AudioEnc.

            # As AudioEnc needs a spectrogram of the whole width, we add one frame of zeros to the beginning to compen-
            # sate the last frame of the mels we will cut off.
            zero_frame = torch.zeros(mel[:, :1, :].shape).to(device)
            S = torch.cat((zero_frame, mel[:, :-1, :]), dim=1)

            # Run Text2Mel
            Y_logits, Y, A, _ = net(text, S)

            l1_loss = l1_criterion(Y, mel)  # mel.transpose(1, 2)
            bd_loss = bd_criterion(Y_logits, mel)  # mel.transpose(1, 2)

            # Loss for guided attention:
            # Pad A matrix with zeros
            A_pad = F.pad(input=A, pad=[0, W.shape[2]-A.shape[2], 0, W.shape[1]-A.shape[1], 0, 0],
                          mode='constant',  value=0)
            W_batched = W.repeat(A_pad.shape[0], 1, 1)  # Repeat W tensor batch times
            att_loss = torch.sum(torch.abs(A_pad * W_batched)) / (A.shape[0] * A.shape[1] * A.shape[2])

            loss = l1_loss + bd_loss + att_loss
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            # Tensorboard
            writer.add_scalar('total loss', loss, global_step)
            writer.add_scalar('mel l1 loss', l1_loss, global_step)
            writer.add_scalar('mel bd loss', bd_loss, global_step)
            writer.add_scalar('guided attention loss loss', att_loss, global_step)

            if global_step % args.print_iter == 0:
                print("Step {}, L1={:.4f}, BD={:.4f}, Att={:.4f}, Total={:.4f}"
                      .format(global_step, l1_loss, bd_loss, att_loss, loss))

            if global_step % 1000 == 0:
                # Plot attention
                fig, ax = plt.subplots()
                im = ax.imshow(A.detach().cpu()[0])
                fig.colorbar(im)
                plt.title('{} Steps'.format(global_step))
                writer.add_figure('attention matrix', fig, global_step)

            if global_step % args.save_iter == 0:
                state = {
                    "global_step": global_step,
                    "config": Config.get_config(),
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "l1_criterion": l1_criterion.state_dict(),
                    "bd_criterion": bd_criterion.state_dict()
                }
                print("Saving checkpoint...")
                torch.save(state, os.path.join(args.save_dir, "checkpoint-{}.pth".format(global_step)))

            global_step += 1
