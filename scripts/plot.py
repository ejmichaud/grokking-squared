from time import strftime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import Normalize
import torch
import os
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from argparse import ArgumentParser
from itertools import product
from phasegrok.utils.utils import standardize, read_scalars, make_path, gen_log_space
from tbparse import SummaryReader

parser = ArgumentParser()
parser.add_argument("--which", type=str, default="both",
                    choices=["both", "tsne", "pca"], help="which dim reduction to animate")
parser.add_argument("--nskip", type=int, default=10,
                    help="how many frames to skip")
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--plot-model", action="store_true", default=False)
parser.add_argument("--fixed", action="store_true", default=False)

args = parser.parse_args()


root = "/home/kitouni/projects/Grok/grokking-squared/runs/"
# non modular add
# timestamp = "0405-1454" #"0405-1354"
# directory = root + timestamp + "/modular-1-53/"

# modular add 0405-1532/modular-59-59
# name = "0405-1532/modular-59-59/"
# name = "0425-1047/modular-59-59"
# Namespace(batch_size=-1, epochs=20000, lr_decoder=0.0002, lr_rep=0.001, dropout=0.05, weight_decay=1.0, rep_noise=False, tune=False, log=20, save_weights=True, plot=0, perfect_rep=False, seed=1, p=59, m=59, split_ratio=0.8, latent_dim=256, decoder_width=128, exp_name='modular add', device='cuda:0', loss='cross_entropy')
# name = "0425-1356/modular-97-97" # 5e-5 decoder learning rate
# name = "0426-1018/modular-97-97"  # 1e-5 decoder learning rate
name = "modular-addition59-deep/0427-1456" #  one layer deeper decoder
# name = "modular-addition59-shallow/0427-1600" # this time actually shallow
# name = "modular-addition59-mlp/0429-1337"
# name = "modular-addition59-mlp/0429-1349"
name = args.name or name
directory = os.path.join(root, name)

# modular add
# timestamp = "0405-1354"
# directory = root + timestamp + "/modular-53-53/"
scalars = read_scalars(glob.glob(os.path.join(directory, "events*"))[0])

repr_files = glob.glob(os.path.join(directory, "weights/*.embd"))
repr_files = sorted(repr_files, key=lambda x: int(os.path.basename(x).split(".")[0]))
model_files = glob.glob(os.path.join(directory, "weights/*.ckpt"))
if args.plot_model:
    model = torch.load(os.path.join(directory, "model.pt"))
    model.eval()


loss_train, loss_test, acc_train, acc_test = scalars[
    "loss/train"], scalars["loss/test"], scalars["acc/train"], scalars["acc/test"]

# loss_train, loss_test, acc_train, acc_test = [np.ones((len(repr_files), 2))] * 4


def apply_model(x, file):
    if args.plot_model:
        model.load_state_dict(torch.load(file))
        return model(x).detach().numpy().argmax(axis=1).reshape(-1)
    else:
        return x

def animate(reducer, name, nskip=1):
    n_repr, len_repr = torch.load(repr_files[0]).cpu().numpy().shape
    cmap = cm.get_cmap("viridis").reversed()
    embdding_colors = 'k' if args.plot_model else np.arange(n_repr)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_axis_off()
    sc = ax.scatter(*np.ones((2, n_repr)), c=embdding_colors, #c=np.arange(n_repr),
                    cmap=cmap, s=100)
    text = ax.text(0., 1.01, "", transform=ax.transAxes, fontdict={"size": 12})
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    sc.set_paths([make_path(f"${m:02d}$") for m in range(n_repr)])
    # plt.colorbar(sc, fraction=0.046, pad=0.04)

    if len_repr == 2 and args.plot_model:
        cmap_grid = cm.get_cmap("coolwarm")
        linspace = np.linspace(-0.05, 1.05, 250)
        grid = torch.tensor(list(product(*[linspace]*2))).float()
        heatmap = ax.scatter(*grid.T, s=7, alpha=1, zorder=0, marker="s")
        out_classes = model(grid[[0]]).shape[1]
        plt.colorbar(cm.ScalarMappable(norm=Normalize(0, out_classes - 1), cmap=cmap_grid), ax=ax, fraction=0.046, pad=0.04)

    def update(i):
        file = repr_files[i]
        representation = torch.load(file).cpu().numpy()
        if representation.shape[1] == 2 and args.plot_model:
            transformed = reducer.fit_transform(representation)

            # transformed = representation
            grid_to_repr = grid * (transformed.max(0) - transformed.min(0)) + transformed.min(0)
            output = apply_model(grid_to_repr, model_files[i])
            output = (output - 0) / (n_repr - 1)
            heatmap.set_color(cmap_grid(output))
            
        else:
            # emb = standardize(representation)
            emb = representation
            if args.fixed:
                transformed = reducer.transform(emb)
            else:
                transformed = reducer.fit_transform(emb)
        if isinstance(reducer, PCA):
            transformed = transformed[:, :2]
        transformed = (transformed - transformed.min(0)
                       ) / (transformed.max(0) - transformed.min(0))
        sc.set_offsets(transformed)
        title = os.path.basename(file).split(".")[0]
        title += f"\nLoss: {loss_train[i][1]:.2e}|{loss_test[i][1]:.2e}"
        title += f" Acc: {acc_train[i][1]:.2f}|{acc_test[i][1]:.2f}"
        if isinstance(reducer, PCA) and not args.plot_model:
            p = reducer.explained_variance_ratio_
            entropy = -np.sum(p * np.log(p))
            dimension = np.exp(entropy)
            # title += f" S: {entropy:.2f}"
            # title += f" D: {dimension:.2f}"

        text.set_text(title)
        if representation.shape[1] == 2 and args.plot_model:
            return heatmap, sc, text
        else:
            return sc, text

    print(f"now animating {name}")
    # range_obj = range(0, len(repr_files), nskip)
    range_obj = gen_log_space(len(repr_files), len(repr_files) // nskip)
    tbar = tqdm(range_obj)
    animation = FuncAnimation(
        fig, update, frames=range_obj, repeat=False, blit=True)
    fname = f"/home/kitouni/projects/Grok/grokking-squared/{name}.mp4"
    animation.save(fname, fps=get_fps(nskip), 
                   writer="ffmpeg", progress_callback=lambda *_: tbar.update())
    plt.close()
    print(f"animation saved to {fname}")

def get_fps(nskip):
    if nskip <= 10:
        return 10
    elif nskip < 100:
        return 10
    else:
        return 2

if __name__ == "__main__":
    nskip = args.nskip
    which = ["tsne", "pca"] if args.which == "both" else [args.which]
    timestamp = args.name.split("/")[-1] if args.name is not None else None
    if "tsne" in which:
        tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                    init="pca", learning_rate="auto", )
        animate(tsne, "-".join(name.split("/")[-3:-1]) + "_tsne", nskip=nskip)
    if "pca" in which:
        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(torch.load(repr_files[-1]).cpu().numpy())
        name = "-".join(name.split("/")[-3:-1]) + f"_pca{nskip}"
        if timestamp is not None:
            name += f"_{timestamp}"
        if args.fixed:
            name += "_fixed"
        animate(pca, name, nskip=nskip)
        
