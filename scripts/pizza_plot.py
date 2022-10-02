import numpy as np
import matplotlib.pyplot as plt
from tbparse import SummaryReader
import torch
from phasegrok.utils.utils import read_scalars, standardize, make_path
from sklearn.decomposition import PCA
import os
import glob
plt.style.use("mystyle-bright")

root = "/home/kitouni/projects/Grok/grokking-squared/runs/"
# Namespace(batch_size=-1, epochs=20000, lr_decoder=0.0002, lr_rep=0.001, dropout=0.05, weight_decay=1.0, rep_noise=False, tune=False, log=20, save_weights=True, plot=0, perfect_rep=False, seed=1, p=59, m=59, split_ratio=0.8, latent_dim=256, decoder_width=128, exp_name='modular add', device='cuda:0', loss='cross_entropy')
# name = "0425-1047/modular-59-59" # nice circle but unordered
name = "modular-addition59-deep/0427-1456"
# name = "addition59-mse/0512-175847"
fname_prefix = "un_normed"
# new data testing
# root = "/data/kitouni/modular-addition59-deep-entropyEdition/"
# name = "0504-190528"
# root = "/home/kitouni/projects/Grok/grokking-squared/runs/modular-mudltip59/"
# name = "0510-222504"
# fname_prefix = "modular-multip59"

directory = os.path.join(root, name)
# scalars = read_scalars(glob.glob(os.path.join(directory, "events*"))[0])
scalars = SummaryReader(directory, pivot=True).scalars

files = glob.glob(os.path.join(directory, "weights/*.embd"))
files = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))

loss_train, loss_test, acc_train, acc_test = scalars[
    "loss/train"], scalars["loss/test"], scalars["acc/train"], scalars["acc/test"]

n_repr = torch.load(files[0]).cpu().numpy().shape[0]


cmap = plt.get_cmap('viridis')
fig, axes = plt.subplots(1, 3, dpi=300, sharex=True, sharey=True, figsize=(13, 4), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})
ax = axes[0]
ax.set_xlim(-0.05, 1.1)
ax.set_ylim(-0.05, 1.1)


def update(i, model, title=None, sc=None, text=None, text2=None):
    file = files[i]
    representation = torch.load(file).cpu().numpy()
    emb = standardize(representation)
    transformed = model.fit_transform(emb)
    if isinstance(model, PCA):
        transformed = transformed[:, :2]
    # transformed = (transformed - transformed.min(0)
    #                ) / (transformed.max(0) - transformed.min(0))
    sc.set_offsets(transformed)

    # subtitle = os.path.basename(file).split(".")[0]
    # subtitle += f"\nLoss: {loss_train[i][1]:.2e}|{loss_test[i][1]:.2e}"
    subtitle = f" train acc: {acc_train[i]:.1f} | val acc: {acc_test[i]:.1f}"
    # if isinstance(model, PCA):
    #     p = model.explained_variance_ratio_
    #     entropy = -np.sum(p * np.log(p))
    #     dimension = np.exp(entropy)
    #     subtitle += f" S: {entropy:.2f}"
    #     subtitle += f" D: {dimension:.2f}"

    text.set_text(title)
    text2.set_text(subtitle)
    return sc, text


if __name__ == "__main__":
    pca = PCA()
    n = len(files)
    epochs = [int(file.split("/")[-1].split(".")[0]) for file in files]
    titles = ["Initialization (0 iterations)", "Overfitting (1000 iterations)",
              "Representation Learning (20000 iterations)"]
    for index, filenumber in enumerate([0, 30, n-1]):
        print(f"Plotting epoch {epochs[filenumber]}")
        ax = axes[index]
        ax.set_axis_off()
        sc = ax.scatter(*np.ones((2, n_repr)), c=np.arange(n_repr),
                        cmap=cmap, label="train", s=100)
        text = ax.text(0.5, 1.01, "", transform=ax.transAxes,
                    fontdict={"size": "x-large"}, ha='center')
        text2 = ax.text(0.5, .96, "", transform=ax.transAxes,
                        fontdict={"size": "large"}, ha='center')
        sc.set_paths([make_path(f"${m:02d}$") for m in range(n_repr)])
        update(filenumber, pca, title=titles[index], sc=sc, text=text, text2=text2)
    figname = "-".join(name.split("/"))
    filename = f"/home/kitouni/projects/Grok/grokking-squared/paper-plots/3step_representations/{fname_prefix}{figname}.pdf"
    # plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.savefig(filename, bbox_inches='tight')
    print(f"saved to {filename}")
