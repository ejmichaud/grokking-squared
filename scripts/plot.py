from matplotlib import pyplot as plt
from celluloid import Camera
import torch
import os
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def standardize(x):
    return (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)


def get_events(path):
    return event_accumulator.EventAccumulator(path).Reload()


def get_scalars(event):
    scalars = {
        tag: np.array(list(map(lambda x: [x.step, x.value],
                               event.scalars.Items(tag)))) for tag in
        event.scalars.Keys()}
    return scalars


def read_scalars(path):
    event = get_events(path)
    scalars = get_scalars(event)
    return scalars


root = "/home/kitouni/projects/Grok/grokking-squared/runs/"
# non modular add
# timestamp = "0405-1454" #"0405-1354"
# directory = root + timestamp + "/modular-1-53/"

# modular add 0405-1532/modular-59-59
timestamp = "0405-1532"
directory = root + timestamp + "/modular-59-59/"


# modular add
# timestamp = "0405-1354"
# directory = root + timestamp + "/modular-53-53/"
scalars = read_scalars(glob.glob(directory + "events*")[0])

files = glob.glob(directory + "weights/*")
files = sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))


loss_train, loss_test, acc_train, acc_test = scalars[
    "loss/train"], scalars["loss/test"], scalars["acc/train"], scalars["acc/test"]


def animate(model, name):
    fig = plt.figure(figsize=(10, 10))
    camera = Camera(fig)
    pbar = tqdm(range(len(files)))
    for i in range(0, len(files), 10):
        file = files[i]
        representation = torch.load(file).cpu().numpy()
        emb = standardize(representation)
        transformed = model.fit_transform(emb)
        transformed = (transformed - transformed.min(0)
                       ) / (transformed.max(0) - transformed.min(0))
        cmap = plt.get_cmap('viridis')
        sc = plt.scatter(
            *transformed.T, c=np.arange(len(transformed)),
            s=200, cmap=cmap)
        title = os.path.basename(file).split(".")[0]
        title += f" Loss: {loss_train[i][1]:.2e}|{loss_test[i][1]:.2e}"
        title += f" Acc: {acc_train[i][1]:.2f}|{acc_test[i][1]:.2f}"
        plt.legend([sc], [title], fontsize=20, markerscale=0,
                   loc=(0, 1.02))
        camera.snap()
        pbar.update()
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    print(f"now animating {name}")
    tbar = tqdm(range(len(files)))
    animation = camera.animate(blit=True)
    animation.save(f"/home/kitouni/projects/Grok/grokking-squared/{name}.mp4",
                   writer="ffmpeg", progress_callback=lambda frame, total: tbar.update())
    plt.close()


pca = PCA(n_components=2)
tsne = TSNE(n_components=2, perplexity=10, n_iter=1000,
            init="pca", learning_rate="auto", )
animate(pca, timestamp + "_pca")
animate(tsne, timestamp + "_tsne")
