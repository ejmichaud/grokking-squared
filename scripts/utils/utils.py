from typing import Iterable, List, Dict
from torch import Tensor
from torch.utils.data import random_split, TensorDataset, DataLoader
from itertools import combinations
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from celluloid import Camera
import torch


def get_loss(loss):
    if loss == "mse":
        loss_func = torch.nn.functional.mse_loss
    elif loss == "cross_entropy":
        loss_func = torch.nn.functional.cross_entropy
    else:
        raise ValueError("Unknown loss function")
    return loss_func


def split_dataset(*Tensors: Tensor, split_ratio: float = 0.8,
                  bs: Iterable = None) -> List[DataLoader]:
    """
    Splits the dataset into training and validation sets.
    """
    dataset = TensorDataset(*Tensors)
    train_size = int(len(dataset) * split_ratio)
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    bs = bs if isinstance(bs, Iterable) else [len(train_set), len(val_set)]
    return [DataLoader(train_set, batch_size=bs[0]),
            DataLoader(val_set, batch_size=bs[1])]


def to_tuple(x):
    return map(tuple, x)


def get_parallelograms(x):
    return [(pair1, pair2) for pair1, pair2 in combinations(to_tuple(x), 2)
            if sum(pair1) == sum(pair2)]


class Logger:
    def __init__(self, args, experiment=""):
        self.timestamp = time.strftime("%m%d-%H%M")
        path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = '/'.join(path.split('/')[:-2])
        runs_path = os.path.join(self.root_path, 'runs')
        self.log_path = os.path.join(runs_path, f"{self.timestamp}", experiment)
        self.args = args
        self.epoch = 0
        self.plot_ready = False
        self.metrics = None
        self.writer = None
        self.log_every = args.log

    def _log_hyperparameters(self,):
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer.add_hparams(hparam_dict=self.args.__dict__,
                                metric_dict={'z': 0},
                                run_name='.'
                                )
        os.makedirs(os.path.join(self.log_path, "weights"), exist_ok=True)

    def log(self, metrics: Dict, weights=None):
        if self.log_every < 1:
            return
        if self.epoch == 0:
            self._log_hyperparameters()
        if self.epoch % self.log_every == 0:
            for metric, value in metrics.items():
                self.writer.add_scalar(metric, value, self.epoch)
            self.metrics = metrics
            if self.args.save_weights:
                if weights is not None:
                    weights = weights.detach().cpu()
                    torch.save(weights, os.path.join(
                        self.log_path, 'weights', f"{self.epoch}.embd"))
        self.epoch += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print("saved to", self.log_path)
        if self.plot_ready:
            plt.close("all")

    def plot_embedding(self, embedding, metrics=None, epoch=None):
        if not self.plot_ready:
            # Make a figure
            self.fig = plt.figure(figsize=(16, 10))
            self.camera = Camera(self.fig)
            # Set the axes
            self.ax = self.fig.add_subplot(111)
            self.plot_ready = True

            # Plot the points with labels

        if embedding.shape[1] >= 2:
            x, y = embedding[:, 0], embedding[:, 1]
        else:
            x, y = embedding[:, 0], np.arange(len(embedding))

        x = (x - x.min()) / (x.max() - x.min())
        y = (y - y.min()) / (y.max() - y.min())

        cmap = plt.get_cmap('viridis')
        for i, point in enumerate(zip(x, y)):
            sc = self.ax.scatter(*point, c=cmap([1 - i / len(x)]), marker=f"${i}$", s=200)
        # Set a title
        epoch = epoch if epoch is not None else self.epoch

        title = [f"{epoch}"]
        self.metrics = metrics if metrics is not None else self.metrics
        for metric, value in self.metrics.items():
            title.append(f"{metric}: {value:.4f}")
        title = '| '.join(title)

        plt.legend([sc], [title], fontsize=20, markerscale=0, loc=(0, 1.02),
                   )
        self.camera.snap()

    def save_anim(self, filename=None):
        if filename is None:
            filename = f"{self.timestamp}-anim.mp4"
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        filename = "plots/" + filename
        filename = os.path.join(self.root_path, filename)
        animation = self.camera.animate(blit=True)
        animation.save(filename)
