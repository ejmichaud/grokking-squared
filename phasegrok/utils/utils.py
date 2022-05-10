from typing import Iterable, List, Dict
from torch import Tensor
from torch.utils.data import random_split, TensorDataset, DataLoader
from itertools import combinations
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import torch
from tensorboard.backend.event_processing import event_accumulator
from matplotlib.markers import MarkerStyle

# Matplotlib


def make_path(m):
    ms = MarkerStyle(m)
    return ms.get_path().transformed(ms.get_transform())

# tensorboard


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

# data-related


def standardize(x):
    return (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)


def get_loss(loss, n_classes=None):
    if loss == "mse":
        def loss_func(pred, target ,**kwargs):
          return torch.nn.functional.mse_loss(pred, torch.nn.functional.one_hot(
                target, n_classes).float(), **kwargs)
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
    train_set, val_set = random_split(
        dataset, [train_size, len(dataset) - train_size])
    bs = bs if isinstance(bs, Iterable) else [len(train_set), len(val_set)]
    return [DataLoader(train_set, batch_size=bs[0]),
            DataLoader(val_set, batch_size=bs[1])]


def to_tuple(x):
    return map(tuple, x)


def get_parallelograms(x):
    return [(pair1, pair2) for pair1, pair2 in combinations(to_tuple(x), 2)
            if sum(pair1) == sum(pair2)]


class Logger:
    def __init__(self, args=None, experiment="", timestamp=False, debug=False, log_every=1, save_weights=False, save_ckpt=False, model=None, hparam_dict={}, overwrite=False):
        self._timestamp = time.strftime("%m%d-%H%M%S") if timestamp else ""
        path = os.path.abspath(os.path.dirname(__file__))
        self.root_path = '/'.join(path.split('/')[:-2])
        self.experiment_dir = os.path.join(experiment, self._timestamp)
        self.log_path = os.path.join(self.root_path, self.experiment_dir)
        self.args = args
        self.epoch = 0
        self.plot_ready = False
        self.metrics = None
        self.writer = None
        self.debug = args.debug if args is not None else debug
        if not self.debug:
            self.log_every = args.log if args is not None else log_every
        else:
            self.log_every = 0
        self.save_weights = args.save_weights if args is not None else save_weights
        self.save_ckpt = args.save_ckpt if args is not None else save_ckpt
        self.model = model
        self.hparam_dict = args.__dict__ if args is not None else hparam_dict
        self.overwrite = overwrite
        if self.overwrite:
            if os.path.exists(self.log_path):
                print("Overwriting", self.log_path)
                os.system(f"trash {self.log_path}")
            os.makedirs(self.log_path)

    def _log_hyperparameters(self,):
        if self.debug:
            return
        self.writer = SummaryWriter(log_dir=self.log_path)
        self.writer.add_hparams(hparam_dict=self.hparam_dict,
                                metric_dict={'z': 0},
                                run_name='.'
                                )
        with open(os.path.join(self.log_path, 'hparams.txt'), 'w') as f:
            f.write(str(self.hparam_dict))
        

        os.makedirs(os.path.join(self.log_path, "weights"), exist_ok=True)
        if self.model is not None:
            torch.save(self.model, os.path.join(
                self.log_path, 'model.pt'))

    def log(self, metrics: Dict, weights=None, ckpt=None):
        if self.log_every < 1:
            return
        if self.epoch == 0:
            self._log_hyperparameters()
        if self.epoch % self.log_every == 0:
            for metric, value in metrics.items():
                self.writer.add_scalar(metric, value, self.epoch)
            self.metrics = metrics
            if self.save_weights:
                if weights is not None:
                    weights = weights.detach().cpu()
                    torch.save(weights, os.path.join(
                        self.log_path, 'weights', f"{self.epoch}.embd"))
            if self.save_ckpt:
                if ckpt is not None:
                    torch.save(ckpt, os.path.join(
                        self.log_path, 'weights', f'{self.epoch}.ckpt'))
        self.epoch += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()
            print("saved to", self.log_path)
        if self.plot_ready:
            plt.close("all")

    def plot_embedding(self, embedding, metrics=None, epoch=None):
        from celluloid import Camera
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
            sc = self.ax.scatter(
                *point, c=cmap([1 - i / len(x)]), marker=f"${i}$", s=200)
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
            filename = "-".join(self.experiment_dir.split("/"))
            filename += "-anim.mp4"
        if not filename.endswith('.mp4'):
            filename += '.mp4'
        filename = "plots/" + filename
        filename = os.path.join(self.root_path, filename)
        animation = self.camera.animate(blit=True)
        animation.save(filename)



from matplotlib import colors as mcolors
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

    
hex_list = ['0d47a1', '#c385ef', '#edd2ef', '#f8fca2', '#e6f7a0', '#d2f39b']


from matplotlib import cm
from matplotlib import colors as mcolors

def mycolors():
    array = cm.get_cmap('viridis_r', 256)
    array = array(np.linspace(0, 1, 256)) + np.array([0.2, 0.1, 0, 0])
    array = (array - array.min()) / (array.max() - array.min())
    newcolors = mcolors.ListedColormap(array)
    return newcolors
