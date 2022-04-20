# %%
from itertools import product
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from models import Decoder
from utils import split_dataset
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
# Hyper parameters
EPOCHS = 50000
SEED = 2
CONCAT = False
PLOT = False
torch.manual_seed(SEED)

# X = torch.rand(100, 2, 2)
# Y = X.norm(dim=1).sum(-1)
# unit_circle_cart = torch.hstack([r * torch.cos(theta), r * torch.sin(theta)])
p = 10
theta = torch.linspace(0, 2 * torch.pi, p).view(-1, 1)
r = torch.tensor([1])
unit_circle_polar = torch.hstack([r * torch.ones_like(theta), theta])
index_pairs = torch.tensor(list(product(range(p), range(p))))
X = unit_circle_polar[index_pairs]
Y = X[:, :, 1].sum(-1, keepdim=True).view(p, p)
train_idx, test_idx = split_dataset(index_pairs, split_ratio=0.5)

representation = torch.randn_like(unit_circle_polar).requires_grad_()
model = Decoder(input_dim=2, output_dim=1, w=10, concat=CONCAT)
param_groups = [{"params": (representation, ), "lr": 0.1},
                {"params": model.parameters(), "lr": 0.01}]
optimizer = optim.Adam(param_groups)
loss_func = nn.MSELoss()
accuracy = Accuracy()
# Adding is some sort of common denominator for various problems


def step(idx):
    pred = model(representation[idx])
    target = Y[idx[:, 0], idx[:, 1]].view(-1, 1)
    loss = loss_func(pred, target)
    acc = accuracy(pred, target)
    return loss


def train(idx):
    optimizer.zero_grad()
    loss = step(idx)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(idx):
    with torch.no_grad():
        loss = step(idx)
    return loss.item()


topbar = tqdm(bar_format='{desc}{postfix}')
topbar.set_description("Train | Test")
pbar = tqdm(total=EPOCHS)


def next_step():
    for indices, *_ in train_idx:
        train_loss = train(indices)
    for indices, *_ in test_idx:
        test_loss = test(indices)
    pbar.set_description(f"Loss {train_loss:.2e} | {test_loss:.2e}")
    pbar.update()


# %%
if PLOT:
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    sc = plt.scatter([], [])

    def init():
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        return sc,

    def update(frame):
        for i in range(10):
            next_step()
        sc.set_offsets(representation.detach().cpu().numpy())
        return sc,

    ani = FuncAnimation(fig, update, frames=EPOCHS,
                        init_func=init, blit=True, repeat=False)
    plt.show()
else:
    for i in range(EPOCHS):
        next_step()

# %%
