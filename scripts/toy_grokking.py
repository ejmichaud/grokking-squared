import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import os
from models import AE
from torch.nn.parameter import Parameter
from torchmetrics import Accuracy
from itertools import combinations

path = os.path.abspath(os.path.dirname(__file__))
runs_path = os.path.join('/'.join(path.split('/')[:-1]), 'runs')

# This notebook is a simplified version of toy_model.ipynb,
# only focusing on neural network training.

# -------------------- Part I: Hyperparameters -------------------- #
# encoder learning rate (gradient)/ representation learning rate (natural gradient)
ETA1 = 1e-2
ETA2 = 1e-4  # 1 works well for SGD  # decoder learning rate
DROPOUT = 0.8
SEED = 8  # random seed
device = "cuda:0"
input_dim = 10  # dimension of input random vector
latent_dim = 1  # dimension of representation space
output_dim = 10  # dimension of input random vector
p = 10  # base. i,j are integers in {0,1,2,...,p-1}.
EPOCHS = 200000  # training iterations
LOG = np.inf  # logging frequency
RQI = False
dec_w = 200  # decoder width
enc_w = 1  # encoder width
wd = 0  # decoder weight decay
# size of training set, no replacement (full dataset size=p(p+1)/2. 55 for p=10)
train_num = int(p * (p + 1) / 2 * .8)
print("train_num:", train_num)
modulo = False  # If true, o=i+j(mod p); else o=i+j.
# If true, use natural gradient; else use the common parameter gradient.
CONCAT = False
ENCODER_GRAD = "natural"  # "Natural" or "Parameter" or "Perfect"
OPTIM = torch.optim.AdamW
SOFTMAX = False  # If true, use softmax for classification.
LOSS = torch.nn.CrossEntropyLoss() if SOFTMAX else torch.nn.MSELoss()

hparams = {
    # "ETA1": ETA1, "ETA2": ETA2, "SEED": SEED, "input_dim": input_dim,
    # "latent_dim": latent_dim, "output_dim": output_dim, "p": p,
    "EPOCHS": EPOCHS,
    "DEC/width": dec_w, "DEC/wd": wd,
    "ENC/width": enc_w,
    "latent_dim": latent_dim,
    "EncoderGrad": ENCODER_GRAD,
    "CONCAT": CONCAT,
    "train_num": train_num,
    "Eta1": ETA1, "Eta2": ETA2, "modulo": modulo,
    "DROPOUT": DROPOUT,
    "OPTIM": OPTIM.__name__,
    "Seed": SEED,
    "p": p,
    # "train_num": train_num, "modulo": modulo
}


# -------------------- Part III: trainining neural networks -------------------- #
np.random.seed(SEED)
torch.manual_seed(SEED)

# dataset
# D0 is the full dataset, D0_id=[(0,0),(0,1),...,(p-1,p-1)].
#  D0 contains p*(p-1)/2 samples.
D0_id = []
xx_id = []  # xx_id is the list of i in (i,j) in D0_id. xx_id = [0,0,...,p-1]
yy_id = []  # yy_id is the list of j in (i,j) in D0_id. yy_id = [0,1,...,p-1]

for i in range(p):
    for j in range(i, p):
        D0_id.append((i, j))
        xx_id.append(i)
        yy_id.append(j)

xx_id = np.array(xx_id)
yy_id = np.array(yy_id)

all_num = int(p * (p + 1) / 2)
train_id = np.random.choice(all_num, train_num, replace=False)  # select training set id
test_id = np.array(list(set(np.arange(all_num)) - set(train_id)))  # select testing set id

# parallelogram set
P0 = []  # P0 is the set of all possible parallelograms
P = []
for i, j in combinations(train_id, 2):
    if sum(D0_id[i]) == sum(D0_id[j]):
        parallelogram = {D0_id[i], D0_id[j]}
        P0.append(parallelogram)
        P.append(parallelogram)

for i, j in combinations(test_id, 2):
    if sum(D0_id[i]) == sum(D0_id[j]):
        parallelogram = {D0_id[i], D0_id[j]}
        P0.append(parallelogram)

for id in test_id:
    pair = D0_id[id]
    for parallelogram in P:
        if set(pair) in tuple(map(set, parallelogram)):
            print(pair, "in", parallelogram)
            counter += 1
            continue
best_acc = counter / len(test_id)
print("best attainable test acc:", best_acc)
exit()

# for i in range(all_num):
#     for j in range(i + 1, all_num):
#         if sum(D0_id[i]) == sum(D0_id[j]):
#             P0.append({D0_id[i], D0_id[j]})
P0_num = len(P0)


# inputs
x_templates = torch.rand(p, output_dim)  # input random vectors
y_template_len = modulo if modulo else 2 * p - 1
y_templates = torch.arange(y_template_len) if SOFTMAX else torch.rand(
    y_template_len, output_dim)  # output random vectors


x_templates = x_templates.to(device)
y_templates = y_templates.to(device)
output_dim = y_template_len if SOFTMAX else output_dim
# labels
inputs_id = np.vstack([xx_id, yy_id]).T

out_id = (xx_id + yy_id) if modulo is False else (xx_id + yy_id) % modulo
out_id = torch.from_numpy(out_id)

out_id_train = out_id[train_id]
labels_train = y_templates[out_id_train].clone()
inputs_train = torch.cat(
    [x_templates[xx_id[train_id]],
     x_templates[yy_id[train_id]]],
    dim=1)

# testing set
out_id_test = out_id[test_id]
labels_test = y_templates[out_id_test].clone()
inputs_test = torch.cat([x_templates[xx_id[test_id]], x_templates[yy_id[test_id]]], dim=1)


# Define neural networks
model = AE(input_dim=input_dim, output_dim=output_dim, latent_dim=latent_dim,
           enc_w=enc_w, dec_w=dec_w, concat_decoder=CONCAT).to(device)


if ENCODER_GRAD.lower() == "natural":
    latent = Parameter(torch.rand(y_template_len, latent_dim).to(device))
    latent_params = [latent]
elif ENCODER_GRAD.lower() == "perfect":
    latent = torch.rand(1, latent_dim) * torch.arange(p).unsqueeze(1)
    latent_params = [Parameter()]
else:
    latent_params = model.enc.parameters()

param_groups = [{"params": model.dec.parameters(), "lr": ETA2, "weight_decay": wd},
                {"params": latent_params, "lr": ETA1}]
optimizer = OPTIM(param_groups)


# ----- training ----- #

test_acc_epochs = []
train_acc_epochs = []


logdir = time.strftime("%m%d-%H%M")
EXPERIMENT = f"DW{dec_w}-EG{ENCODER_GRAD}"
log_path = os.path.join(runs_path, f"{logdir}", EXPERIMENT)
writer = SummaryWriter(log_path)
writer.add_hparams(hparam_dict=hparams,
                   metric_dict={'final/test_acc': 0},
                   run_name=''
                   )
print("logdir:", log_path)

pbar = tqdm(range(EPOCHS))
pbar0 = tqdm(bar_format='{desc}{postfix}')
for epoch in pbar:
    optimizer.zero_grad()
    if ENCODER_GRAD.lower() == "parameter":
        latent = model.enc(x_templates)
    # update model parameters
    outputs_train = model.dec(latent, inputs_id[train_id])
    with torch.no_grad():
        outputs_test = model.dec(latent, inputs_id[test_id])
        loss_test = LOSS(outputs_test, labels_test)
    loss_train = LOSS(outputs_train, labels_train)
    loss_train.backward()
    optimizer.step()

    # calculate accuracy based on nearest neighbor

    if not SOFTMAX:
        pred_train_id = torch.argmin(
            torch.sum(
                (outputs_train.unsqueeze(dim=1) - y_templates.unsqueeze(dim=0)) ** 2,
                dim=2),
            dim=1)
        pred_test_id = torch.argmin(
            torch.sum(
                (outputs_test.unsqueeze(dim=1) - y_templates.unsqueeze(dim=0)) ** 2,
                dim=2),
            dim=1)
    else:
        pred_train_id = torch.argmax(outputs_train, dim=1)
        pred_test_id = torch.argmax(outputs_test, dim=1)
    acc = Accuracy()
    acc_train = acc(pred_train_id.cpu(), out_id_train)
    acc_test = acc(pred_test_id.cpu(), out_id_test)
    # whole accuracy
    acc_nn = (acc_train * train_id.shape[0] + acc_test * test_id.shape[0]) / all_num
    test_acc_epochs.append(acc_test)
    train_acc_epochs.append(acc_train)

    rqi = False
    if RQI:
        with torch.no_grad():
            latent_scale = latent / torch.std(latent, dim=0).unsqueeze(dim=0)
            for paralellogram in P0:
                (i, j), (m, n) = paralellogram
                dist = latent_scale[i] + latent_scale[j] - latent_scale[m] - latent_scale[n]
                if torch.norm(dist) < 1e-2:
                    rqi += 1
        rqi /= len(P0)

    # logging
    pbar.set_description("Test | Train")
    pbar0.set_description(
        f"Loss : {loss_test:.4f} | {loss_train: .4f} " +
        f"Acc: {acc_test: .4f} | {acc_train : .4f} \
        RQI: {rqi: .4f}")
    # pbar.set_postfix({"train_loss": loss_train, "test_loss": loss_test})
    # if epoch % LOG == 0:
    #     print("epoch: %d  | loss: %.8f " % (epoch, loss_train.detach().numpy()))

    writer.add_scalar('loss/train', loss_train.detach().item(), epoch)
    writer.add_scalar('loss/test', loss_test.detach().item(), epoch)
    writer.add_scalar('acc/train', acc_train, epoch)
    writer.add_scalar('acc/test', acc_test, epoch)
    writer.add_scalar('rqi', rqi, epoch)
writer.add_hparams(hparam_dict=hparams,
                   metric_dict={'final/test_acc': test_acc_epochs[-1]},
                   run_name=''
                   )
writer.close()
