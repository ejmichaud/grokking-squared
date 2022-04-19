#!/usr/bin/env python
# coding: utf-8
"""
This script performs a training run on the toy model. There are 
lots of options for configuration.
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import copy

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("run-toy-model")
ex.captured_out_filter = apply_backspaces_and_linefeeds


class ToyModel(nn.Module):
    def __init__(self, digit_rep_dim,
                 internal_rep_dim,
                 encoder_width=50,
                 encoder_depth=3,
                 decoder_width=50,
                 decoder_depth=3,
                 activation=nn.Tanh,
                 device='cpu'):
        """A toy model for grokking with an encoder, a exact addition operation, and a decoder.

        Arguments:
            digit_rep_dim (int): Dimension of vectors representing symbols in binary op table
            internal_rep_dim (int): Dimension of encoded representation (usually 1 or 2)
            encoder_width (int): Width of MLP for the encoder
            encoder_depth (int): Depth of MLP for the encoder (a depth of 2 is 1 hidden layer)
            decoder_width (int): Width of MLP for the decoder
            decoder_depth (int): Depth of MLP for the decoder (a depth of 2 is 1 hidden layer)
            activation: PyTorch class for activation function to use for encoder/decoder MLPs
            device: device to put the encoder and decoder on
        """
        super(ToyModel, self).__init__()
        self.digit_rep_dim = digit_rep_dim
        self.internal_rep_dim = internal_rep_dim

        # ------ Create Encoder ------
        encoder_layers = []
        for i in range(encoder_depth):
            if i == 0:
                encoder_layers.append(nn.Linear(digit_rep_dim, encoder_width))
                encoder_layers.append(activation())
            elif i == encoder_depth - 1:
                encoder_layers.append(nn.Linear(encoder_width, internal_rep_dim))
            else:
                encoder_layers.append(nn.Linear(encoder_width, encoder_width))
                encoder_layers.append(activation())
        self.encoder = nn.Sequential(*encoder_layers).to(device)

        # ------ Create Decoder ------
        decoder_layers = []
        for i in range(decoder_depth):
            if i == 0:
                decoder_layers.append(nn.Linear(internal_rep_dim, decoder_width))
                decoder_layers.append(activation())
            elif i == decoder_depth - 1:
                decoder_layers.append(nn.Linear(decoder_width, digit_rep_dim))
            else:
                decoder_layers.append(nn.Linear(decoder_width, decoder_width))
                decoder_layers.append(activation())
        self.decoder = nn.Sequential(*decoder_layers).to(device)

    def forward(self, x):
        """Runs the toy model on input `x`.

        `x` must contain vectors of dimension 2 * `digit_rep_dim`, since it represents a pair
        of symbols that we want to compute our binary operation between.
        """
        x1 = x[..., :self.internal_rep_dim]
        x2 = x[..., self.internal_rep_dim:]
        # return self.decoder(self.encoder(x1) + self.encoder(x2))
        return self.decoder(x1 + x2)


def distance(x, y):
    """L2 distance between x and y."""
    assert x.shape == y.shape
    return torch.norm(x - y, 2)


# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    """To perform multiple runs with the same parameters, 
    simply set a different seed for each run."""
    p = 10               # number of symbols (numbers) on each axis of the addition table
    symbol_rep_dim = 10  # dimension of space to map symbols (numbers) into
    train_fraction = 0.8  # fraction of the table to use in the train set
    encoder_width = 200
    encoder_depth = 3
    hidden_rep_dim = 1   # dimension of internal representation (encoder output)
    decoder_width = 200
    decoder_depth = 3
    activation_fn = nn.Tanh
    train_steps = 2
    log_freq = 1  # train_steps / 100
    optimizer = torch.optim.Adam
    encoder_lr = 1e-3
    encoder_weight_decay = 0.0
    decoder_lr = 1e-3
    decoder_weight_decay = 0.0
    latent_init_scale = 1.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64


def ideal_test_acc(train_id, test_id, p):
    # computer ideal (test accuracy). This only depends on training dataset, assuming an ideal algorithm.
    all_num = len(train_id) + len(test_id)
    pairs = [(i, j) for (i, j) in product(range(p), range(p)) if i <= j]
    #-----------Parallelogram set---------#
    P0 = []  # P0 is the set of all possible parallelograms
    P0_id = []

    ii = 0
    for i in range(all_num):
        for j in range(i + 1, all_num):
            if np.sum(pairs[i]) == np.sum(pairs[j]):
                P0.append(frozenset({pairs[i], pairs[j]}))
                P0_id.append(ii)
                ii += 1

    P0_num = len(P0_id)  # P0_num is the number of elements in P0

    #---------Linear Equation set---------#
    A = []
    eq_id = 0

    for i1 in range(P0_num):
        i, j = list(P0[i1])[0]
        m, n = list(P0[i1])[1]
        if i + j == m + n:
            x = np.zeros(p,)
            x[i] = x[i] + 1
            x[j] = x[j] + 1
            x[m] = x[m] - 1
            x[n] = x[n] - 1
            A.append(x)
            eq_id = eq_id + 1

    A = np.array(A).astype(int)

    # P0(D)
    P0D_id = []

    #----------Predict testing accuracy with Parallelogram set-------------#
    ii = 0
    for i in range(all_num):
        for j in range(i + 1, all_num):
            if np.sum(pairs[i]) == np.sum(pairs[j]):
                if i in train_id and j in train_id:
                    P0D_id.append(ii)
                ii += 1

    P0D = []
    for i in P0D_id:
        P0D.append(P0[i])

    # P0D_c
    P0D_c_id = set(P0_id) - set(P0D_id)

    # PD
    PD_id = []

    for i in P0D_c_id:
        P0D_id_aug = copy.deepcopy(P0D_id)
        P0D_id_aug.append(i)
        P0D_aug = []
        for j in P0D_id_aug:
            P0D_aug.append(P0[j])
        null_dim_1 = np.sum(np.linalg.eigh(
            np.matmul(np.transpose(A[P0D_id]), A[P0D_id]))[0] < 1e-8)
        null_dim_2 = np.sum(np.linalg.eigh(
            np.matmul(np.transpose(A[P0D_id_aug]), A[P0D_id_aug]))[0] < 1e-8)
        if null_dim_1 == null_dim_2:
            PD_id.append(i)

    PD_id = PD_id + P0D_id

    PD = []
    for i in PD_id:
        PD.append(P0[i])

    RQI_ideal = len(PD) / P0_num  # This is ideal RQI

    # Dbar(D)
    Dbar_id = list(copy.deepcopy(train_id))

    for i1 in test_id:
        flag = 0
        for j1 in train_id:
            i, j = pairs[i1]
            m, n = pairs[j1]
            if {(i, j), (m, n)} in PD:
                flag = 1
                break
        if flag == 1:
            Dbar_id.append(i1)

    # accuracy in the ideal case. acc_ideal: the whole dataset; acc_ideal_test: only testing set.
    acc_ideal = len(Dbar_id) / all_num
    num_generalize = len(Dbar_id) - len(train_id)
    acc_ideal_test = num_generalize / len(test_id)
    return num_generalize, len(test_id), acc_ideal_test


# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(p,
        symbol_rep_dim,
        train_fraction,
        encoder_width,
        encoder_depth,
        hidden_rep_dim,
        decoder_width,
        decoder_depth,
        activation_fn,
        train_steps,
        log_freq,
        optimizer,
        encoder_lr,
        encoder_weight_decay,
        decoder_lr,
        decoder_weight_decay,
        latent_init_scale,
        device,
        dtype,
        seed):

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Define data set

    latent = dict()

    latent_tensor = latent_init_scale * torch.randn(p, hidden_rep_dim).to(device)
    latent_tensor.requires_grad = True
    for i in range(p):
        latent[i] = latent_tensor[i:i + 1]

    symbol_reps = dict()
    for i in range(2 * p - 1):
        symbol_reps[i] = torch.randn((1, symbol_rep_dim)).to(device)

    def get_i_from_rep(rep, symbol_reps):
        assert next(iter(symbol_reps.values())).shape == rep.shape
        for i, candidate_rep in symbol_reps.items():
            if torch.all(rep == candidate_rep):
                return i

    table = dict()
    pairs = [(i, j) for (i, j) in product(range(p), range(p)) if i <= j]
    all_num = len(pairs)
    for (i, j) in pairs:
        table[(i, j)] = (i + j)
    train_id = random.sample(
        list(np.arange(len(pairs))),
        int(len(pairs) * train_fraction))
    train_pairs = [pairs[i] for i in train_id]
    test_id = set(np.arange(len(pairs))) - set(train_id)
    test_pairs = [pairs[i] for i in test_id]

    # compute ideal test accuracy
    num_generalize, test_size, acc_ideal_test = ideal_test_acc(train_id, test_id, p)
    print("ideal test accuracy={}/{}={}".format(num_generalize, test_size, acc_ideal_test))

    train_data = (
        torch.cat([torch.cat((latent[i], latent[j]), dim=1) for i, j in train_pairs], dim=0),
        torch.cat([symbol_reps[table[pair]] for pair in train_pairs])
    )
    test_data = (
        torch.cat([torch.cat((latent[i], latent[j]), dim=1) for i, j in test_pairs], dim=0),
        torch.cat([symbol_reps[table[pair]] for pair in test_pairs])
    )

    # initialize model
    model = ToyModel(
        digit_rep_dim=symbol_rep_dim,
        internal_rep_dim=hidden_rep_dim,
        encoder_width=encoder_width,
        encoder_depth=encoder_depth,
        decoder_width=decoder_width,
        decoder_depth=decoder_depth,
        activation=activation_fn,
        device=device)

    optim = optimizer(
        [{'params': latent_tensor, 'lr': encoder_lr,
          'weight_decay': encoder_weight_decay},
         {'params': model.decoder.parameters(),
          'lr': decoder_lr, 'weight_decay': decoder_weight_decay}])
    loss_fn = nn.MSELoss()

    ex.info['train_losses'] = list()
    ex.info['train_accuracies'] = list()
    ex.info['test_losses'] = list()
    ex.info['test_accuracies'] = list()
    ex.info['steps'] = list()

    for step in tqdm(range(int(train_steps))):
        optim.zero_grad()

        # update train_data and test_data, since latent is updated
        train_data = (
            torch.cat([torch.cat((latent[i], latent[j]), dim=1) for i, j in train_pairs], dim=0),
            torch.cat([symbol_reps[table[pair]] for pair in train_pairs])
        )
        test_data = (
            torch.cat([torch.cat((latent[i], latent[j]), dim=1) for i, j in test_pairs], dim=0),
            torch.cat([symbol_reps[table[pair]] for pair in test_pairs])
        )
        x, y_target = train_data

        y_train = model(x)
        l = loss_fn(y_target, y_train)

        if step % int(log_freq) == 0:
            # record train accuracy and loss
            with torch.no_grad():
                correct = 0
                for i in range(x.shape[0]):
                    closest_rep = min(symbol_reps.values(),
                                      key=lambda
                                      pos_rep: distance(
                                          pos_rep, y_train[i: i + 1, ...]))
                    pred_i = get_i_from_rep(closest_rep, symbol_reps)
                    target_i = get_i_from_rep(y_target[i:i + 1, ...], symbol_reps)
                    if pred_i == target_i:
                        correct += 1
                ex.info['steps'].append(step)
                ex.info['train_accuracies'].append(correct / x.shape[0])
                ex.info['train_losses'].append(l.item())

            # record test accuracy and loss
            with torch.no_grad():
                x_test, y_test_target = test_data
                y_test = model(x_test)
                l_test = loss_fn(y_test_target, y_test)
                correct = 0
                for i in range(x_test.shape[0]):
                    closest_rep = min(
                        symbol_reps.values(),
                        key=lambda pos_rep: distance(pos_rep, y_test[i: i + 1, ...]))
                    pred_i = get_i_from_rep(closest_rep, symbol_reps)
                    target_i = get_i_from_rep(y_test_target[i:i + 1, ...], symbol_reps)
                    if pred_i == target_i:
                        correct += 1
                ex.info['test_accuracies'].append(correct / x_test.shape[0])
                ex.info['test_losses'].append(l_test.item())

        # backprop and step
        l.backward()
        optim.step()

    torch.save(model.state_dict(), "/tmp/model-final-state-dict.pt")
    ex.add_artifact("/tmp/model-final-state-dict.pt", "model-final-state-dict.pt")
    torch.save((symbol_reps, train_pairs, test_pairs,
               train_data, test_data), "/tmp/dataset.pt")
    ex.add_artifact("/tmp/dataset.pt", "dataset.pt")
