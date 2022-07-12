#!/usr/bin/env python
# coding: utf-8
"""
This script trains a transformer on algorithmic datasets, the setting of grokking.

    TODO: Allow for other optimizers like SGD as an alternative to AdamW
    TODO: Add option for full-model checkpoints or just embeddings
    TODO: Make positional encodings optional? - DONE
"""

from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from multigrok.data import ArithmeticDataset
from multigrok.transformer import Transformer

import seml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("train")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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
    
    operators = ['+']
    p = 59
    optimization_steps = 100000
    batch_size = -1                 # -1 -> entire dataset, 0 < batch_size < 1 -> fraction of dataset, batch_size > 1 -> batch_size
    n_layers = 2
    n_heads = 8
    d_model = 256
    use_positional_encoding = True
    dropout = 0.0
    non_linearity = 'relu'          # 'relu' or 'gelu'
    training_data_fraction = 0.8

    halve_abelian = False
    only_input_tokens = False

    embedding_lr = 1e-3
    decoder_lr = 1e-3
    embedding_weight_decay = 0.0
    decoder_weight_decay = 0.0
    eps = 1e-8

    log_freq = math.ceil(optimization_steps / 500)
    embeddings_save_freq = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(operators,
        p,
        optimization_steps,
        batch_size, 
        n_layers, 
        n_heads, 
        d_model,
        use_positional_encoding,
        dropout,
        non_linearity,
        training_data_fraction,
        halve_abelian,
        only_input_tokens,
        embedding_lr,
        decoder_lr,
        embedding_weight_decay,
        decoder_weight_decay,
        eps,
        log_freq,
        embeddings_save_freq,
        device,
        dtype,
        seed,
        _log):

    operators = list(operators)
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dataset = ArithmeticDataset(operators, p=p, halve_abelian=halve_abelian, only_input_tokens=only_input_tokens)
    n_train, n_val = int(training_data_fraction * len(dataset)), len(dataset) - int(training_data_fraction * len(dataset)) 
    train, val = torch.utils.data.random_split(dataset, [n_train, n_val], torch.Generator().manual_seed(seed))
    if batch_size == -1:
        bs = len(train)
    elif 0 < batch_size < 1:
        bs = int(batch_size * len(train))
    elif batch_size > 1 and batch_size <= len(train):
        bs = int(batch_size)
    else:
        raise Exception(f"Invalid batch_size config {batch_size}.")
    train_loader = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)

    model = Transformer(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        dropout=dropout,
        max_context_len=4, # TODO: make this a configurable parameter?
        vocab_len=dataset.ntokens,
        non_linearity=non_linearity,
        weight_noise=0, # TODO: make this configurable?
        use_positional_encoding=use_positional_encoding
    ).to(device)

    optimizer = torch.optim.AdamW(
        [{
            "params": model.embedding.parameters(),
            "lr": embedding_lr,
            "weight_decay": embedding_weight_decay,
            "eps": eps
        },
        {
            "params": list(model.decoder.parameters()) + list(model.linear.parameters()),
            "lr": decoder_lr,
            "weight_decay": decoder_weight_decay,
            "eps": eps
        }]
    )

    loss_fn = nn.CrossEntropyLoss()

    # prepare for logging
    ex.info['log_steps'] = []
    ex.info['total'] = {
        'train': {
            'loss': [],
            'accuracy': []
        },
        'val': {
            'loss': [],
            'accuracy': []
        }
    }
    for op in dataset.operators:
        ex.info[op] = {
        'train': {
            'loss': [],
            'accuracy': []
        },
        'val': {
            'loss': [],
            'accuracy': []
        }
    }

    pos = dataset.sequence_length - 1
    steps = 0
    with tqdm(total=optimization_steps) as pbar:
        for equation, answer in islice(cycle(train_loader), optimization_steps):
            
            if steps % log_freq == 0:
                eval_loss_fn = nn.CrossEntropyLoss(reduction='sum')
                with torch.no_grad():

                    # compute train metrics
                    train_evaluation_loader = torch.utils.data.DataLoader(train, batch_size=min(500, len(train)), shuffle=False)
                    ops_losses = defaultdict(float)
                    ops_accuracies = defaultdict(int)
                    ops_totals = defaultdict(int)
                    for e_e, e_a in train_evaluation_loader:
                        e_e = e_e.to(device)
                        e_a = e_a.to(device)
                        logits, _, _ = model(e_e, pos=pos)
                        for i in range(e_e.shape[0]):
                            if only_input_tokens:
                                op = dataset.operators[0]
                            else:
                                op = dataset.operation_from_token(e_e[i][1])
                            ops_losses[op] += eval_loss_fn(logits[i:i+1], e_a[i:i+1]).item()
                            predicted_token = torch.argmax(logits[i:i+1]).item()
                            ops_accuracies[op] += int(predicted_token == e_a[i:i+1].item())
                            ops_totals[op] += 1
                    for op in dataset.operators:
                        ex.info[op]['train']['loss'].append(ops_losses[op] / ops_totals[op])
                        ex.info[op]['train']['accuracy'].append(ops_accuracies[op] / ops_totals[op])
                    ex.info['total']['train']['loss'].append(sum(ops_losses.values()) / sum(ops_totals.values()))
                    ex.info['total']['train']['accuracy'].append(sum(ops_accuracies.values()) / sum(ops_totals.values()))
                    
                    # compute test metrics
                    val_evaluation_loader = torch.utils.data.DataLoader(val, batch_size=min(500, len(val)), shuffle=False)
                    ops_losses = defaultdict(float)
                    ops_accuracies = defaultdict(int)
                    ops_totals = defaultdict(int)
                    for e_e, e_a in val_evaluation_loader:
                        e_e = e_e.to(device)
                        e_a = e_a.to(device)
                        logits, _, _ = model(e_e, pos=pos)
                        for i in range(e_e.shape[0]):
                            if only_input_tokens:
                                op = dataset.operators[0]
                            else:
                                op = dataset.operation_from_token(e_e[i][1])
                            ops_losses[op] += eval_loss_fn(logits[i:i+1], e_a[i:i+1]).item()
                            predicted_token = torch.argmax(logits[i:i+1]).item()
                            ops_accuracies[op] += int(predicted_token == e_a[i:i+1].item())
                            ops_totals[op] += 1
                    for op in dataset.operators:
                        ex.info[op]['val']['loss'].append(ops_losses[op] / ops_totals[op])
                        ex.info[op]['val']['accuracy'].append(ops_accuracies[op] / ops_totals[op])
                    ex.info['total']['val']['loss'].append(sum(ops_losses.values()) / sum(ops_totals.values()))
                    ex.info['total']['val']['accuracy'].append(sum(ops_accuracies.values()) / sum(ops_totals.values()))
                ex.info['log_steps'].append(steps)
                pbar.set_description("{0:2.1f}% | {1:2.1f}%".format(ex.info['total']['train']['accuracy'][-1] * 100, ex.info['total']['val']['accuracy'][-1] * 100))

            if embeddings_save_freq and steps % embeddings_save_freq == 0:
                with torch.no_grad():
                    torch.save(model.embedding.weight, f"/tmp/embd_{steps}.pt")
                    ex.add_artifact(f"/tmp/embd_{steps}.pt")
            optimizer.zero_grad()
            equation = equation.to(device)
            answer = answer.to(device)
            logits, _, _ = model(equation, pos=pos)
            loss = loss_fn(logits, answer)
            loss.backward()
            optimizer.step()
            steps += 1
            pbar.update(1)

