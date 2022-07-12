# %%
import numpy as np
from itertools import combinations_with_replacement, product
from phasegrok.utils import get_parallelograms
from torch.utils.data import DataLoader, TensorDataset
import torch


def random_split_index(total_num, train_frac=0.8):
    train_num = int(total_num * train_frac)
    train_indecies = np.random.choice(total_num, train_num, replace=False)
    test_indecies = list(set(np.arange(total_num)) - set(train_indecies))
    return train_indecies, test_indecies


def get_row(parallelogram, p):
    row = np.zeros(p)
    (i, j), (k, l) = parallelogram
    row[i] += 1
    row[j] += 1
    row[k] -= 1
    row[l] -= 1
    return tuple(row.tolist())

# Generate a table of binary operations


def augment_matrix(parallelograms_matrix, matrix):
    def _rank(x):
        return np.linalg.matrix_rank(np.array(list(x)))

    new_matrix = matrix.copy()
    for row in parallelograms_matrix:
        if row in matrix:
            continue
        tmp = new_matrix.copy()
        tmp.add(row)
        if _rank(new_matrix) == _rank(tmp):
            new_matrix.add(row)
    return new_matrix


def _compute_best_acc(pairs, train_id, test_id, p):
    parallelograms_all = get_parallelograms(pairs)
    parallelograms_train = get_parallelograms(pairs[train_id])
    # Build matrix
    parallelograms_matrix = set()
    parallelograms_matrix_train = set()
    for parallelogram in parallelograms_all:
        row = get_row(parallelogram, p)
        parallelograms_matrix.add(row)
        if parallelogram in parallelograms_train:
            parallelograms_matrix_train.add(row)

        # Get test accuracy
        count = 0
        parallelograms_matrix_train_augmented = augment_matrix(
            parallelograms_matrix, parallelograms_matrix_train)
        for pair in pairs[test_id]:
            for parallelogram in parallelograms_all:
                if tuple(pair) in parallelogram:
                    if get_row(parallelogram, p) in parallelograms_matrix_train_augmented:
                        count += 1
                        break
        test_acc = count / len(test_id)
    return test_acc


def generate_data(p=10, seed=0, split_ratio=0.8, ignore_symmetric=True, batch_size=-1,
                  compute_best_acc=False, shuffle=False):
    if ignore_symmetric:
        pairs = list(combinations_with_replacement(range(p), 2))
    else:
        pairs = list(product(range(p), range(p)))
    pairs = np.array(pairs)
    total_num = len(pairs)

    np.random.seed(seed)
    train_id, test_id = random_split_index(total_num, split_ratio)
    train_id = set(train_id)
    train_id.add(0)
    train_id.add(1)
    test_id = set(range(total_num)) - set(train_id)
    train_id = sorted(train_id)
    test_id = sorted(test_id)

    test_acc = _compute_best_acc(pairs, train_id, test_id,
                                 p) if compute_best_acc else None

    batch_size = [batch_size] * 2 if batch_size > 0 else [len(train_id), len(test_id)]

    pairs = torch.tensor(pairs)

    def _to_loader(idx):
        return DataLoader(
            TensorDataset(pairs[idx]),
            batch_size=batch_size[0],
            shuffle=shuffle)

    train_id = _to_loader(train_id)
    test_id = _to_loader(test_id)

    return pairs, train_id, test_id, test_acc
