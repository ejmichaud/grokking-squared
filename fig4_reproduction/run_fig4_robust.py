import torch
torch.set_num_threads(1)
from multiprocessing import get_context
from produce_figure_data import train_many_fig4, _make_pairwise_combs
train_nums = [20, 35, 45]
seeds = [0, 1, 2, 3, 4]
if __name__ == '__main__':
    params = _make_pairwise_combs(train_nums, seeds)
    print(f"TOTAL {len(params)}", flush=True)
    print(params, flush=True)
    with get_context('spawn').Pool(4) as p:
        p.map(train_many_fig4, params)
    print("DONE", flush=True)
