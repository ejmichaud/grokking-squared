import torch
torch.set_num_threads(1)
from multiprocessing import get_context
import numpy as np
import os
from train_add import train_add
from produce_figure_data import _make_pairwise_combs, HiddenPrints

fig4eta3_path = "./results/fig4_eta3/"
os.makedirs(fig4eta3_path, exist_ok=True)
train_nums = [20, 35, 45]
seeds = [0,1,2,3,4]

def train_many_fig4_eta3(params):
    train_num=int(params[0]); seed=int(params[1])
    print(f"train_num={train_num}, seed={seed}", flush=True)
    with HiddenPrints():
        dic=train_add(train_num=train_num, seed=seed, steps=int(1e4), eff_steps=1, eta_reprs=3e-3)
    np.savetxt(fig4eta3_path+f"rqistep_num_{train_num}_seed_{seed}.txt", np.array([dic["iter_rqi"]]))

if __name__ == '__main__':
    params=_make_pairwise_combs(train_nums, seeds)
    print(f"TOTAL {len(params)}", flush=True)
    with get_context('spawn').Pool(4) as p:
        p.map(train_many_fig4_eta3, params)
    print("DONE", flush=True)
