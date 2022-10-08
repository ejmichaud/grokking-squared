import numpy as np
from multiprocessing import get_context
import os
import sys

from train_add import train_add

path3 = "./results/fig3/"
path4 = "./results/fig4/"

train_nums = [5,10,15,20,25,30,35,40,45,50,54]
seeds = [0,1,2]

for p in [path3, path4]:
  isExist = os.path.exists(p)
  if not isExist:
      os.makedirs(p)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, _, __, ___):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def train_many_fig3(params):
    train_num = int(params[0])
    seed = int(params[1])
    print("train_num={}, seed={}".format(train_num, seed))
    with HiddenPrints():
        dic = train_add(train_num=train_num, seed=seed, steps=5000, eff_steps=1)
    np.savetxt(path3+"trainfrac_num_%d_seed_%d.txt"%(train_num, seed), np.array([dic["train_ratio"]]))
    np.savetxt(path3+"predacc_num_%d_seed_%d.txt"%(train_num, seed), np.array([dic["pred_acc"]]))
    np.savetxt(path3+"realacc_num_%d_seed_%d.txt"%(train_num, seed), np.array([dic["acc"][-1]]))

def train_many_fig4(params):
    train_num = int(params[0])
    seed = int(params[1])
    print("train_num={}, seed={}".format(train_num, seed))
    with HiddenPrints():
        dic = train_add(train_num=train_num, seed=seed, steps=int(1e4), eff_steps=1)
    np.savetxt(path4+"rqistep_num_%d_seed_%d.txt"%(train_num, seed), np.array([dic["iter_rqi"]]))

xx, yy = np.meshgrid(train_nums, seeds)
params = list(np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,)])))



if __name__ == '__main__':
    ctx = get_context('spawn')
    with ctx.Pool(2) as p: # adjust to your needs
        p.map(train_many_fig3, params)
        p.map(train_many_fig4, params)
