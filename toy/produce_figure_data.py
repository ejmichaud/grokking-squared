import numpy as np
from multiprocessing import get_context
import os
import sys

from train_add import train_add

path3 = "./results/fig3/"
path4 = "./results/fig4/"
path5a = "./results/fig5a/"
path5b = "./results/fig5b/"
path5c = "./results/fig5c/"

for p in [path3, path4, path5a, path5b, path5c]:
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


def train_many_fig5a(params):
    eta_reprs = params[0]
    eta_dec = params[1]
    print("eta_reprs=%.5f, eta_dec=%.5f"%(eta_reprs, eta_dec))
    with HiddenPrints():
        dic = train_add(eta_reprs=eta_reprs, eta_dec=eta_dec, steps=int(1e4), eff_steps=1)
        np.savetxt(path5a+"trainstep_eta1_%.5f_eta2_%.5f.txt"%(eta_reprs, eta_dec), np.array([dic["iter_train"]]))
        np.savetxt(path5a+"teststep_eta1_%.5f_eta2_%.5f.txt"%(eta_reprs, eta_dec), np.array([dic["iter_test"]]))

def train_many_fig5b(params):
    eta_dec = params[0]
    weight_decay_dec = params[1]
    #print(eta_dec, weight_decay_dec)
    print("eta_dec=%.5f, weight_decay=%.5f"%(eta_dec, weight_decay_dec))
    with HiddenPrints():
        dic = train_add(weight_decay_dec=weight_decay_dec, eta_dec=eta_dec, steps=int(1e4), eff_steps=1)
        np.savetxt(path5b+"trainstep_eta_%.5f_wd_%.5f.txt"%(eta_dec, weight_decay_dec), np.array([dic["iter_train"]]))
        np.savetxt(path5b+"teststep_eta_%.5f_wd_%.5f.txt"%(eta_dec, weight_decay_dec), np.array([dic["iter_test"]]))

def train_many_fig5c(params):
    eta_dec = params[0]
    weight_decay_dec = params[1]
    #print(eta_dec, weight_decay_dec)
    print("eta_dec=%.5f, weight_decay=%.5f"%(eta_dec, weight_decay_dec))
    with HiddenPrints():
        dic = train_add(weight_decay_dec=weight_decay_dec, eta_dec=eta_dec, steps=int(1e4), eff_steps=1, loss_type="CE", seed=1)
        np.savetxt(path5c+"trainstep_eta_%.5f_wd_%.5f.txt"%(eta_dec, weight_decay_dec), np.array([dic["iter_train"]]))
        np.savetxt(path5c+"teststep_eta_%.5f_wd_%.5f.txt"%(eta_dec, weight_decay_dec), np.array([dic["iter_test"]]))

#Fig3,4
train_nums = [5,10,15,20,25,30,35,40,45,50,54]
seeds = [0,1,2]

#Fig5
eta_reprs = 10**np.linspace(-5,-2,num=10) # Fig5a
eta_decs = 10**np.linspace(-5,-2,num=10) #Fig5abc
wds = np.arange(11,) # Fig5bc

def _make_pairwise_combs(xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    return list(np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,)])))


if __name__ == '__main__':
    #multiprocessing context
    ctx = get_context('spawn')

    # Figure 3, 4
    params34 = _make_pairwise_combs(train_nums, seeds)
    params5a = _make_pairwise_combs(eta_reprs, eta_decs)
    params5bc = _make_pairwise_combs(eta_decs, wds)
    with ctx.Pool(2) as p: # adjust to your needs
        p.map(train_many_fig3, params34)
        p.map(train_many_fig4, params34)
        p.map(train_many_fig5a, params5a)
        p.map(train_many_fig5b, params5bc)
        p.map(train_many_fig5c, params5bc)
