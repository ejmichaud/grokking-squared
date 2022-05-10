import numpy as np

class Ranges:
  weight_decay = list(np.exp(np.linspace(-4, 3, 19)))
  #weight_decay = list(np.exp(np.linspace(-4, 2, 19)))
  dropout = np.linspace(0, .5, 10)
  decoder_lr = list(10**np.linspace(-9, -1, 15))
  #decoder_lr = list(10**np.linspace(-6, -3, 12))
  esam_rho = list(np.exp(np.linspace(-4, 1, 9)))
  seed = [1,2,3]

class Locations:
  weight_decay = "runs/weight_decay_{wd}_seed_{seed}"
  dropout = "runs/dropout_{do}_seed_{seed}"
  slow_decoder = "runs/decoder_lr_{lr}_seed_{seed}"
  esam = "runs/esam_rho_{rho}_beta_{beta}_seed_{seed}"
  weight_decay_vs_decoder_lr = "runs/phaseplot_weight_decay_{wd}_decoder_lr_{lr}_seed_{seed}"
