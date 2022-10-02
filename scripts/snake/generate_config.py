import os
from const import project_dir
from itertools import product

EXPERIMENT_NAME = 'modular-addition-53-53'
DEVICE = 'cuda:0'
hparams_defaults = dict(batch_size=-1, epochs=20000, lr_decoder=0.0002, lr_rep=0.001, dropout=0.05, weight_decay=1.0, rep_noise=False, tune=False, log=20, save_weights=True,
                        plot=0, perfect_rep=False, seed=1, p=59, m=59, split_ratio=0.8, latent_dim=256, decoder_width=128, exp_name=EXPERIMENT_NAME, device=DEVICE, loss='cross_entropy')


decoder_learning_rates = [0.001, 0.01, 0.1, 1.0]
rep_learning_rates = [0.001, 0.01, 0.1, 1.0]
seeds = [1, 2]

for seed, rep_lr, dec_lr in product(seeds, decoder_learning_rates, rep_learning_rates):
    hparams = hparams_defaults.copy()
    hparams['seed'] = seed
    hparams['lr_decoder'] = dec_lr
    hparams['lr_rep'] = rep_lr

    # Create config file
    with open(os.path.join(project_dir, 'configs', EXPERIMENT_NAME + '.yaml'), 'w') as f:
        f.write(hparams)
