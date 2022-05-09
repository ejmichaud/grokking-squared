from argparse import ArgumentParser

parser = ArgumentParser(description='Grokking Squared')
parser.add_argument('--batch_size', type=int, default=-1, metavar='N')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
parser.add_argument('--lr_decoder', type=float, default=1e-3, metavar='LR')
parser.add_argument('--lr_rep', type=float, default=1e-3, metavar='LR')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--rep_noise', action='store_true', default=False,)
parser.add_argument('--tune', type=int, default=0)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--log', type=int, default=0)
parser.add_argument('--save_weights', action='store_true', default=False,)
parser.add_argument('--save_ckpt', action='store_true', default=False,)
parser.add_argument('--plot', type=int, default=0)
parser.add_argument('--perfect_rep', action='store_true', default=False,)
parser.add_argument('--seed', type=int, default=1, metavar='SEED')
parser.add_argument('--p', type=int, default=53, help='number of datapoints')
parser.add_argument('--m', type=int, default=53, help='modulus')
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help="train/test split ratio")
parser.add_argument('--latent_dim', type=int, default=256, metavar='N',
                    help='latent dimension')
parser.add_argument('--decoder_width', type=int, default=256, metavar='N',
                    help='decoder width')
parser.add_argument('--decoder_depth', type=int, default=2, metavar='N',
                    help='decoder depth')
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--loss', type=str, default='cross_entropy')
parser.add_argument('--stop_early', action='store_true', default=False)


# args, unknown = parser.parse_known_args()
args = parser.parse_args()
