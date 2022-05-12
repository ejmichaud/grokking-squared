import os
import matplotlib.pyplot as plt
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.style.use(os.path.join(root, "scripts", "grok.mplstyle"))
import numpy as np
import glob
import sys
from phasegrok.utils.utils import read_scalars
from phasegrok.definitions import Ranges, Locations

if len(sys.argv) != 2:
    print("Usage: python plot_generalization_time.py [weight_decay|dropout|esam_rho|decoder_lr]")
    sys.exit(1)


if "weight_decay" == sys.argv[1]:
  paths = [Locations.weight_decay.format(wd=wd, seed=seed) for wd in Ranges.weight_decay for seed in Ranges.seed]
  Xs = [wd for wd in Ranges.weight_decay for _ in Ranges.seed]
elif "dropout" == sys.argv[1]:
  paths = [Locations.dropout.format(do=do, seed=seed) for do in Ranges.dropout for seed in Ranges.seed]
  Xs = [do for do in Ranges.dropout for _ in Ranges.seed]
elif "decoder_lr" == sys.argv[1]:
  paths = [Locations.slow_decoder.format(lr=lr, seed=seed) for lr in Ranges.decoder_lr for seed in Ranges.seed]
  Xs = [lr for lr in Ranges.decoder_lr for _ in Ranges.seed]
else:
  raise ValueError(f"Unknown parameter {sys.argv[1]}")


train_ttgs = []
test_ttgs = []
diff_ttgs = []

for path in paths:
  data = read_scalars(glob.glob(os.path.join(root, path, "*/events*.0"))[0])
  test_accs = np.array(data["acc/test"])
  train_accs = np.array(data["acc/train"])
  test_ttg = int(test_accs[np.argmax(test_accs[:,1] > .99),0])
  train_ttg = int(train_accs[np.argmax(train_accs[:,1] > .99),0])
  if test_ttg == 0:
    test_ttg = data["acc/test"][-1][0] # last step
  if train_ttg == 0:
    train_ttg = test_ttg
  if train_ttg == test_ttg == data["acc/test"][-1][0]:
    diff_ttg = test_ttg
  else:
    diff_ttg = max(1, test_ttg - train_ttg + 1)
  test_ttgs.append(test_ttg)
  train_ttgs.append(train_ttg)
  diff_ttgs.append(diff_ttg)

test_ttgs_median = np.median(np.array(test_ttgs).reshape(-1, len(Ranges.seed)), axis=1)
train_ttgs_median = np.median(np.array(train_ttgs).reshape(-1, len(Ranges.seed)), axis=1)
diff_ttgs_median = np.median(np.array(diff_ttgs).reshape(-1, len(Ranges.seed)), axis=1)

plt.scatter(Xs, test_ttgs, label="test set", alpha=.4)
plt.scatter(Xs, train_ttgs, label="training set", alpha=.4)
plt.scatter(Xs, diff_ttgs, label="test - training", alpha=.4)

plt.plot(Xs[::3], test_ttgs_median, alpha=.6)
plt.plot(Xs[::3], train_ttgs_median, alpha=.6)
plt.plot(Xs[::3], diff_ttgs_median, alpha=.6)

plt.ylabel('epochs to 99\% accuracy')

if "weight_decay" == sys.argv[1]:
  plt.xlabel('decoder weight decay')
  plt.loglog()
elif "dropout" == sys.argv[1]:
  plt.xlabel('dropout rate')
  plt.semilogy()
elif "esam_rho" == sys.argv[1]:
  plt.xlabel(r'ESAM $\rho$ parameter')
  plt.loglog()
elif "decoder_lr" == sys.argv[1]:
  plt.xlabel('decoder learning rate')  
  plt.loglog()
  
if "dropout" == sys.argv[1]:
  plt.legend(loc="upper center")
else:
  plt.legend(loc="best")

plt.tight_layout()
#plt.show()
plt.savefig(f"generalization_time_{sys.argv[1]}.pdf")
