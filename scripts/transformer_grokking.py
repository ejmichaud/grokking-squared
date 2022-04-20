import torch
from models import Decoder
from config import args
from data_gen import generate_data
from tqdm import tqdm
from torchmetrics.functional import accuracy
from utils import Logger, get_loss
from utils.modDivide import modDivide

# args is the configuration environment. The defaults are in config.py

pairs, train_indices, test_indices, _ = generate_data(args.p, args.seed, args.split_ratio,
                                                      ignore_symmetric=False,
                                                      batch_size=args.batch_size)
# modular division
# nums = torch.arange(args.p) + 1
# Y = torch.from_numpy(modDivide(nums, nums.view(-1, 1), args.m))
# division
# nums = torch.arange(1, args.p + 1)
# Y = nums.div(nums.view(-1, 1))
# addition
nums = torch.arange(args.p)
Y = nums + nums.view(-1, 1)
Y = Y % args.m if args.m > 1 else Y
Y = Y.long().to(args.device)


representation = torch.randn(args.p, args.latent_dim).to(args.device).requires_grad_()
out_classes = args.m if args.m > 1 else 2 * args.p - 1
model = Decoder(input_dim=args.latent_dim, output_dim=out_classes,
                w=args.decoder_width,
                concat=True, dropout=args.dropout).to(args.device)


param_groups = [{"params": (representation, ), "lr": args.lr_rep},
                {"params": model.parameters(), "lr": args.lr_decoder,
                "weight_decay": args.weight_decay}]

optimizer = torch.optim.AdamW(param_groups)
loss_func = get_loss(args.loss)

print(args)


def step(idx):
    x = representation[idx]
    pred = model(x)
    target = Y[idx[:, 0], idx[:, 1]]
    if loss_func == torch.nn.functional.mse_loss:
        # pred = pred.softmax(1)
        loss = loss_func(pred, torch.nn.functional.one_hot(target, out_classes).float())
    else:
        loss = loss_func(pred, target)
    acc = accuracy(pred, target)
    return loss, acc


pbar = tqdm(range(args.epochs))
logger = Logger(args, experiment=f"modular-{args.m}-{args.p}")

for epoch in pbar:
    model.train()
    metrics = {}
    for idx, *_ in train_indices:
        optimizer.zero_grad()
        loss_train, acc_train = step(idx)
        loss_train.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        for idx, *_ in test_indices:
            loss_test, acc_test = step(idx)

    # logging
    msg = f"Loss {loss_train.item():.2e}|{loss_test.item():.2e} || "
    msg += f"Acc {acc_train:.3f}|{acc_test:.3f}"
    pbar.set_description(msg)

    # Logging metrics and embeddings
    metrics = {"loss/train": loss_train.item(), "loss/test": loss_test.item(),
               "acc/train": acc_train, "acc/test": acc_test}
    logger.log(metrics, weights=representation.data)

    # Plotting embeddings
    if args.plot > 0:
        if epoch % args.plot == 0:
            logger.plot_embedding(representation.detach(), metrics, epoch)
        if epoch == args.epochs - 1:
            logger.plot_embedding(representation.detach(), metrics, epoch)
            logger.save_anim("bruv2")
logger.close()
