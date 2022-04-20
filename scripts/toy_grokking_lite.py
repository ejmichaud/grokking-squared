# %%
import torch
from torch import nn
from torch import optim
from torchmetrics.functional import accuracy
from tqdm import tqdm
from data_gen import generate_data
from models import Decoder
from config import args
import optuna
from utils import Logger, get_loss

torch.manual_seed(args.seed)

pairs, train_idx, test_idx, best_acc = generate_data(
    p=args.p, seed=args.seed, split_ratio=args.split_ratio, batch_size=args.batch_size,
    ignore_symmetric=True, compute_best_acc=True)


print("Best attainable accuracy:", f"{best_acc:.2f}")
args.__dict__["ntokens"] = 2 * args.p - 1  # args.p
Y = torch.arange(args.p) + torch.arange(args.p).view(-1, 1)
if args.ntokens != 2 * args.p - 1:
    Y = Y % args.ntokens


def main():
    torch.manual_seed(args.seed)
    if args.perfect_rep:
        representation = torch.randn(1, args.latent_dim)
        representation = representation * torch.arange(args.p).view(-1, 1)
        representation = representation + torch.randn_like(representation) * 0.5
        representation.requires_grad_()
    else:
        representation = torch.randn(args.p, args.latent_dim).requires_grad_()

    model = Decoder(input_dim=args.latent_dim, output_dim=args.ntokens,
                    w=args.decoder_width, dropout=args.dropout)

    param_groups = [{"params": (representation, ), "lr": args.lr_rep,
                    #  "beta1": 0, "beta2": 1., "weight_decay": 0, "eps": 1e-8
                     },
                    {"params": model.parameters(), "lr": args.lr_decoder,
                    #  "beta1": 0, "beta2": 1., "eps": 1,
                     "weight_decay": args.weight_decay}
                    ]

    optimizer = optim.AdamW(param_groups)
    loss_func = get_loss(args.loss)

    print(args)
    # Adding is some sort of common denominator for various problems
    # %%

    def step(idx):
        x = representation[idx]
        if args.rep_noise:
            x = x + torch.randn_like(x)
        pred = model(x)
        target = Y[idx[:, 0], idx[:, 1]]
        if loss_func == nn.functional.mse_loss:
            loss = loss_func(
                pred,
                nn.functional.one_hot(target, args.ntokens).float())
            acc = accuracy(pred, target)
        else:
            loss = loss_func(pred, target)
            acc = accuracy(pred, target)
        return loss, acc

    def train(idx):
        model.train()
        optimizer.zero_grad()
        loss, acc = step(idx)
        loss.backward()
        optimizer.step()
        return loss.item(), acc

    def test(idx):
        model.eval()
        with torch.no_grad():
            loss, acc = step(idx)
        return loss.item(), acc

    topbar = tqdm(bar_format='{desc}{postfix}')
    topbar.set_description("Train | Test")
    pbar = tqdm(total=args.epochs)

    logger = Logger(args, experiment="toy_grokking_lite")

    def next_step():
        for indices, *_ in train_idx:
            train_loss, train_acc = train(indices)
        for indices, *_ in test_idx:
            test_loss, test_acc = test(indices)
        pbar.set_description(
            f"Loss {train_loss:.2e}|{test_loss:.2e} Acc {train_acc:.2f}|{test_acc:.2f}")
        pbar.update()

        metrics = {"loss/train": train_loss, "loss/test": test_loss,
                   "acc/train": train_acc, "acc/test": test_acc, }
        logger.log(metrics,)
        return test_loss, test_acc, train_loss, train_acc
    # %%
    # if args.plot:
    #     for i in range(args.epochs):
    #         test_loss, test_acc, train_loss, train_acc = next_step()
    #         metrics = {"loss/train": train_loss, "loss/test": test_loss,
    #                    "acc/train": train_acc, "acc/test": test_acc, }
    #         if i % args.plot == 0:
    #             logger.plot_embedding(representation.detach(), metrics=metrics, epoch=i)
    #     logger.save_anim("bruv")

    for i in range(args.epochs):
        test_loss, test_acc, *_ = next_step()
        if test_acc >= best_acc:
            print("Best test achieved acc:", test_acc)
            break
    return test_loss, test_acc


if __name__ == "__main__":
    if args.tune:
        def objective(trial):
            args.__dict__["lr_rep"] = trial.suggest_loguniform("lr_rep", 1e-5, 1e-1)
            args.__dict__["lr_decoder"] = trial.suggest_loguniform(
                "lr_decoder", 1e-4, 1e-1)
            args.__dict__["weight_decay"] = trial.suggest_loguniform(
                "weight_decay", 1e-5, 0.1)
            args.__dict__["dropout"] = trial.suggest_loguniform("dropout", 1e-3, 1e-1)
            args.__dict__["decoder_width"] = trial.suggest_int("decoder_width", 10, 100)
            test_loss, test_acc, *_ = main()
            return test_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        print("_________________________________________" * 2)
        print("Done! Now training:")
        print(study.best_params)
        for k, v in study.best_params.items():
            args.__dict__[k] = v
        args.__dict__["epochs"] = 100000
        args.__dict__["no-log"] = False
        main()
    else:

        main()
