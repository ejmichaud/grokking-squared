import torch
from phasegrok.models import Decoder
from phasegrok.config import args
from phasegrok.data_gen import generate_data
from tqdm import tqdm
from torchmetrics.functional import accuracy
from phasegrok.utils import Logger, get_loss
from phasegrok.utils.modDivide import modDivide
import os
import optuna 

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

def main():
    torch.manual_seed(args.seed)
    representation = torch.randn(args.p, args.latent_dim).to(
        args.device).requires_grad_()
    out_classes = args.m if args.m > 1 else 2 * args.p - 1
    model = Decoder(input_dim=args.latent_dim, output_dim=out_classes,
                    w=args.decoder_width, depth=args.decoder_depth,
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
            loss = loss_func(pred, torch.nn.functional.one_hot(
                target, out_classes).float())
        else:
            loss = loss_func(pred, target)
        acc = accuracy(pred, target)
        return loss, acc


    pbar = tqdm(range(args.epochs))
    logger = Logger(args, experiment=f"{args.exp_name}", timestamp=True)
        
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
        logger.log(metrics, weights=representation.data, ckpt=model.state_dict())
        # if epoch == 0:
        #     if args.save_ckpt:
        #         torch.save(model.cpu(), os.path.join(logger.log_path, "model.pt"))
        #         model.to(args.device)

        # Plotting embeddings
        if args.plot > 0:
            if epoch % args.plot == 0:
                logger.plot_embedding(representation.detach(), metrics, epoch)
            if epoch == args.epochs - 1:
                logger.plot_embedding(representation.detach(), metrics, epoch)
                logger.save_anim("bruv2")
        # stop early if the accuracy is over 99.9%
        if acc_test > 0.999:
          break

    logger.close()
    return loss_test, acc_test


if __name__ == "__main__":
    if args.tune:
        def objective(trial):
            # args.__dict__["lr_rep"] = trial.suggest_loguniform("lr_rep", 1e-5, 1e-1)
            # args.__dict__["lr_decoder"] = trial.suggest_loguniform(
            #     "lr_decoder", 1e-4, 1e-1)
            # args.__dict__["weight_decay"] = trial.suggest_loguniform(
            #     "weight_decay", 1e-5, 0.1)
            # args.__dict__["dropout"] = trial.suggest_loguniform("dropout", 1e-3, 1e-1)
            # args.__dict__["decoder_width"] = trial.suggest_int("decoder_width", 10, 100)
            args.__dict__["seed"] = trial.suggest_int("seed", 0, 1000)
            test_loss, test_acc, *_ = main()
            return test_acc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.tune)
        print("_________________________________________" * 2)
        print("Done! Now training:")
        print(study.best_params)
        # for k, v in study.best_params.items():
        #     args.__dict__[k] = v
        # args.__dict__["epochs"] = 100000
        # args.__dict__["no-log"] = False
        main()
    else:
        main()
