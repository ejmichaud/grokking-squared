from torch import nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU(),
                 bn=False, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.bn = nn.BatchNorm1d(output_dim) if bn else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.activation(out)
        if self.bn is not None:
            out = self.bn(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out + residual
        return out


class DecoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1) -> None:
        dim_feedforward = dim_feedforward if dim_feedforward is not None else d_model
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation="relu", batch_first=True, norm_first=True)


class ReshapeBlock(nn.Module):
    def __init__(self, shape="half"):
        super(ReshapeBlock, self).__init__()
        self.shape = shape

    def forward(self, x):
        if x.dim() == 2:
            return x.view(x.shape[0], 2, -1)
        elif x.dim() == 3:
            return x.view(x.shape[0], -1)
        else:
            raise ValueError("Invalid input dimension in ReshapeBlock")


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, w=200, concat=False,
                 dropout=0.0, depth=1):
        super(Decoder, self).__init__()
        self.concat = concat
        if self.concat:
            self.net = nn.Sequential(*(DecoderLayer(input_dim, 1, w, dropout=dropout) for _ in range(depth)),
                                     ReshapeBlock(),
                                     nn.Linear(2 * input_dim, output_dim))
        else:
            # input_dim = 2 * input_dim if concat else input_dim
            self.net = nn.Sequential(nn.Linear(input_dim, w),
                                     nn.SELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(w, w),
                                     #   nn.LeakyReLU(),
                                     #  ResidualBlock(w, w),
                                     #  nn.Dropout(dropout),
                                     #  ResidualBlock(w, w),
                                     #  nn.Dropout(dropout),
                                     nn.Linear(w, output_dim))

    def forward(self, x):
        if not self.concat:
            x = x.sum(-2)
        return self.net(x)


class NET(nn.Module):  # base MLP model
    def __init__(self, input_dim, output_dim, w=200):
        super(NET, self).__init__()
        self. model = nn.Sequential(nn.Linear(input_dim, w), nn.ReLU(),
                                    nn.Linear(w, w), nn.ReLU(),
                                    nn.Linear(w, output_dim))

    def forward(self, x):
        return self.model(x)


class DEC(nn.Module):  # Decoder
    def __init__(self, input_dim, output_dim, w=200, concat=False, dropout=0.0):
        super(DEC, self).__init__()
        self.concat = concat
        if self.concat:
            self.net = nn.Sequential(ReshapeBlock(),
                                     DecoderLayer(input_dim, 2, w),
                                     DecoderLayer(input_dim, 2, w),
                                     ReshapeBlock(),
                                     nn.Linear(2 * input_dim, output_dim))
        else:
            # input_dim = 2 * input_dim if concat else input_dim
            self.net = nn.Sequential(nn.Linear(input_dim, w),
                                     nn.ReLU(),
                                     nn.Linear(w, w),
                                     nn.ReLU(),
                                     ResidualBlock(w, w, dropout=dropout),
                                     ResidualBlock(w, w, dropout=dropout),
                                     nn.Linear(w, output_dim))

    def _parse_input(self, x, x_id):
        if self.concat:
            return torch.cat([x[x_id[:, 0]], x[x_id[:, 1]]], dim=1)
        else:
            return x[x_id[:, 0]] + x[x_id[:, 1]]
        # self.net = NET(input_dim, output_dim, w=w)

    def forward(self, latent, idx):
        add = self._parse_input(latent, idx)
        return self.net(add)


class AE(nn.Module):
    def __init__(self, enc_w=200, dec_w=200, input_dim=1, output_dim=1, latent_dim=1,
                 concat_decoder=False, dropout=0.0):
        super(AE, self).__init__()
        self.enc = NET(input_dim, latent_dim, w=enc_w)
        self.dec = DEC(latent_dim, output_dim, w=dec_w,
                       concat=concat_decoder, dropout=dropout)

    def forward(self, x, x_id):
        self.latent = self.enc(x)
        self.out = self.dec(self.latent, x_id)

        return self.out
