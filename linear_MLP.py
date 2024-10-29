from torch import nn


class MLP(nn.Module):

    def __init__(self,
                 input_dim=284,
                 hidden_dim=512,
                 output_dim=283,
                 dropout=0.1):
        super(MLP, self).__init__()
        self.mlplayer = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(hidden_dim, output_dim),
                                      nn.SiLU())

    def forward(self, x):
        return self.mlplayer(x)


if __name__ == "__main__":
    from utility import train_predict2
    train_predict2(MLP,
                   modelname="LR_v2.0",
                   batch_size=256,
                   train_epochs=10000)
