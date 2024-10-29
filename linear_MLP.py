from torch import nn


class MLP(nn.Module):

    def __init__(self,
                 input_dim=284,
                 hidden_dim=512,
                 output_dim=283,
                 dropout=0.1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


if __name__ == "__main__":
    from utility import train_predict2
    train_predict2(MLP,
                   modelname="LR_v1.0",
                   batch_size=256,
                   train_epchos=20000)
