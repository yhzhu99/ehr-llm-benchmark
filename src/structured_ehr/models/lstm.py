from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x, **kwargs):
        y, _ = self.lstm(x)
        y = y[:, -1, :]  # Get the last time step output
        return y