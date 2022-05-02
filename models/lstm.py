import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_size = 128
        self.num_layers = 2
        self.input_size = 32
        self.output_size = 10
        self.lstm = nn.LSTM(32, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.reshape(-1, self.input_size, self.input_size)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out