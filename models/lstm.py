import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, params):
        super(LSTM, self).__init__()
        self.params = params
        self.hidden_size = 128
        self.num_layers = 2
        self.input_size = 32
        self.output_size = 10
        self.lstm = nn.LSTM(32, 128, 2, batch_first=True)
        self.fc = nn.Linear(128, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if params.t_init == 'xavier-normal':
                    nn.init.xavier_normal_(m.weight)    # Xavier init
                elif params.t_init == 'xavier-uniform':
                    nn.init.xavier_uniform_(m.weight)   # Xavier init
                elif params.t_init == 'he-normal':
                    nn.init.kaiming_normal_(m.weight)  # He init
                elif params.t_init == 'he-uniform':
                    nn.init.kaiming_uniform_(m.weight)  # He init
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if params.t_init == 'xavier-normal':
                    nn.init.xavier_normal_(m.weight)  # Xavier init
                elif params.t_init == 'xavier-uniform':
                    nn.init.xavier_uniform_(m.weight)  # Xavier init
                elif params.t_init == 'he-normal':
                    nn.init.kaiming_normal_(m.weight)  # He init
                elif params.t_init == 'he-uniform':
                    nn.init.kaiming_uniform_(m.weight)  # He init
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.reshape(-1, self.input_size, self.input_size)
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.params.t_gpu_no)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.params.t_gpu_no)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out