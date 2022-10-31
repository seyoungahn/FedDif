import torch.nn as nn

class Logistic(nn.Module):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x