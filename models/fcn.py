import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, class_num):
        super(FCN, self).__init__()
        self.n_cls = class_num
        self.fc1 = nn.Linear(1 * 28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, self.n_cls)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
