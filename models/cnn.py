import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    def __init__(self, params):
        super(CNN, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )
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
        x = self.convolution(x)
        x = x.view(-1, 256 * 4 * 4)
        x = self.classifier(x)
        return x