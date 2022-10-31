import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, params):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(32*32*1, 10)
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
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
