import torch.nn as nn
from utils.utils import *


class GlobalModelForKUHAR(nn.Module):
    def __init__(self, args):
        super(GlobalModelForKUHAR, self).__init__()
        self.args = args
        fusion_input_dim = 30 * self.args.client_num if self.args.aggregation == "concat" else 30
        self.linear = nn.Linear(fusion_input_dim, 30)
        self.classifier = nn.Linear(30, 18)

    def forward(self, input_list):
        if self.args.aggregation == 'sum':
            tensor_t = torch.stack(input_list).sum(dim=0)
        elif self.args.aggregation == 'mean':
            tensor_t = torch.stack(input_list).mean(dim=0)
        elif self.args.aggregation == 'concat':
            tensor_t = torch.cat(input_list, dim=1)

        # forward
        x = tensor_t
        x = self.linear(x)
        x = self.classifier(x)
        return x


class LocalModelForKUHAR(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForKUHAR, self).__init__()
        self.args = args
        if self.args.client_num == 2:
            if client_number == 0:
                self.backbone = nn.Sequential(
                    nn.Linear(900, 300),
                    nn.ReLU(),
                    nn.Linear(300, 100),
                    nn.ReLU(),
                    nn.Linear(100, 30),
                    nn.ReLU()
                )
            else:
                self.backbone = nn.Sequential(
                    nn.Linear(900, 300),
                    nn.ReLU(),
                    nn.Linear(300, 100),
                    nn.ReLU(),
                    nn.Linear(100, 30),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.backbone(x)
        return x
