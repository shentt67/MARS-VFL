import torch.nn as nn
from utils.utils import *


class GlobalModelForUCIHAR(nn.Module):
    def __init__(self, args):
        super(GlobalModelForUCIHAR, self).__init__()
        self.args = args
        fusion_input_dim = 16 * self.args.client_num if self.args.aggregation == "concat" else 16
        self.linear = nn.Linear(fusion_input_dim, 16)
        self.classifier = nn.Linear(16, 6)

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


class LocalModelForUCIHAR(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForUCIHAR, self).__init__()
        self.args = args
        if self.args.client_num == 2:
            if client_number == 0:
                self.backbone = nn.Sequential(
                    nn.Linear(348, 140),
                    nn.ReLU(),
                    nn.Linear(140, 70),
                    nn.ReLU(),
                    nn.Linear(70, 16),
                    nn.ReLU()
                )
            else:
                self.backbone = nn.Sequential(
                    nn.Linear(213, 140),
                    nn.ReLU(),
                    nn.Linear(140, 70),
                    nn.ReLU(),
                    nn.Linear(70, 16),
                    nn.ReLU()
                )

    def forward(self, x):
        x = self.backbone(x)
        return x
