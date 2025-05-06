import torch.nn as nn
from utils.utils import *


class GlobalModelForNUSWIDE(nn.Module):
    def __init__(self, args):
        super(GlobalModelForNUSWIDE, self).__init__()
        self.linear1 = nn.Linear(100, 50)
        self.classifier = nn.Linear(50, 5)
        self.args = args

    def forward(self, input_list):
        tensor_t = torch.cat((input_list[0], input_list[1]), dim=1)

        # multiple clients
        if len(input_list) > 2:
            for i in range(len(input_list) - 2):
                tensor_t = torch.cat((tensor_t, input_list[i + 2]), dim=1)
                if i + 3 == len(input_list):
                    break

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.classifier(x)
        return x


class LocalModelForNUSWIDE(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForNUSWIDE, self).__init__()
        self.args = args
        # if self.args.client_num > 2:
        #     raise_split_exception()
        backbone_I = nn.Sequential(
                    nn.Linear(634, 320),
                    nn.ReLU(),
                    nn.Linear(320, 80),
                    nn.ReLU(),
                    nn.Linear(80, 40),
                    nn.ReLU()
                )
        backbone_T = self.backbone = nn.Sequential(
                    nn.Linear(1000, 500),
                    nn.ReLU(),
                    nn.Linear(500, 125),
                    nn.ReLU(),
                    nn.Linear(125, 60),
                    nn.ReLU()
                )
        if client_number == 0:
            self.backbone = backbone_I
        else:
            self.backbone = backbone_T

    def forward(self, x):
        x = self.backbone(x)
        return x
