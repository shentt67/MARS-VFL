import torch
import torch.nn as nn
from .common_models import GRU, MLP, Concat


class GlobalModelForMUSTARD(nn.Module):
    def __init__(self, args):
        super(GlobalModelForMUSTARD, self).__init__()
        self.args = args
        if args.aggregation == 'concat':
            fusion = Concat()
            self.classifier = MLP(1460, 1460, 2)
        self.fusion = fusion
        self.args = args

    def forward(self, input_list):
        output_fusion = self.fusion(input_list)
        output = self.classifier(output_fusion)
        return output


class LocalModelForMUSTARD(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForMUSTARD, self).__init__()
        self.args = args
        backbone_V = GRU(371, 700, dropout=True, has_padding=True, batch_first=True)
        backbone_A = GRU(81, 160, dropout=True, has_padding=True, batch_first=True)
        backbone_T = GRU(300, 600, dropout=True, has_padding=True, batch_first=True)
        if client_number == 0:
            self.backbone = backbone_V
            self.out_features = 371
        elif client_number == 1:
            self.backbone = backbone_A
            self.out_features = 81
        else:
            self.backbone = backbone_T
            self.out_features = 300      

    def forward(self, x):
        x = self.backbone(x)
        return x