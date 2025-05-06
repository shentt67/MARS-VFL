import torch.nn as nn
from .common_models import GRU, MLP, Concat


class GlobalModelForMOSI(nn.Module):
    def __init__(self, args):
        super(GlobalModelForMOSI, self).__init__()
        self.args = args
        if args.aggregation == 'concat':
            fusion = Concat()
            self.classifier = MLP(870, 870, 1)
        self.fusion = fusion
        self.args = args

    def forward(self, input_list):
        output_fusion = self.fusion(input_list)
        output = self.classifier(output_fusion)
        return output


class LocalModelForMOSI(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForMOSI, self).__init__()
        self.args = args
        backbone_V = GRU(35, 70, dropout=True, has_padding=True, batch_first=True)
        backbone_A = GRU(74, 200, dropout=True, has_padding=True, batch_first=True)
        backbone_T = GRU(300, 600, dropout=True, has_padding=True, batch_first=True)

        if client_number == 0:
            self.backbone = backbone_V
        elif client_number == 1:
            self.backbone = backbone_A
        else:
            self.backbone = backbone_T

    def forward(self, x):
        x = self.backbone(x)
        return x
