import torch.nn as nn
from .common_models import GRU, MLP, Concat


class GlobalModelForPTBXL(nn.Module):
    def __init__(self, args):
        super(GlobalModelForPTBXL, self).__init__()
        self.args = args
        if args.aggregation == 'concat':
            fusion = Concat()
            self.classifier = MLP(30010, 100, 5, dropout=False)
        self.fusion = fusion
        self.args = args

    def forward(self, input_list):
        output_fusion = self.fusion(input_list)
        output = self.classifier(output_fusion)
        return output


class LocalModelForPTBXL(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForPTBXL, self).__init__()
        self.args = args
        backbone_T = MLP(4, 10, 10, dropout=False)
        backbone_S_L = GRU(6, 15, dropout=False, batch_first=True)
        backbone_S_C = GRU(6, 15, dropout=False, batch_first=True)

        if client_number == 0:
            self.backbone = backbone_T
        elif client_number == 1:
            self.backbone = backbone_S_L
        elif client_number == 2:
            self.backbone = backbone_S_C

    def forward(self, x):
        x = self.backbone(x)
        return x
