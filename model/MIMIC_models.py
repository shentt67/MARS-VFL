import torch.nn as nn
from .common_models import GRU, MLP, Concat


class GlobalModelForMIMIC(nn.Module):
    def __init__(self, args):
        super(GlobalModelForMIMIC, self).__init__()
        self.args = args
        # fusion_input_dim = 16 * self.args.client_num if self.args.aggregation == "concat" else 16
        # self.linear = nn.Linear(fusion_input_dim, 16)
        if args.aggregation == 'concat':
            fusion = Concat()
            self.classifier = MLP(730, 40, 2, dropout=False)
        self.fusion = fusion
        self.args = args

    def forward(self, input_list):
        output_fusion = self.fusion(input_list)
        output = self.classifier(output_fusion)
        return output


class LocalModelForMIMIC(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForMIMIC, self).__init__()
        self.args = args
        backbone_T = MLP(5, 10, 10, dropout=False)
        backbone_S = GRU(12, 30, dropout=False, batch_first=True)

        if client_number == 0:
            self.backbone = backbone_T
        elif client_number == 1:
            self.backbone = backbone_S

    def forward(self, x):
        x = self.backbone(x)
        return x
