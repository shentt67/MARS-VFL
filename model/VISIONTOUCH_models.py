import torch.nn as nn
from .common_models import MLP, ImageEncoder, ForceEncoder, ProprioEncoder, DepthEncoder, ActionEncoder, Sequential2, roboticsConcat, LowRankTensorFusion


class GlobalModelForVISIONTOUCH(nn.Module):
    def __init__(self, args):
        super(GlobalModelForVISIONTOUCH, self).__init__()
        self.args = args
        self.fusion = Sequential2(roboticsConcat(
            "noconcat"), LowRankTensorFusion([256, 256, 256, 256, 32], 200, 40))
        # self.head = MLP(200, 128, 2)
        self.head = MLP(200, 128, 4)

    def forward(self, input_list):
        output_fusion = self.fusion(input_list)
        output = self.head(output_fusion)
        return output


class LocalModelForVISIONTOUCH(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForVISIONTOUCH, self).__init__()
        self.args = args

        backbone_img = ImageEncoder(128, alpha=1.0)
        backbone_fce = ForceEncoder(128, alpha=1.0)
        backbone_prio = ProprioEncoder(128, alpha=1.0)
        backbone_dep = DepthEncoder(128, alpha=1.0)
        backbone_act = ActionEncoder(4)

        if client_number == 0:
            self.backbone = backbone_img
        elif client_number == 1:
            self.backbone = backbone_fce
        elif client_number == 2:
            self.backbone = backbone_prio
        elif client_number == 3:
            self.backbone = backbone_dep
        else:
            self.backbone = backbone_act

    def forward(self, x):
        x = self.backbone(x)
        return x
