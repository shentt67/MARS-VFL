import torch.nn as nn
from .common_models import MaxOut_MLP, Concat
import torch


class GlobalModelForMMIMDB(nn.Module):
    def __init__(self, args):
        super(GlobalModelForMMIMDB, self).__init__()
        self.args = args
        fusion_input_dim = 512 * self.args.client_num if self.args.aggregation == "concat" else 512
        self.classifier = nn.Linear(fusion_input_dim, 23)

        self.args = args

    def forward(self, input_list):
        if self.args.aggregation == 'sum':
            tensor_t = torch.stack(input_list).sum(dim=0)
        elif self.args.aggregation == 'mean':
            tensor_t = torch.stack(input_list).mean(dim=0)
        elif self.args.aggregation == 'concat':
            tensor_t = torch.cat(input_list, dim=1)

        x = tensor_t
        x = self.classifier(x)

        return x


class LocalModelForMMIMDB(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForMMIMDB, self).__init__()
        self.args = args
        backbone_img = MaxOut_MLP(512, 512, 300, linear_layer=False)
        backbone_text = MaxOut_MLP(512, 1024, 4096, 512, False)

        if client_number == 0:
            self.backbone = backbone_img
        elif client_number == 1:
            self.backbone = backbone_text

    def forward(self, x):
        x = self.backbone(x)
        return x
