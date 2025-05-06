import torch.nn as nn
from .common_models import ConcatEarly
from .common_models import Sequential, Transpose, Reshape


class GlobalModelForMUJOCO(nn.Module):
    def __init__(self, args):
        super(GlobalModelForMUJOCO, self).__init__()
        self.args = args
        fusion_input_dim = 256 * self.args.client_num if self.args.aggregation == "concat" else 256
        self.head = nn.Linear(fusion_input_dim, 2)
        # if args.aggregation == 'concat':
        #     fusion = ConcatEarly()
        #     self.head = nn.Linear(256 * 4, 2)
        # self.fusion = fusion

    def forward(self, input_list):
        if self.args.aggregation == 'sum':
            tensor_t = torch.stack(input_list).sum(dim=0)
        elif self.args.aggregation == 'mean':
            tensor_t = torch.stack(input_list).mean(dim=0)
        elif self.args.aggregation == 'concat':
            tensor_t = torch.cat(input_list, dim=2)
        # output_fusion = self.fusion(input_list)
        # output = self.head(output_fusion)
        output = self.head(tensor_t)
        return output


class LocalModelForMUJOCO(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForMUJOCO, self).__init__()
        self.args = args
        backbone_pos = Sequential(Transpose(0, 1), observation_pos_layers(
        64), GentlePushLateLSTM(64, 256), Transpose(0, 1))
        backbone_sensor = Sequential(Transpose(0, 1), observation_sensors_layers(
        64), GentlePushLateLSTM(64, 256), Transpose(0, 1))
        backbone_image = Sequential(Transpose(0, 1), Reshape([-1, 1, 32, 32]), observation_image_layers(
        64), Reshape([16, -1, 64]), GentlePushLateLSTM(64, 256), Transpose(0, 1))
        backbone_control = Sequential(Transpose(0, 1), control_layers(64),
               GentlePushLateLSTM(64, 256), Transpose(0, 1))

        if client_number == 0:
            self.backbone = backbone_pos
        elif client_number == 1:
            self.backbone = backbone_sensor
        elif client_number == 2:
            self.backbone = backbone_image
        else:
            self.backbone = backbone_control

    def forward(self, x):
        x = self.backbone(x)
        return x


"""
=====================================
"""
import torch
from fannypack.nn import resblocks

state_dim = 2
control_dim = 7
obs_pos_dim = 3
obs_sensors_dim = 7


def control_layers(units: int) -> nn.Module:
    """Create a control command encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(control_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


class _DualSpanningAvgPool(nn.Module):
    """Module with two average pools: one that spans the full height of the image and another the spans the full width. Outputs are flattened and concatenated."""

    def __init__(self, rows, cols, reduce_size=1):
        """Instantiate DualSpanningAvgPool Module.

        Args:
            rows (int): Number of rows in image.
            cols (int): Number of columns in image.
            reduce_size (int): Reduction size.
        """
        super().__init__()
        self.pool_h = nn.Sequential(
            nn.AvgPool2d((rows, reduce_size)),
            nn.Flatten(),
        )
        self.pool_w = nn.Sequential(
            nn.AvgPool2d((reduce_size, cols)),
            nn.Flatten(),
        )

    def forward(self, x):
        """Apply DualSpanningAvgPool Module to Layer Input.

        Args:
            x: Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return torch.cat((self.pool_h(x), self.pool_w(x)), dim=-1)


def observation_image_layers(units: int, spanning_avg_pool: bool = False) -> nn.Module:
    """Create an image encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    if spanning_avg_pool:
        # Architecture with full width/height average pools
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2,
                      kernel_size=3, padding=1),
            _DualSpanningAvgPool(rows=32, cols=32, reduce_size=2),
            nn.Linear(32 * 2, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
    else:
        # Default model
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=3, padding=1),
            nn.Flatten(),  # 32 * 32 * 8
            nn.Linear(8 * 32 * 32, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )


def observation_pos_layers(units: int) -> nn.Module:
    """Create an end effector position encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(obs_pos_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


def observation_sensors_layers(units: int) -> nn.Module:
    """Create an F/T sensor encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(obs_sensors_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


class GentlePushLateLSTM(nn.Module):
    """Implements Gentle Push's Late LSTM model."""

    def __init__(self, input_size, hidden_size):
        """Instantiate GentlePushLateLSTM Module.

        Args:
            input_size (int): Input dimension
            hidden_size (int): Hidden dimension
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        """Apply GentlePushLateLSTM to Model Input.

        Args:
            x (torch.Tensor): Model Input

        Returns:
            torch.Tensor: Model Output
        """
        x, _ = self.lstm(x)
        return x
