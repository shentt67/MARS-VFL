from utils.utils import *
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable


class LowRankTensorFusion(nn.Module):

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        super(LowRankTensorFusion, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        # low-rank factors
        self.factors = nn.ParameterList()
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim + 1, self.output_dim))
            nn.init.xavier_normal(factor)
            self.factors.append(factor)
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1, self.output_dim))
        # init the fusion weights
        nn.init.xavier_normal(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):
        batch_size = modalities[0].shape[0]
        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(modality.device)
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output

class Concat(nn.Module):

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.cat(flattened, dim=1)

class ConcatEarly(nn.Module):

    def __init__(self):
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        return torch.cat(modalities, dim=2)

class roboticsConcat(nn.Module):

    def __init__(self, name=None):
        super(roboticsConcat, self).__init__()
        self.name = name

    def forward(self, x):
        if self.name == "noconcat":
            return [x[0][0].squeeze(), x[1].squeeze(), x[2].squeeze(), x[3][0].squeeze(), x[4]]
        if self.name == "image":
            return torch.cat([x[0][0].squeeze(), x[1][0].squeeze(), x[2]], 1)
        if self.name == "simple":
            return torch.cat([x[0].squeeze(), x[1]], 1)
        return torch.cat([x[0][0].squeeze(), x[1].squeeze(), x[2].squeeze(), x[3][0].squeeze(), x[4]], 1)


class Sequential2(nn.Module):

    def __init__(self, a, b):
        super(Sequential2, self).__init__()
        self.model = nn.Sequential(a, b)

    def forward(self, x):
        return self.model(x)


class CausalConv1D(nn.Conv1d):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        self.__padding = (kernel_size - 1) * dilation

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        res = super().forward(x)
        if self.__padding != 0:
            return res[:, :, : -self.__padding]
        return res


class Flatten(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    image_transform = image * scale
    return image_transform.transpose(1, 3).transpose(2, 3)


def filter_depth(depth_image):
    depth_image = torch.where(
        depth_image > 1e-7, depth_image, torch.zeros_like(depth_image)
    )
    return torch.where(depth_image < 2, depth_image, torch.zeros_like(depth_image))


class MLP(torch.nn.Module):

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class GRU(torch.nn.Module):

    def __init__(self, indim, hiddim, dropout=False, dropoutp=0.1, flatten=False, has_padding=False, last_only=False,
                 batch_first=True):
        super(GRU, self).__init__()
        self.gru = nn.GRU(indim, hiddim, batch_first=True)
        self.dropout = dropout
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.flatten = flatten
        self.has_padding = has_padding
        self.last_only = last_only
        self.batch_first = batch_first

    def forward(self, x):
        if self.has_padding:
            x = pack_padded_sequence(
                x[0], x[1], batch_first=self.batch_first, enforce_sorted=False)
            out = self.gru(x)[1][-1]
        elif self.last_only:
            out = self.gru(x)[1][0]

            return out
        else:
            out, l = self.gru(x)
        if self.dropout:
            out = self.dropout_layer(out)
        if self.flatten:
            out = torch.flatten(out, 1)

        return out


class Sequential(nn.Sequential):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if 'training' in kwargs:
            del kwargs['training']
        return super().forward(*args, **kwargs)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)


class Reshape(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return torch.reshape(x, self.shape)


class ProprioEncoder(nn.Module):

    def __init__(self, z_dim, alpha, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.proprio_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, proprio):
        return self.proprio_encoder(self.alpha * proprio).unsqueeze(2)


class ForceEncoder(nn.Module):

    def __init__(self, z_dim, alpha, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.frc_encoder = nn.Sequential(
            CausalConv1D(6, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(32, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(64, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, 2 * self.z_dim, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, force):
        return self.frc_encoder(self.alpha * force)


class ImageEncoder(nn.Module):

    def __init__(self, z_dim, alpha, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, vis_in):
        image = rescaleImage(vis_in)

        # image encoding layers
        out_img_conv1 = self.img_conv1(self.alpha * image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs


class DepthEncoder(nn.Module):

    def __init__(self, z_dim, alpha, initialize_weights=True):
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.depth_conv1 = conv2d(1, 32, kernel_size=3, stride=2)
        self.depth_conv2 = conv2d(32, 64, kernel_size=3, stride=2)
        self.depth_conv3 = conv2d(64, 64, kernel_size=4, stride=2)
        self.depth_conv4 = conv2d(64, 64, stride=2)
        self.depth_conv5 = conv2d(64, 128, stride=2)
        self.depth_conv6 = conv2d(128, self.z_dim, stride=2)

        self.depth_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, depth_in):
        depth = filter_depth(depth_in)

        # depth encoding layers
        out_depth_conv1 = self.depth_conv1(self.alpha * depth)
        out_depth_conv2 = self.depth_conv2(out_depth_conv1)
        out_depth_conv3 = self.depth_conv3(out_depth_conv2)
        out_depth_conv4 = self.depth_conv4(out_depth_conv3)
        out_depth_conv5 = self.depth_conv5(out_depth_conv4)
        out_depth_conv6 = self.depth_conv6(out_depth_conv5)

        depth_out_convs = (
            out_depth_conv1,
            out_depth_conv2,
            out_depth_conv3,
            out_depth_conv4,
            out_depth_conv5,
            out_depth_conv6,
        )

        # depth embedding parameters
        flattened = self.flatten(out_depth_conv6)
        depth_out = self.depth_encoder(flattened).unsqueeze(2)

        return depth_out, depth_out_convs


class ActionEncoder(nn.Module):

    def __init__(self, action_dim):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, action):
        if action is None:
            return None
        return self.action_encoder(action)


class Maxout(nn.Module):

    def __init__(self, d, m, k):
        super(Maxout, self).__init__()
        self.d_in, self.d_out, self.pool_size = d, m, k
        self.lin = nn.Linear(d, m * k)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, _ = out.view(*shape).max(dim=max_dim)
        return m


class MaxOut_MLP(nn.Module):

    def __init__(
            self, num_outputs, first_hidden=64, number_input_feats=300, second_hidden=None, linear_layer=True):
        super(MaxOut_MLP, self).__init__()

        if second_hidden is None:
            second_hidden = first_hidden
        self.op0 = nn.BatchNorm1d(number_input_feats, 1e-4)
        self.op1 = Maxout(number_input_feats, first_hidden, 2)
        self.op2 = nn.Sequential(nn.BatchNorm1d(first_hidden), nn.Dropout(0.3))
        # self.op2 = nn.BatchNorm1d(first_hidden)
        # self.op3 = Maxout(first_hidden, first_hidden * 2, 5)
        self.op3 = Maxout(first_hidden, second_hidden, 2)
        self.op4 = nn.Sequential(nn.BatchNorm1d(
            second_hidden), nn.Dropout(0.3))
        # self.op4 = nn.BatchNorm1d(second_hidden)

        # The linear layer that maps from hidden state space to output space
        if linear_layer:
            self.hid2val = nn.Linear(second_hidden, num_outputs)
        else:
            self.hid2val = None

    def forward(self, x):
        o0 = self.op0(x)
        o1 = self.op1(o0)
        o2 = self.op2(o1)
        o3 = self.op3(o2)
        o4 = self.op4(o3)
        if self.hid2val is None:
            return o4
        o5 = self.hid2val(o4)

        return o5
