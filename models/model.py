""" Models used in the pipeline

"""
import torch.nn as nn
import torch.nn.functional as F

# __all__ = ["CustomResNet18", "build_model"]


class feature_extractor(nn.Module):
    def __init__(self, in_channels=30):
        super(feature_extractor, self).__init__()
        self.poorconv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=4, padding=0)
        self.poorconv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.poornorm1 = nn.BatchNorm2d(16)
        self.poorrelu1 = nn.ReLU(inplace=True)

        self.richconv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=4, padding=0)
        self.richconv2 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1)
        self.richnorm1 = nn.BatchNorm2d(16)
        self.richrelu1 = nn.ReLU(inplace=True)

    def forward(self, x, y):
        poorout = self.poorconv1(x)
        poorout = self.poorconv2(poorout)
        poorout = self.poornorm1(poorout)
        poorout = self.poorrelu1(poorout)

        richout = self.richconv1(y)
        richout = self.richconv2(richout)
        richout = self.richnorm1(richout)
        richout = self.richrelu1(richout)

        return richout - poorout


class ResNet(nn.Module):

    def __init__(self, n=7, res_option='A', use_dropout=False):
        super(ResNet, self).__init__()
        self.res_option = res_option
        self.use_dropout = use_dropout
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.layers1 = self._make_layer(n, 16, 16, 1)
        self.layers2 = self._make_layer(n, 32, 16, 2)
        self.layers3 = self._make_layer(n, 64, 32, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 1)

    def _make_layer(self, layer_count, channels, channels_in, stride):
        return nn.Sequential(
            ResBlock(channels, channels_in, stride, res_option=self.res_option,
                     use_dropout=self.use_dropout),
            *[ResBlock(channels) for _ in range(layer_count - 1)])

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.layers1(out)
        out = self.layers2(out)
        out = self.layers3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResBlock(nn.Module):

    def __init__(self, num_filters, channels_in=None, stride=1, res_option='A',
                 use_dropout=False):
        super(ResBlock, self).__init__()

        # uses 1x1 convolutions for downsampling
        if not channels_in or channels_in == num_filters:
            channels_in = num_filters
            self.projection = None
        else:
            if res_option == 'A':
                self.projection = IdentityPadding(num_filters, channels_in, stride)
            elif res_option == 'B':
                self.projection = ConvProjection(num_filters, channels_in, stride)
            elif res_option == 'C':
                self.projection = AvgPoolPadding(num_filters, channels_in, stride)
            else:
                raise ValueError(f"unknown res_option {res_option!r}")
        self.use_dropout = use_dropout

        self.conv1 = nn.Conv2d(channels_in, num_filters, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        if self.use_dropout:
            self.dropout = nn.Dropout(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_dropout:
            out = self.dropout(out)
        if self.projection:
            residual = self.projection(x)
        out += residual
        out = self.relu2(out)
        return out


# various projection options to change number of filters in residual connection
# option A from paper
class IdentityPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(IdentityPadding, self).__init__()
        # with kernel_size=1, max pooling is equivalent to identity mapping with stride
        self.identity = nn.MaxPool2d(1, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out


# option B from paper
class ConvProjection(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(ConvProjection, self).__init__()          # was `super(ResA, ...)`
        self.conv = nn.Conv2d(channels_in, num_filters, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv(x)
        return out


# experimental option C
class AvgPoolPadding(nn.Module):
    def __init__(self, num_filters, channels_in, stride):
        super(AvgPoolPadding, self).__init__()
        self.identity = nn.AvgPool2d(stride, stride=stride)
        self.num_zeros = num_filters - channels_in

    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.num_zeros))
        out = self.identity(out)
        return out


class CustomResNet18(nn.Module):
    def __init__(self, in_channels=30, n=7, res_option='A', use_dropout=False):
        super(CustomResNet18, self).__init__()
        self.residuals = feature_extractor(in_channels)
        self.classifier = ResNet(n, res_option, use_dropout)
        orig_conv1 = self.classifier.conv1
        self.classifier.conv1 = nn.Conv2d(
            in_channels=self.residuals.poorconv1.out_channels,
            out_channels=orig_conv1.out_channels,
            kernel_size=(3, 3),
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
        )

    def forward(self, poor, rich):
        out = self.residuals(poor, rich)
        out = self.classifier(out)
        return out


def build_custom_resnet_model():
    """Default model factory (referenced by training_parameters / Trainer)."""
    return CustomResNet18(in_channels=1, n=7, res_option='A', use_dropout=False)
