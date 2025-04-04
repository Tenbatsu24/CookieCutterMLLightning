import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, linear_bias=True, bn_affine=True):
        super(BasicBlock, self).__init__()

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=self.bn_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, linear_bias=True, bn_affine=True):
        super(Bottleneck, self).__init__()

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=self.bn_affine)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes, affine=self.bn_affine),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        F.relu(out, inplace=True)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        in_channels=3,
        num_classes=10,
        init_stride=1,
        linear_bias=True,
        bn_affine=True,
        return_latent=False,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.linear_bias = linear_bias
        self.bn_affine = bn_affine

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=init_stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64, affine=self.bn_affine)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.return_latent = return_latent
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if not self.return_latent:
            self.linear = nn.Linear(512 * block.expansion, num_classes, bias=self.linear_bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.linear_bias, self.bn_affine))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)

        if self.return_latent:
            return out
        else:
            return self.linear(out)


class ResNet18(ResNet):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        init_stride=1,
        linear_bias=True,
        bn_affine=True,
        return_latent=False,
        **kwargs,
    ):
        super(ResNet18, self).__init__(
            BasicBlock,
            [2, 2, 2, 2],
            in_channels=in_channels,
            num_classes=num_classes,
            init_stride=init_stride,
            linear_bias=linear_bias,
            bn_affine=bn_affine,
            return_latent=return_latent,
        )


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cardinality, base_width, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()

        dim = int(math.floor(planes * (base_width / 64.0)))

        self.conv_reduce = nn.Conv2d(
            inplanes, dim * cardinality, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_reduce = nn.BatchNorm2d(dim * cardinality)

        self.conv_conv = nn.Conv2d(
            dim * cardinality,
            dim * cardinality,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(dim * cardinality)

        self.conv_expand = nn.Conv2d(
            dim * cardinality, planes * 4, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn_expand = nn.BatchNorm2d(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        bottleneck = self.conv_reduce(x)
        bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

        bottleneck = self.conv_conv(bottleneck)
        bottleneck = F.relu(self.bn(bottleneck), inplace=True)

        bottleneck = self.conv_expand(bottleneck)
        bottleneck = self.bn_expand(bottleneck)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + bottleneck, inplace=True)


class CifarResNeXt(nn.Module):
    """ResNext optimized for the Cifar dataset, as specified in https://arxiv.org/pdf/1611.05431.pdf."""

    def __init__(
        self,
        block,
        depth,
        cardinality,
        base_width,
        in_channels,
        num_classes,
        init_stride,
        return_latent=False,
    ):
        super(CifarResNeXt, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 9 == 0, "depth should be one of 29, 38, 47, 56, 101"
        layer_blocks = (depth - 2) // 9

        self.cardinality = cardinality
        self.base_width = base_width
        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(in_channels, 64, 3, init_stride, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)

        self.inplanes = 64
        self.stage_1 = self._make_layer(block, 64, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 128, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 256, layer_blocks, 2)

        self.return_latent = return_latent

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if not self.return_latent:
            self.classifier = nn.Linear(256 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, self.cardinality, self.base_width, stride, downsample)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.cardinality, self.base_width))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.return_latent:
            return x
        else:
            return self.classifier(x)


class ResNeXt29(CifarResNeXt):

    def __init__(self, in_channels=3, num_classes=10, init_stride=1, return_latent=False):
        super(ResNeXt29, self).__init__(
            ResNeXtBottleneck,
            29,
            4,
            32,
            in_channels,
            num_classes,
            init_stride=init_stride,
            return_latent=return_latent,
        )
