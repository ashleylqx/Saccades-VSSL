"""R2plus1D"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple


class SpatioTemporalConv(nn.Module):
    """Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  (1, kernel_size[1], kernel_size[2])
        spatial_stride =  (1, stride[1], stride[2])
        spatial_padding =  (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        # compute the number of intermediary channels (M) using formula
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        # print(intermed_channels)

        # the spatial conv is effectively a 2D conv due to the
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm
        # and ReLU added inside the model constructor, not here. This is an
        # intentional design choice, to allow this module to externally act
        # identical to a standard Conv3D, so it can be reused easily in any
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


class SpatioTemporalResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()

        # If downsample == True, the first conv of the layer has stride = 2
        # to halve the residual output size, and the input x is passed
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample

        # to allow for SAME padding
        padding = kernel_size // 2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2 when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock.
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, block_type=SpatioTemporalResBlock,
                 downsample=False):

        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock, with_classifier=False, return_conv=False, num_classes=101,
                 use_dropout=True, dropout=0.5, use_l2_norm=False, use_final_bn=False):
        super(R2Plus1DNet, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes
        self.use_l2_norm = use_l2_norm
        self.use_final_bn = use_final_bn

        message = 'Classifier to %d classes with r21d backbone;' % num_classes
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_final_bn: message += ' + final BN'
        print(message)

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        # if self.return_conv:
        self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))   # 9216
            # self.feature_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) 4182

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

        if self.with_classifier:
            if use_dropout:
                self.linear = nn.Sequential(
                    nn.Dropout(dropout),  # dropout=0.5, but args.dropout=0.9
                    nn.Linear(512, self.num_classes))
            else:
                self.linear = nn.Linear(512, self.num_classes)

            if use_final_bn:
                # self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
                self.final_bn = nn.BatchNorm1d(512)
                self.final_bn.weight.data.fill_(1)
                self.final_bn.bias.data.zero_()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        if self.return_conv:
            x = self.feature_pool(x)
            # print(x.shape)
            return x.view(x.shape[0], -1)

        x = self.pool(x)
        x = x.view(-1, 512)
        
        if self.with_classifier:
            if self.use_l2_norm:
                x = F.normalize(x, p=2, dim=1)

            if self.use_final_bn:
                x = self.linear(self.final_bn(x))
            else:
                x = self.linear(x)

        return x 


class R2Plus1DNet_Saccade(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
    """

    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock, with_classifier=False, return_conv=False, num_classes=101,
                 focus_level=5, conv_level=1, sample_num=1, focus_num=5, pro_p=1., pro_clamp_value=0.):
        super(R2Plus1DNet_Saccade, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes
        self.sample_num = sample_num * 2 * focus_num
        self.conv_level = conv_level
        FEAT_DIMS = [64, 64, 128, 256, 512]
        NUM_PREDS = [16, 16, 8, 4, 2]
        self.feat_dim = FEAT_DIMS[conv_level - 1]
        self.num_pred = NUM_PREDS[conv_level - 1] // 2

        self.pixpro_p = pro_p
        self.pixpro_clamp_value = pro_clamp_value

        assert focus_level in list(range(-1, 6)), 'focus level should be in 0~5, or -1 means no focus.'
        self.focus_level = focus_level  # 0~5
        self.F_SIZES = [112, 56, 56, 28, 14, 7]
        self.f_size = self.F_SIZES[self.focus_level]
        self.f_upsample = nn.Upsample(size=(self.f_size, self.f_size), mode='bilinear', align_corners=True)

        # first conv, with stride 1x2x2 and kernel size 1x7x7
        self.conv1 = SpatioTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU()
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], block_type=block_type, downsample=True)

        self.feature_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))   # 9216

        self.value_transform = nn.Sequential(
            conv1x1(self.feat_dim, self.feat_dim),
        )

        self.projector = conv1x1(self.feat_dim, self.feat_dim)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

        if self.with_classifier:
            self.linear = nn.Linear(512, self.num_classes)

    def forward(self, x, focus_map):
        focus_map = self.f_upsample(focus_map).unsqueeze(1)
        if self.focus_level == 0:
            tmp = torch.mul(x, focus_map)
            x = x + tmp

        x = self.relu1(self.bn1(self.conv1(x)))

        if self.focus_level == 1:
            x += torch.mul(x, focus_map)
        if self.conv_level == 1:
            x_proj = self.projector(x)
            x_reshape = x_proj.reshape(-1, self.sample_num, x.size(1), x.size(2), x.size(3), x.size(4))
            x_tgt_1 = x_reshape[:, :self.sample_num // 2, :, :, :, :]
            x_tgt_1 = x_tgt_1.permute(0, 2, 1, 3, 4, 5)
            x_tgt_1 = x_tgt_1.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_tgt_2 = x_reshape[:, self.sample_num // 2:, :, :, :, :]
            x_tgt_2 = x_tgt_2.permute(0, 2, 1, 3, 4, 5)
            x_tgt_2 = x_tgt_2.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_pred_2 = self.featprop(x_tgt_1)
            x_pred_1 = self.featprop(x_tgt_2)

        x = self.conv2(x)

        if self.focus_level == 2:
            x += torch.mul(x, focus_map)
        if self.conv_level == 2:
            x_proj = self.projector(x)
            x_reshape = x_proj.reshape(-1, self.sample_num, x.size(1), x.size(2), x.size(3), x.size(4))
            x_tgt_1 = x_reshape[:, :self.sample_num // 2, :, :, :, :]
            x_tgt_1 = x_tgt_1.permute(0, 2, 1, 3, 4, 5)
            x_tgt_1 = x_tgt_1.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_tgt_2 = x_reshape[:, self.sample_num // 2:, :, :, :, :]
            x_tgt_2 = x_tgt_2.permute(0, 2, 1, 3, 4, 5)
            x_tgt_2 = x_tgt_2.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_pred_2 = self.featprop(x_tgt_1)
            x_pred_1 = self.featprop(x_tgt_2)

        x = self.conv3(x)

        if self.focus_level == 3:
            x += torch.mul(x, focus_map)
        if self.conv_level == 3:
            x_proj = self.projector(x)
            x_reshape = x_proj.reshape(-1, self.sample_num, x.size(1), x.size(2), x.size(3), x.size(4))
            x_tgt_1 = x_reshape[:, :self.sample_num // 2, :, :, :, :]
            x_tgt_1 = x_tgt_1.permute(0, 2, 1, 3, 4, 5)
            x_tgt_1 = x_tgt_1.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_tgt_2 = x_reshape[:, self.sample_num // 2:, :, :, :, :]
            x_tgt_2 = x_tgt_2.permute(0, 2, 1, 3, 4, 5)
            x_tgt_2 = x_tgt_2.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_pred_2 = self.featprop(x_tgt_1)
            x_pred_1 = self.featprop(x_tgt_2)

        x = self.conv4(x)

        if self.focus_level == 4:
            x += torch.mul(x, focus_map)
        if self.conv_level == 4:
            x_proj = self.projector(x)
            x_reshape = x_proj.reshape(-1, self.sample_num, x.size(1), x.size(2), x.size(3), x.size(4))
            x_tgt_1 = x_reshape[:, :self.sample_num // 2, :, :, :, :]
            x_tgt_1 = x_tgt_1.permute(0, 2, 1, 3, 4, 5)
            x_tgt_1 = x_tgt_1.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_tgt_2 = x_reshape[:, self.sample_num // 2:, :, :, :, :]
            x_tgt_2 = x_tgt_2.permute(0, 2, 1, 3, 4, 5)
            x_tgt_2 = x_tgt_2.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_pred_2 = self.featprop(x_tgt_1)
            x_pred_1 = self.featprop(x_tgt_2)

        x = self.conv5(x)

        if self.focus_level == 5:
            x += torch.mul(x, focus_map)
        if self.conv_level == 5:
            x_proj = self.projector(x)
            x_reshape = x_proj.reshape(-1, self.sample_num, x.size(1), x.size(2), x.size(3), x.size(4))
            x_tgt_1 = x_reshape[:, :self.sample_num // 2, :, :, :, :]
            x_tgt_1 = x_tgt_1.permute(0, 2, 1, 3, 4, 5)
            x_tgt_1 = x_tgt_1.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_tgt_2 = x_reshape[:, self.sample_num // 2:, :, :, :, :]
            x_tgt_2 = x_tgt_2.permute(0, 2, 1, 3, 4, 5)
            x_tgt_2 = x_tgt_2.reshape(x_reshape.size(0), x.size(1), -1, x.size(3), x.size(4)).contiguous()
            x_pred_2 = self.featprop(x_tgt_1)
            x_pred_1 = self.featprop(x_tgt_2)

        if self.return_conv:
            x = self.feature_pool(x)
            pred_x = self.predictor(x.view(x.shape[0], -1))
            return x.view(x.shape[0], -1), pred_x

        x = self.pool(x)
        x = x.view(-1, 512)

        if self.with_classifier:
            x = self.linear(x)

        if self.conv_level > 0:
            return x, x_pred_1, x_pred_2, x_tgt_1, x_tgt_2
        else:
            return x, None, None, None, None

    def featprop(self, feat):
        N, C, T, H, W = feat.shape

        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = F.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)

        # Similarity calculation
        feat = F.normalize(feat, dim=1)
        feat = feat.view(N, C, -1)

        attention = torch.bmm(feat.transpose(1, 2), feat)
        attention = torch.clamp(attention, min=self.pixpro_clamp_value)
        if self.pixpro_p < 1.:
            attention = attention + 1e-6
        attention = attention ** self.pixpro_p

        feat = torch.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, T, H, W)


def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)

