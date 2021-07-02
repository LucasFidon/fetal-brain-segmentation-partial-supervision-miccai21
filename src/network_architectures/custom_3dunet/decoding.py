from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from src.network_architectures.custom_3dunet.conv import ConvolutionalBlock

CHANNELS_DIMENSION = 1
UPSAMPLING_MODES = (
    'nearest',
    'linear',
    'bilinear',
    'bicubic',
    'trilinear',
)


class Decoder(nn.Module):
    def __init__(
            self,
            out_channels_encoding_blocks,
            dimensions: int,
            upsampling_type: str,
            num_decoding_blocks: int,
            normalization: Optional[str],
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            gradient_checkpointing: bool = False,
            ):
        super().__init__()
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        for i in range(num_decoding_blocks):
            is_last_block = (i == num_decoding_blocks-1)
            in_channels = out_channels_encoding_blocks[-(i+1)]
            in_channels_skip_connection = out_channels_encoding_blocks[-(i+2)]
            decoding_block = DecodingBlock(
                in_channels,
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
                is_last_block=is_last_block,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2

    def forward(self, skip_connections, x):
        zipped = zip(reversed(skip_connections), self.decoding_blocks)
        for skip_connection, decoding_block in zipped:
            x = decoding_block(skip_connection, x)
        return x


class DecodingBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_channels_skip_connection: int,
            dimensions: int,
            upsampling_type: str,
            normalization: Optional[str],
            preactivation: bool = True,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            dilation: Optional[int] = None,
            dropout: float = 0,
            is_last_block: bool = False,
            gradient_checkpointing: bool = False,
            ):
        super().__init__()

        self.residual = residual
        self.gradient_checkpointing = gradient_checkpointing
        kernel_size = (3, 3, 3)
        if upsampling_type == 'conv':
            # out_channels = in_channels
            out_channels = in_channels_skip_connection  # like in nnUNet
            self.upsample = get_conv_transpose_layer(
                dimensions, in_channels, out_channels)
            self.conv_bottleneck = None
        else:
            self.upsample = get_upsampling_layer(upsampling_type)
            if in_channels <= in_channels_skip_connection:
                self.conv_bottleneck = None
            else:
                # Reduce the number of feature of the low-res feature
                # to have in_channels = in_channels_skip_connection
                self.conv_bottleneck = ConvolutionalBlock(
                    dimensions,
                    in_channels,
                    in_channels_skip_connection,
                    normalization=normalization,
                    kernel_size=1,
                    preactivation=preactivation,
                    padding=padding,
                    padding_mode=padding_mode,
                    activation=activation,
                    dilation=dilation,
                    dropout=dropout,
                )
                in_channels = in_channels_skip_connection

        in_channels_first = in_channels + in_channels_skip_connection
        out_channels = in_channels_skip_connection

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels_first,
            out_channels,
            normalization=normalization,
            kernel_size=kernel_size,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )
        in_channels_second = out_channels
        self.conv2 = ConvolutionalBlock(
            dimensions,
            in_channels_second,
            out_channels,
            normalization=normalization,
            kernel_size=kernel_size,
            preactivation=preactivation,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels_first,
                out_channels,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

    def segment(self):
        def custom_forward(*inputs):
            if self.residual:
                connection = self.conv_residual(inputs[0])
                x = self.conv1(inputs[0])
                x = self.conv2(x)
                x += connection
            else:
                x = self.conv1(inputs[0])
                x = self.conv2(x)
            return x
        return custom_forward

    def forward(self, skip_connection, x):
        do_grad_check = self.training and self.gradient_checkpointing
        x = self.upsample(x)
        if self.conv_bottleneck is not None:
            x = self.conv_bottleneck(x)
        x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        if do_grad_check:
            x = checkpoint(self.segment(), x)
        else:
            x = self.segment()(x)
        return x


def get_upsampling_layer(upsampling_type: str, scale_factor=2) -> nn.Upsample:
    if upsampling_type not in UPSAMPLING_MODES:
        message = (
            'Upsampling type is "{}"'
            ' but should be one of the following: {}'
        )
        message = message.format(upsampling_type, UPSAMPLING_MODES)
        raise ValueError(message)
    return nn.Upsample(scale_factor=scale_factor, mode=upsampling_type, align_corners=False)


def get_conv_transpose_layer(dimensions, in_channels, out_channels):
    class_name = 'ConvTranspose{}d'.format(dimensions)
    conv_class = getattr(nn, class_name)
    conv_layer = conv_class(in_channels, out_channels, kernel_size=2, stride=2)
    return conv_layer


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == 'linear':
        if dimensions == 2:
            upsampling_type = 'bilinear'
        elif dimensions == 3:
            upsampling_type = 'trilinear'
    return upsampling_type
