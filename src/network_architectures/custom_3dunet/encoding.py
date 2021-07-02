"""
Based on https://github.com/fepegar/unet/tree/master/unet
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from src.network_architectures.custom_3dunet.conv import ConvolutionalBlock

MAX_CHANNELS = 320


class ModuleWrapperIgnores2ndArg(nn.Module):
    """
    Only used in a trick for gradient checkpointing
    that allow to apply gradient checkpointing to the
    first layer without computing the derivative wrt the input.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, dummy_arg=None):
        assert dummy_arg is not None
        x = self.module(x)
        return x


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels_first: int,
            dimensions: int,
            pooling_type: str,
            num_encoding_blocks: int,
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

        self.encoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        self.gradient_checkpointing = gradient_checkpointing
        is_first_block = True
        out_channel = out_channels_first

        # define a dummy tensor for a trick that allow to use gradient checkpointing
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # define the encoding blocks
        for b in range(num_encoding_blocks):
            encoding_block = EncodingBlock(
                in_channels,
                out_channel,  # same as nnUNet, increase the nb of fea only once per level at the beginning
                out_channel,
                dimensions,
                normalization,
                pooling_type,
                preactivation,
                is_first_block=is_first_block,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            if is_first_block:
                # A trick for gradient checkpointing
                # this does not change the network architecture
                encoding_block = ModuleWrapperIgnores2ndArg(encoding_block)
            is_first_block = False
            self.encoding_blocks.append(encoding_block)
            in_channels = out_channel
            out_channel = min(2 * out_channel, MAX_CHANNELS)  # clip the max nb of channels
            if self.dilation is not None:
                self.dilation *= 2

    def forward(self, x):
        skip_connections = []
        do_grad_check = self.training and self.gradient_checkpointing
        for i, encoding_block in enumerate(self.encoding_blocks):
            if i == 0:
                if do_grad_check:
                    x, skip_connection = checkpoint(encoding_block, x, self.dummy_tensor)
                else:
                    x, skip_connection = encoding_block(x, self.dummy_tensor)
            else:
                if do_grad_check:
                    x, skip_connection = checkpoint(encoding_block, x)
                else:
                    x, skip_connection = encoding_block(x)
            skip_connections.append(skip_connection)
        return skip_connections, x

    @property
    def out_channels(self):
        return self.encoding_blocks[-1].out_channels


class EncodingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_first: int,
        out_channels_second: int,
        dimensions: int,
        normalization: Optional[str],
        pooling_type: Optional[str],
        preactivation: bool = False,
        is_first_block: bool = False,
        residual: bool = False,
        padding: int = 0,
        padding_mode: str = 'zeros',
        activation: Optional[str] = 'ReLU',
        dilation: Optional[int] = None,
        dropout: float = 0,
        ):
        super().__init__()

        self.normalization = normalization
        kernel_size = (3, 3, 3)
        self.residual = residual

        self.conv1 = ConvolutionalBlock(
            dimensions,
            in_channels,
            out_channels_first,
            normalization=normalization,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        self.conv2 = ConvolutionalBlock(
            dimensions,
            out_channels_first,
            out_channels_second,
            normalization=self.normalization,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            dilation=dilation,
            dropout=dropout,
        )

        if residual:
            self.conv_residual = ConvolutionalBlock(
                dimensions,
                in_channels,
                out_channels_second,
                kernel_size=1,
                normalization=None,
                activation=None,
            )

        self.downsample = None
        if pooling_type is not None:
            # downsampling op for the next level
            self.downsample = get_downsampling_layer(dimensions, pooling_type)

    def forward(self, x):
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)

        if self.downsample is None:
            return x
        else:
            skip_connection = x  # skip connection before pooling
            x = self.downsample(x)
            return x, skip_connection

    @property
    def out_channels(self):
        return self.conv2.conv_layer.out_channels


def get_downsampling_layer(
        dimensions,
        pooling_type,
        kernel_size=2,
        ) -> nn.Module:
    class_name = '{}Pool{}d'.format(pooling_type.capitalize(), dimensions)
    class_ = getattr(nn, class_name)
    return class_(kernel_size)
