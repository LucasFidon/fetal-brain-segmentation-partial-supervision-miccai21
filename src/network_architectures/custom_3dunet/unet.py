# -*- coding: utf-8 -*-

"""
Main module.
Based on https://github.com/fepegar/unet/tree/master/unet
"""

from typing import Optional
import torch.nn as nn
from src.network_architectures.custom_3dunet.encoding import Encoder, EncodingBlock
from src.network_architectures.custom_3dunet.decoding import Decoder
from src.network_architectures.custom_3dunet.conv import ConvolutionalBlock

MAX_CHANNELS = 320  # cap the number of features


class UNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_classes: int = 2,
            dimensions: int = 2,
            num_encoding_blocks: int = 5,
            out_channels_first_layer: int = 32,
            normalization: Optional[str] = None,
            pooling_type: str = 'max',
            upsampling_type: str = 'conv',
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
        depth = num_encoding_blocks - 1

        out_channels_encoding_blocks = [
            min(MAX_CHANNELS, out_channels_first_layer * (2 ** i))
            for i in range(num_encoding_blocks)
        ]

        # Force padding if residual blocks
        if residual:
            padding = 1

        # Encoder
        self.encoder = Encoder(
            in_channels,
            out_channels_first_layer,
            # out_channels_encoding_blocks,
            dimensions,
            pooling_type,
            depth,
            normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=initial_dilation,
            dropout=dropout,
            gradient_checkpointing=gradient_checkpointing,
        )

        bottom_in_channel = min(
            MAX_CHANNELS,
            out_channels_first_layer * (2 ** (num_encoding_blocks - 2))
        )
        bottom_out_channels = min(
            MAX_CHANNELS,
            bottom_in_channel * 2,
        )

        self.bottom_block = EncodingBlock(
            in_channels=bottom_in_channel,
            out_channels_first=bottom_out_channels,
            out_channels_second=bottom_out_channels,
            dimensions=dimensions,
            normalization=normalization,
            pooling_type=None,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Decoder
        num_decoding_blocks = depth
        self.decoder = Decoder(
            out_channels_encoding_blocks,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )

        # Classifier
        if dimensions == 2:
            in_channels = out_channels_first_layer
        elif dimensions == 3:
            in_channels = out_channels_first_layer
        self.classifier = ConvolutionalBlock(
            dimensions, in_channels, out_classes,
            kernel_size=1, activation=None,
        )

    def forward(self, x):
        skip_connections, encoding = self.encoder(x)
        encoding = self.bottom_block(encoding)
        x = self.decoder(skip_connections, encoding)
        return self.classifier(x)


class UNet3D(UNet):
    def __init__(self, *args, **user_kwargs):
        kwargs = {}
        kwargs['dimensions'] = 3
        kwargs['num_encoding_blocks'] = 4
        kwargs['out_channels_first_layer'] = 30
        kwargs['normalization'] = 'instance'
        kwargs.update(user_kwargs)
        super().__init__(*args, **kwargs)
