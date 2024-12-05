from typing import Union, Type, List, Tuple
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

class ResidualEncoderUNetWithClassifier(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.classifier = EncoderClassifier(self.encoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips), self.classifier(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

class EncoderClassifier(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 deep_supervision: bool = False,
                 conv_bias: bool = None):
        """
        Simplified UNet decoder for 3-class classification.
        """
        super().__init__()

        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = 3  # Fixed to 3 for this decoder

        n_stages_encoder = len(encoder.output_channels)

        # Build decoder stages
        self.transpconvs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            self.transpconvs.append(nn.ConvTranspose2d(
                in_channels=input_features_below,
                out_channels=input_features_skip,
                kernel_size=stride_for_transpconv,
                stride=stride_for_transpconv,
                bias=conv_bias
            ))

            self.stages.append(nn.Sequential(
                nn.Conv2d(2 * input_features_skip, input_features_skip, kernel_size=3, padding=1, bias=conv_bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_features_skip, input_features_skip, kernel_size=3, padding=1, bias=conv_bias),
                nn.ReLU(inplace=True)
            ))

            self.seg_layers.append(nn.Conv2d(
                in_channels=input_features_skip,
                out_channels=self.num_classes,
                kernel_size=1,
                bias=True
            ))

    def forward(self, skips):
        """
        Forward pass for the 3-class UNet decoder.
        :param skips: Encoder skip connections.
        :return: Segmentation logits for 3 classes.
        """
        lres_input = skips[-1]
        seg_outputs = []

        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), dim=1)
            x = self.stages[s](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        seg_outputs = seg_outputs[::-1]  # Reverse to prioritize largest resolution
        print(seg_outputs[0].shape)
        return seg_outputs[0] if not self.deep_supervision else seg_outputs
