import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from safetensors.torch import load_file

from layers import (
    Snake1d,
    WNConv1d,
    ResidualUnit,
    WNConvTranspose1d,
    init_weights,
)

def remove_weight_norm_recursive(m):
    """
    Recursively removes weight normalization from a module.
    """
    try:
        if hasattr(m, 'weight_g') and hasattr(m, 'weight_v'):
            # This is a good sign of weight_norm
            nn.utils.remove_weight_norm(m)
    except Exception as e:
        print(f"Could not remove weight norm from {m}: {e}")

def load_config(config_path: Path):
    
    # Load the initial configuration from the given path
    config = OmegaConf.load(config_path)

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(config["base_config"])
        config = OmegaConf.merge(base_config, config)

    return config
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        kernel_size: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        kernel_sizes,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, rates)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, kernel_size, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, x):
        return self.model(x)


class AudioTokenizer():
    def __init__(
        self,
        model_path: str,
    ):
        """loads the audio detokenizer with predefined config, uses torch as onnx is infact slower"""
        
        model_config = {'input_channel': 1024, 'channels': 1536, 'rates': [8, 5, 4, 2], 'kernel_sizes': [16, 11, 8, 4]}
        self.detokenizer = Decoder(**model_config)
        self.detokenizer.apply(remove_weight_norm_recursive)

        state_dict = load_file(model_path)
        missing_keys, unexpected_keys = self.detokenizer.load_state_dict(state_dict, strict=False)
        self.detokenizer = self.detokenizer.eval().half().to("cuda:0").half()
        

    def decode(self, x):
        return self.detokenizer(x)
