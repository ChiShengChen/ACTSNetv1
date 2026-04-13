from dataclasses import dataclass, field
from typing import List


@dataclass
class ACTSNetConfig:
    # Input
    n_channels: int = 35  # 7 electrodes × 5 sub-bands
    n_classes: int = 2     # responder vs non-responder
    n_timesteps: int = 128  # will be inferred from data if not set

    # Multi-scale encoding
    n_groups: int = 3
    group_conv_filters: int = 128
    group_kernel_size: int = 8

    # AC feature extractor
    conv_filters: List[int] = field(default_factory=lambda: [128, 256, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [8, 5, 3])

    # Prototype learning
    prototype_dim: int = 128   # embedding dimension d
    latent_dim_u: int = 64     # hidden latent space dimension u

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 16
    epochs: int = 200
    train_ratio: float = 0.7
    dropout: float = 0.1
    seed: int = 42

    # Device
    device: str = "cuda"
