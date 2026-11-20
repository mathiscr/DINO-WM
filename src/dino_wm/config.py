from dataclasses import dataclass


@dataclass
class VITConfig:
    window_size: int = 3
    image_size: int = 224
    patch_size: int = 14  # from the repo of dinov2: github.com/facebookresearch/dinov2
    num_patches: int = (224 // 14) ** 2
    n_layer: int = 6
    n_head: int = 16
    dino_dim: int = 384  # from the repo of dinov2
    action_dim: int = 16
    proprio_dim: int = 16
    embd_dim: int = dino_dim + action_dim + proprio_dim
