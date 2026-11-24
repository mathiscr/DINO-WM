import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2


def load_trajectories(data_dir: str, img_size: int = 224):
    """Charge obses.pth et actions.pth, et met les images en (L_i, 3, 224, 224) uint8."""
    root = Path(data_dir)
    obses = torch.load(root / "obses.pth", map_location="cpu", weights_only=True)
    actions = torch.load(root / "actions.pth", map_location="cpu", weights_only=True)

    seq_file = root / "seq_lengths.pkl"
    if seq_file.exists():
        with open(seq_file, "rb") as f:
            seq_lengths = pickle.load(f)
    else:
        seq_lengths = [obses.size(1)] * obses.size(0)

    resize = v2.Resize((img_size, img_size), antialias=True)
    obs_list, act_list = [], []

    for i, L in enumerate(seq_lengths):
        ep_obs = obses[i, :L]
        if ep_obs.shape[-1] == 3:
            ep_obs = ep_obs.permute(0, 3, 1, 2)  # (L_i, 3, H_raw, W_raw)
        if ep_obs.dtype != torch.uint8:
            ep_obs = (ep_obs * 255).clamp(0, 255).to(torch.uint8)
        ep_obs = resize(ep_obs).contiguous()

        obs_list.append(ep_obs)
        act_list.append(actions[i, :L].to(torch.float32).contiguous())

    return obs_list, act_list


class DINOWMDataset(Dataset):
    def __init__(self, obs_list, act_list, config, frameskip, H):
        self.obs_list = obs_list
        self.act_list = act_list
        self.frameskip = frameskip
        self.H = H
        self.cumsum_lengths = np.cumsum(
            [0] + [max(0, obs.size(0) - H * frameskip) for obs in obs_list]
        )
        self.normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return int(self.cumsum_lengths[-1])

    def __getitem__(self, idx):
        idx_episode = np.searchsorted(self.cumsum_lengths, idx + 1, side="left") - 1
        idx_window = idx - self.cumsum_lengths[idx_episode]
        end = idx_window + self.H * self.frameskip
        observations = self.obs_list[idx_episode][idx_window : end + 1 : self.frameskip]
        observations = self.normalize(observations.float() / 255.0)
        actions = self.act_list[idx_episode][idx_window:end]
        actions = actions.view(self.H, -1)
        return observations, actions
