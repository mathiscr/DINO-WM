import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
    def __init__(self, obs_list, act_list, frameskip, H):
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


def build_loader(
    B,
    config,
    split,
    world_size,
    rank,
    num_workers,
    data_dir,
    img_size=224,
):
    split_dir = Path(data_dir) / split
    target_dir = split_dir if split_dir.exists() else Path(data_dir)

    obs_list, act_list = load_trajectories(str(target_dir), img_size)
    token_dataset = DINOWMDataset(obs_list, act_list, config.frameskip, config.H)

    is_train = split == "train"
    sampler = (
        DistributedSampler(
            token_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train,
        )
        if world_size > 1
        else None
    )

    loader = DataLoader(
        token_dataset,
        batch_size=B,
        shuffle=(is_train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )
    return loader


def infinite_loader(loader):
    epoch = 0
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
