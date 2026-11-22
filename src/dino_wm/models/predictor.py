import torch
from torch import nn
from torch.nn import functional as F


class Predictor(nn.Module):
    def __init__(self, config, vit, action_module, dino):
        super().__init__()
        self.vit = vit
        self.action_module = action_module
        self.dino = dino
        self.dino.eval()
        self.dino.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, observations, actions, proprios=None, loss_calculation=False):
        B, T, C, H, W = observations.size()
        # encoding the observations
        observations = observations.view(B * T, C, H, W)
        with torch.no_grad():
            features = self.dino.forward_features(observations)
            patch_tokens = features["x_norm_patchtokens"]
            _, n_patches, dino_dim = patch_tokens.size()

        encoded_observations = patch_tokens.view(B, T, n_patches, dino_dim)

        # linking with actions and proprio
        actions = actions[
            :, :-1, :
        ]  # we don't have an action at the end, we predict only an image
        actions = self.action_module(actions)  # (B, T, action_dim)
        action_dim = actions.size()[-1]
        actions = actions[:, :, None, :].expand(-1, -1, n_patches, -1)

        x = encoded_observations[:, :-1, :, :]

        x = torch.cat([x, actions], dim=-1)
        x = x.view(B, (T - 1) * n_patches, dino_dim + action_dim)

        # feeding the vit
        out = self.vit(x)
        # out (B, T*n_patch, dim)
        out = out.view(B, (T - 1), n_patches, dino_dim + action_dim)
        out = out[:, :, :, :dino_dim]  # we only care about the visual aspect

        # calculation of the loss
        loss = None
        if loss_calculation:
            target = encoded_observations[:, 1:, :, :]
            loss = F.mse_loss(out, target)
        return out, loss
