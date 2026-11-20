import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.Linear(config.embd_dim, 3 * config.embd_dim, bias=False)
        self.proj = nn.Linear(config.embd_dim, config.embd_dim)
        self.n_head = config.n_head
        self.embd_dim = config.embd_dim
        assert self.embd_dim % self.n_head == 0
        self.head_dim = self.embd_dim // self.n_head

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # q (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # y (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.embd_dim, 4 * config.embd_dim)
        self.proj = nn.Linear(4 * config.embd_dim, config.embd_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embd_dim)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.embd_dim)
        self.mlp = MLP(config)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class VIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wpe = nn.Embedding(config.block_size, config.embd_dim)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embd_dim)

        num_frames = config.block_size // config.num_patches
        causal_mask = torch.tril(torch.ones(num_frames, num_frames, dtype=torch.bool))
        patch_mask = torch.ones(
            config.num_patches, config.num_patches, dtype=torch.bool
        )
        mask = torch.kron(causal_mask, patch_mask)[None, None, :, :]
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        # x (B, T=window_size * num_patches, embd_dim)
        B, T, C = x.size()
        pos = torch.arange(0, T, device=x.device)
        pos_embd = self.wpe(pos)  # (T, C)
        x = x + pos_embd
        for block in self.h:
            x = block(x, mask=self.mask[:, :, :T, :T])

        x = self.ln_f(x)
        return x
