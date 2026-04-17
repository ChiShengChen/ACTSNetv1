"""Self-supervised pretraining for ACTSNet encoder.

Pretraining tasks:
1. Contrastive learning: augmented views of same sample should have similar embeddings
2. Masked reconstruction: mask random time segments and reconstruct from embedding

The pretrained encoder (MultiScaleEncoder + ACEncoder + final_fc) can then be
loaded for downstream fine-tuning with PrototypicalLearning head.
"""
from __future__ import annotations

import argparse
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .config import ACTSNetConfig
from .model import ACTSNet, MultiScaleEncoder, ACEncoder


class ACTSNetPretrainEncoder(nn.Module):
    """ACTSNet encoder with self-supervised pretrain heads.

    Encoder: MultiScaleEncoder + ACEncoder + final_fc → embedding (prototype_dim)
    Heads:
      - Projector: embedding → projection (for contrastive loss)
      - Reconstructor: embedding → reconstructed time series (for reconstruction loss)
    """

    def __init__(self, config: ACTSNetConfig, max_channels: int = 128):
        super().__init__()
        self.config = config
        self.max_channels = max_channels

        # Channel projection to handle variable channel counts
        self.channel_proj = nn.Conv1d(max_channels, config.conv_filters[0], kernel_size=1)
        nn.init.kaiming_normal_(self.channel_proj.weight, nonlinearity='leaky_relu')

        # Multi-scale encoding branch
        self.multi_scale = MultiScaleEncoder(
            n_channels=max_channels,
            n_groups=config.n_groups,
            conv_filters=config.group_conv_filters,
            kernel_size=config.group_kernel_size,
            seed=config.seed,
        )
        ms_out_dim = self.multi_scale.output_dim

        # AC feature extractor branch
        self.ac_encoder = ACEncoder(
            in_channels=config.conv_filters[0],
            conv_filters=config.conv_filters,
            kernel_sizes=config.kernel_sizes,
            prototype_dim=config.prototype_dim,
            dropout=config.dropout,
        )

        # Final projection
        self.final_fc = nn.Linear(ms_out_dim + config.prototype_dim, config.prototype_dim)
        nn.init.xavier_normal_(self.final_fc.weight)

        # ── Pretrain heads ──
        # Contrastive projector: embedding → 64-dim normalized projection
        proj_dim = 64
        self.projector = nn.Sequential(
            nn.Linear(config.prototype_dim, config.prototype_dim),
            nn.ReLU(),
            nn.Linear(config.prototype_dim, proj_dim),
        )

        # Reconstructor: embedding → reconstruct masked input
        self.reconstructor = nn.Sequential(
            nn.Linear(config.prototype_dim, config.prototype_dim * 2),
            nn.ReLU(),
            nn.Linear(config.prototype_dim * 2, max_channels * config.n_timesteps),
        )

    def pad_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to max_channels. x: (batch, n_ch, T)"""
        n_ch = x.shape[1]
        if n_ch < self.max_channels:
            pad = torch.zeros(x.shape[0], self.max_channels - n_ch, x.shape[2], device=x.device)
            x = torch.cat([x, pad], dim=1)
        elif n_ch > self.max_channels:
            x = x[:, :self.max_channels, :]
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding. x: (batch, max_channels, T)"""
        ms_out = self.multi_scale(x)
        ac_input = self.channel_proj(x)
        ac_out = self.ac_encoder(ac_input)
        combined = torch.cat([ms_out, ac_out], dim=1)
        return self.final_fc(combined)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_ch, T) — raw input (will be padded)
        Returns:
            embedding, projection, reconstruction
        """
        x = self.pad_channels(x)
        emb = self.encode(x)
        proj = F.normalize(self.projector(emb), dim=-1)
        recon = self.reconstructor(emb).view(x.shape[0], self.max_channels, -1)
        return emb, proj, recon

    def get_encoder_state_dict(self) -> dict:
        """Extract encoder weights (without pretrain heads) for downstream loading."""
        state = {}
        for name, param in self.named_parameters():
            if name.startswith(('multi_scale.', 'ac_encoder.', 'channel_proj.', 'final_fc.')):
                state[name] = param.data.clone()
        return state


# ──────────────────────────────────────────────
# Augmentations for contrastive learning
# ──────────────────────────────────────────────
def augment_eeg(x: torch.Tensor, mask_ratio: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """Create two augmented views of x for contrastive learning.

    Augmentations:
      - Temporal masking (random segments zeroed out)
      - Gaussian noise
      - Channel dropout

    Returns:
        view1, view2: augmented versions of x
    """
    batch, n_ch, T = x.shape

    views = []
    for _ in range(2):
        v = x.clone()

        # Temporal masking: zero out random contiguous segments
        mask_len = int(T * mask_ratio)
        for b in range(batch):
            start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            v[b, :, start:start + mask_len] = 0.0

        # Gaussian noise
        noise = torch.randn_like(v) * 0.1
        v = v + noise

        # Channel dropout (10% channels)
        ch_mask = (torch.rand(batch, n_ch, 1, device=x.device) > 0.1).float()
        v = v * ch_mask

        views.append(v)

    return views[0], views[1]


# ──────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """NT-Xent (SimCLR) contrastive loss."""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, proj_dim)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    return F.cross_entropy(sim, labels)


# ──────────────────────────────────────────────
# Dataset for pretraining (loads from npy cache)
# ──────────────────────────────────────────────
class PretrainEEGDataset(Dataset):
    """Dataset that loads (data, n_channels) tuples from npy files.
    Pads channels to max_channels at dataset level so batches can stack."""

    def __init__(self, data: np.ndarray, max_channels: int = 128, max_time_len: int = 1024):
        self.max_channels = max_channels
        # Truncate time if needed
        if data.shape[-1] > max_time_len:
            data = data[:, :, :max_time_len]
        self.n_channels = data.shape[1]
        # Pad or truncate channels to max_channels
        if self.n_channels < max_channels:
            pad = np.zeros((data.shape[0], max_channels - self.n_channels, data.shape[2]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=1)
        elif self.n_channels > max_channels:
            data = data[:, :max_channels, :]
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.n_channels


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def pretrain(
    model: ACTSNetPretrainEncoder,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    logger: logging.Logger,
    checkpoint_dir: str,
    mask_ratio: float = 0.3,
    recon_weight: float = 1.0,
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95)
    )

    total_steps = epochs * len(dataloader)
    warmup_steps = 2 * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(f"Starting pretrain: {epochs} epochs, {total_steps} total steps")
    logger.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_cl, epoch_rl = 0.0, 0.0, 0.0
        t0 = time.time()

        for batch_idx, (x, n_ch) in enumerate(dataloader):
            x = x.to(device)
            x_padded = model.pad_channels(x)

            # Create augmented views
            view1, view2 = augment_eeg(x_padded, mask_ratio=mask_ratio)

            # Forward both views
            _, proj1, recon1 = model(view1)
            _, proj2, recon2 = model(view2)

            # Contrastive loss
            loss_cl = nt_xent_loss(proj1, proj2)

            # Reconstruction loss (reconstruct original from masked view)
            loss_rl = F.mse_loss(recon1, x_padded) + F.mse_loss(recon2, x_padded)
            loss_rl = loss_rl * 0.5

            loss = loss_cl + recon_weight * loss_rl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_cl += loss_cl.item()
            epoch_rl += loss_rl.item()

            if batch_idx > 0 and batch_idx % 500 == 0:
                logger.info(
                    f"  [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {epoch_loss/batch_idx:.4f} "
                    f"(CL: {epoch_cl/batch_idx:.4f}, RL: {epoch_rl/batch_idx:.4f})"
                )

        n = max(1, len(dataloader))
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {epoch_loss/n:.4f} (CL: {epoch_cl/n:.4f}, RL: {epoch_rl/n:.4f}) | "
            f"LR: {lr_now:.6f} | {dt:.0f}s"
        )

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            ckpt_path = os.path.join(checkpoint_dir, f"pretrain_epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.get_encoder_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model.config,
                'max_channels': model.max_channels,
            }, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

    # Save final
    final_path = os.path.join(checkpoint_dir, "pretrain_final.pt")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.get_encoder_state_dict(),
        'config': model.config,
        'max_channels': model.max_channels,
    }, final_path)
    logger.info(f"Pretrain complete! Final: {final_path}")
    return model
