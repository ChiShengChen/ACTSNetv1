import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionalConvBlock(nn.Module):
    """Single block: Conv1D → InstanceNorm1d → PReLU"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.PReLU(out_channels)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class AttentionalConvolution(nn.Module):
    """AC + Mul: Softmax attention over feature maps, then element-wise multiply."""

    def forward(self, x):
        # x: (batch, channels, time)
        attn = F.softmax(x, dim=1)  # AC = Softmax(X) along channel dim
        return attn * x              # Mul = AC ⊙ X


class ACEncoder(nn.Module):
    """Full AC feature extractor:
    3 × AttentionalConvBlock + AttentionalConvolution + FC + Sigmoid + InstanceNorm + GlobalAvgPool
    """

    def __init__(self, in_channels, conv_filters, kernel_sizes, prototype_dim, dropout=0.1):
        super().__init__()
        assert len(conv_filters) == 3 and len(kernel_sizes) == 3

        self.blocks = nn.Sequential(
            AttentionalConvBlock(in_channels, conv_filters[0], kernel_sizes[0]),
            AttentionalConvBlock(conv_filters[0], conv_filters[1], kernel_sizes[1]),
            AttentionalConvBlock(conv_filters[1], conv_filters[2], kernel_sizes[2]),
        )
        self.ac = AttentionalConvolution()
        self.fc = nn.Linear(conv_filters[2], prototype_dim)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.InstanceNorm1d(prototype_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.blocks(x)
        x = self.ac(x)
        # Permute for FC: (batch, time, channels)
        x = x.permute(0, 2, 1)
        x = self.sigmoid(self.fc(x))
        # Back to (batch, channels, time) for InstanceNorm
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.dropout(x)
        # Global average pooling: (batch, prototype_dim)
        x = x.mean(dim=2)
        return x


class MultiScaleEncoder(nn.Module):
    """Random dimension permutation → grouped Conv1D+BN+ReLU → GlobalPool → concatenate"""

    def __init__(self, n_channels, n_groups, conv_filters, kernel_size, seed=42):
        super().__init__()
        self.n_channels = n_channels
        self.n_groups = n_groups

        # Generate random dimension permutation groups (fixed per seed)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_channels)
        group_size = n_channels // n_groups
        self.groups = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else n_channels
            self.groups.append(perm[start:end].tolist())

        # Shared-parameter Conv1D+BN+ReLU block for all groups
        max_group_size = max(len(g) for g in self.groups)
        self.shared_conv = nn.Conv1d(max_group_size, conv_filters, kernel_size, padding=kernel_size // 2)
        self.shared_bn = nn.BatchNorm1d(conv_filters)
        self.shared_relu = nn.ReLU()

        nn.init.kaiming_normal_(self.shared_conv.weight, nonlinearity='relu')

        self.output_dim = conv_filters * n_groups

    def forward(self, x):
        # x: (batch, n_channels, time)
        batch_size, _, time_len = x.shape
        max_group_size = max(len(g) for g in self.groups)

        group_outputs = []
        for group_indices in self.groups:
            # Extract group channels
            g = x[:, group_indices, :]  # (batch, group_size, time)
            # Pad to max_group_size if needed (for shared conv)
            if g.shape[1] < max_group_size:
                pad = torch.zeros(batch_size, max_group_size - g.shape[1], time_len, device=x.device)
                g = torch.cat([g, pad], dim=1)
            g = self.shared_relu(self.shared_bn(self.shared_conv(g)))
            # Global average pooling: (batch, conv_filters)
            g = g.mean(dim=2)
            group_outputs.append(g)

        # Concatenate all group outputs: (batch, conv_filters * n_groups)
        return torch.cat(group_outputs, dim=1)


class PrototypicalLearning(nn.Module):
    """Attention-weighted class prototype computation + distance-based classification."""

    def __init__(self, embedding_dim, n_classes, latent_dim_u):
        super().__init__()
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        # Per-class attention parameters
        # V_k: (u, d), w_k: (u, 1)
        self.V = nn.ParameterList([
            nn.Parameter(torch.empty(latent_dim_u, embedding_dim))
            for _ in range(n_classes)
        ])
        self.w = nn.ParameterList([
            nn.Parameter(torch.empty(latent_dim_u, 1))
            for _ in range(n_classes)
        ])

        for v in self.V:
            nn.init.xavier_normal_(v)
        for w in self.w:
            nn.init.xavier_normal_(w)

    def compute_prototypes(self, embeddings, labels):
        """Compute attention-weighted prototypes for each class.

        Args:
            embeddings: (N, d) all sample embeddings
            labels: (N,) class labels

        Returns:
            prototypes: (n_classes, d)
        """
        prototypes = []
        for k in range(self.n_classes):
            mask = (labels == k)
            if mask.sum() == 0:
                # Fallback: use mean of all embeddings
                prototypes.append(embeddings.mean(dim=0))
                continue

            M_k = embeddings[mask]  # (|S_k|, d)

            # W_k = Softmax(w_k^T · tanh(V_k · M_k^T))
            projected = torch.tanh(F.linear(M_k, self.V[k]))  # (|S_k|, u)
            attn_logits = projected @ self.w[k]                 # (|S_k|, 1)
            attn_weights = F.softmax(attn_logits, dim=0)        # (|S_k|, 1)

            # h_k = sum_i W_{k,i} * M_{k,i}
            prototype = (attn_weights * M_k).sum(dim=0)  # (d,)
            prototypes.append(prototype)

        return torch.stack(prototypes, dim=0)  # (n_classes, d)

    def classify(self, query_embeddings, prototypes):
        """Classify via softmax over negative squared Euclidean distances.

        Args:
            query_embeddings: (batch, d)
            prototypes: (n_classes, d)

        Returns:
            log_probs: (batch, n_classes)
        """
        # Squared Euclidean distance: (batch, n_classes)
        dists = torch.cdist(query_embeddings, prototypes, p=2).pow(2)
        return F.log_softmax(-dists, dim=1)

    def forward(self, query_embeddings, support_embeddings, support_labels):
        prototypes = self.compute_prototypes(support_embeddings, support_labels)
        return self.classify(query_embeddings, prototypes)


class ACTSNet(nn.Module):
    """Full ACTSNet: MultiScaleEncoder + ACEncoder + PrototypicalLearning

    Architecture (replacing TapNet's LSTM branch with AC module):
        Input → MultiScaleEncoder → ms_out (pooled vector)
        Input → channel_proj → ACEncoder → ac_out (pooled vector)
        Concatenate(ms_out, ac_out) → FC → PrototypicalLearning → Classification
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Multi-scale encoding branch
        self.multi_scale = MultiScaleEncoder(
            n_channels=config.n_channels,
            n_groups=config.n_groups,
            conv_filters=config.group_conv_filters,
            kernel_size=config.group_kernel_size,
            seed=config.seed,
        )
        ms_out_dim = self.multi_scale.output_dim

        # AC feature extractor branch (operates on raw input, replacing LSTM)
        # Project n_channels → conv_filters[0] to match AC encoder input
        self.channel_proj = nn.Conv1d(config.n_channels, config.conv_filters[0], kernel_size=1)
        nn.init.kaiming_normal_(self.channel_proj.weight, nonlinearity='leaky_relu')

        self.ac_encoder = ACEncoder(
            in_channels=config.conv_filters[0],
            conv_filters=config.conv_filters,
            kernel_sizes=config.kernel_sizes,
            prototype_dim=config.prototype_dim,
            dropout=config.dropout,
        )

        # Final projection: concatenate multi-scale + AC embeddings → prototype_dim
        self.final_fc = nn.Linear(ms_out_dim + config.prototype_dim, config.prototype_dim)
        nn.init.xavier_normal_(self.final_fc.weight)

        # Prototype learning
        self.proto = PrototypicalLearning(
            embedding_dim=config.prototype_dim,
            n_classes=config.n_classes,
            latent_dim_u=config.latent_dim_u,
        )

    def encode(self, x):
        """Extract embedding for input time series.

        Args:
            x: (batch, n_channels, time)

        Returns:
            embedding: (batch, prototype_dim)
        """
        # Branch 1: Multi-scale encoding (pooled vector)
        ms_out = self.multi_scale(x)  # (batch, ms_out_dim)

        # Branch 2: AC encoder on raw input (replaces TapNet's LSTM branch)
        ac_input = self.channel_proj(x)        # (batch, conv_filters[0], time)
        ac_out = self.ac_encoder(ac_input)      # (batch, prototype_dim)

        # Concatenate both branches and project to prototype_dim
        combined = torch.cat([ms_out, ac_out], dim=1)
        embedding = self.final_fc(combined)     # (batch, prototype_dim)
        return embedding

    def forward(self, x, support_x=None, support_labels=None):
        """
        Args:
            x: query samples (batch, n_channels, time)
            support_x: support set samples for prototype computation.
                       If None, uses x itself as support set.
            support_labels: labels for support set

        Returns:
            log_probs: (batch, n_classes)
        """
        query_emb = self.encode(x)

        if support_x is None:
            support_emb = query_emb
        else:
            support_emb = self.encode(support_x)

        return self.proto(query_emb, support_emb, support_labels)
