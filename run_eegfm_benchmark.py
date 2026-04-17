"""ACTSNet Benchmark on EEG-FM-Bench datasets.

Usage:
    python run_eegfm_benchmark.py [--datasets bcic_2a seed_iv tuab tuev] [--seeds 42 123 456]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pyarrow.ipc as ipc
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from actsnet.config import ACTSNetConfig
from actsnet.model import ACTSNet

# ──────────────────────────────────────────────
# Dataset info
# ──────────────────────────────────────────────
EEGFM_DATASETS = {
    "bcic_2a":     {"n_classes": 4, "version": "1.0.0"},
    "seed_iv":     {"n_classes": 4, "version": "1.0.0"},
    "tuab":        {"n_classes": 2, "version": "3.0.1"},
    "tuev":        {"n_classes": 6, "version": "2.0.0"},
    "tusl":        {"n_classes": 3, "version": "2.0.1"},
    "siena_scalp": {"n_classes": 2, "version": "1.0.0"},
}

EEGFM_ROOT_CANDIDATES = [
    "/media/meow/Transcend/time_series_benchmark/eegfm_data_cache",
    "/media/meow/Elements/EEG-FM-Bench-data/processed/fs_256",
    "/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256",
]


def get_eegfm_root(dataset_name: str | None = None):
    """Find a root that contains the given dataset. Falls back to first existing root."""
    for p in EEGFM_ROOT_CANDIDATES:
        if not os.path.isdir(p):
            continue
        if dataset_name and os.path.isdir(os.path.join(p, dataset_name)):
            return p
        if dataset_name is None:
            return p
    raise FileNotFoundError(f"EEG-FM-Bench data root not found for {dataset_name}")


# ──────────────────────────────────────────────
# Arrow data loading
# ──────────────────────────────────────────────
def load_arrow_split(
    root: str, dataset: str, version: str, split: str,
    max_samples: int | None = None, max_time_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a split from arrow shards, return (data, labels) as numpy arrays.

    Supports early subsampling and time truncation to avoid OOM on large datasets.
    """
    base_dir = os.path.join(root, dataset, "finetune", version)
    prefix = f"{dataset}-{split}-"

    shards = sorted([
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(prefix) and f.endswith(".arrow")
    ])
    if not shards:
        raise FileNotFoundError(f"No shards for {dataset}/{split} in {base_dir}")

    # First pass: count total samples and get shape
    total_count = 0
    for shard_path in shards:
        reader = ipc.open_stream(shard_path)
        table = reader.read_all()
        total_count += len(table)

    # Determine which indices to keep (subsample before loading data)
    if max_samples is not None and total_count > max_samples:
        rng = np.random.RandomState(42)
        keep_indices = set(rng.choice(total_count, max_samples, replace=False).tolist())
    else:
        keep_indices = None  # keep all

    # Second pass: load only selected samples
    all_data, all_labels = [], []
    global_idx = 0
    for shard_path in shards:
        reader = ipc.open_stream(shard_path)
        table = reader.read_all()
        for i in range(len(table)):
            if keep_indices is None or global_idx in keep_indices:
                d = np.array(table.column("data")[i].as_py(), dtype=np.float32)
                if max_time_len is not None and d.shape[-1] > max_time_len:
                    d = d[:, :max_time_len]
                all_data.append(d)
                all_labels.append(int(table.column("label")[i].as_py()))
            global_idx += 1

    data = np.stack(all_data, axis=0)   # (N, n_ch, T)
    labels = np.array(all_labels, dtype=np.int64)
    return data, labels


class ArrowEEGDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, max_len: int | None = None):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        # Optionally truncate time dimension for memory
        if max_len is not None and self.data.shape[-1] > max_len:
            self.data = self.data[:, :, :max_len]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        log_probs = model(x, support_x=x, support_labels=y)
        loss = F.nll_loss(log_probs, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n_samples += x.size(0)
    return total_loss / max(1, n_samples)


@torch.no_grad()
def evaluate_batched(model, train_loader, test_loader, device, max_support=5000):
    """Evaluate using train set as support, test set as query.
    For large datasets, subsample support set to max_support."""
    model.eval()

    # Collect support set (from train)
    support_x_list, support_y_list = [], []
    n_collected = 0
    for x, y in train_loader:
        support_x_list.append(x)
        support_y_list.append(y)
        n_collected += x.size(0)
        if n_collected >= max_support:
            break
    support_x = torch.cat(support_x_list, dim=0)[:max_support].to(device)
    support_y = torch.cat(support_y_list, dim=0)[:max_support].to(device)

    # Encode support set in chunks to save memory
    chunk_size = 256
    support_embs = []
    for i in range(0, len(support_x), chunk_size):
        emb = model.encode(support_x[i:i+chunk_size])
        support_embs.append(emb)
    support_emb = torch.cat(support_embs, dim=0)

    # Evaluate query set
    all_preds, all_labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        query_emb = model.encode(x)
        log_probs = model.proto(query_emb, support_emb, support_y)
        preds = log_probs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "cohen_kappa": cohen_kappa_score(all_labels, all_preds),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }
    return metrics


def load_pretrained_encoder(model: ACTSNet, pretrained_path: str, logger: logging.Logger):
    """Load pretrained encoder weights into ACTSNet model."""
    ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    encoder_state = ckpt.get('encoder_state_dict', {})
    if not encoder_state:
        logger.warning("No encoder_state_dict in checkpoint, skipping pretrain load")
        return model

    # Map pretrained weights to ACTSNet
    model_state = model.state_dict()
    loaded = 0
    for name, param in encoder_state.items():
        if name in model_state and model_state[name].shape == param.shape:
            model_state[name] = param
            loaded += 1
        else:
            logger.debug(f"  Skip {name} (shape mismatch or not in model)")

    model.load_state_dict(model_state)
    logger.info(f"Loaded {loaded} pretrained encoder params from {pretrained_path}")
    return model


def run_single_dataset(
    dataset_name: str,
    eegfm_root: str,
    seeds: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
    max_time_len: int | None = None,
    max_train_samples: int | None = None,
    pretrained_path: str | None = None,
):
    info = EEGFM_DATASETS[dataset_name]
    n_classes = info["n_classes"]
    version = info["version"]

    logger.info(f"Loading {dataset_name} ...")
    t0 = time.time()
    train_data, train_labels = load_arrow_split(
        eegfm_root, dataset_name, version, "train",
        max_samples=max_train_samples, max_time_len=max_time_len,
    )
    test_data, test_labels = load_arrow_split(
        eegfm_root, dataset_name, version, "test",
        max_time_len=max_time_len,
    )

    n_channels = train_data.shape[1]
    n_timesteps = train_data.shape[2]
    logger.info(
        f"  Loaded in {time.time()-t0:.0f}s | "
        f"Train: {len(train_data)}, Test: {len(test_data)} | "
        f"Shape: ({n_channels}, {n_timesteps}) | Classes: {n_classes}"
    )

    all_seed_results: dict[str, list[float]] = {}

    for seed in seeds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Seed: {seed}")
        set_seed(seed)

        train_ds = ArrowEEGDataset(train_data, train_labels)
        test_ds = ArrowEEGDataset(test_data, test_labels)

        actual_timesteps = train_ds.data.shape[-1]

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=2
        )

        config = ACTSNetConfig(
            n_channels=n_channels,
            n_classes=n_classes,
            n_timesteps=actual_timesteps,
            n_groups=min(3, n_channels),  # Ensure n_groups <= n_channels
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed,
        )

        model = ACTSNet(config)
        if pretrained_path:
            model = load_pretrained_encoder(model, pretrained_path, logger)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Params: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_metrics = None
        best_bal_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            if epoch % 10 == 0 or epoch == epochs:
                metrics = evaluate_batched(model, train_loader, test_loader, device)
                if metrics["balanced_accuracy"] > best_bal_acc:
                    best_bal_acc = metrics["balanced_accuracy"]
                    best_metrics = metrics.copy()
                logger.info(
                    f"  Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | "
                    f"Bal Acc: {metrics['balanced_accuracy']:.4f} | "
                    f"Kappa: {metrics['cohen_kappa']:.4f} | "
                    f"F1: {metrics['weighted_f1']:.4f}"
                )

        logger.info(f"Seed {seed} best: {best_metrics}")
        for key, val in best_metrics.items():
            all_seed_results.setdefault(key, []).append(val)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name} — Final results ({len(seeds)} seeds):")
    final = {}
    for key, vals in all_seed_results.items():
        arr = np.array(vals)
        final[key] = (arr.mean(), arr.std())
        logger.info(f"  {key}: {arr.mean():.4f} ± {arr.std():.4f}")

    return final


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description="ACTSNet EEG-FM-Bench Benchmark")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["bcic_2a", "seed_iv", "tuab", "tuev"],
        choices=list(EEGFM_DATASETS.keys()),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_time_len", type=int, default=1024,
                        help="Truncate time dim to this length (saves memory)")
    parser.add_argument("--max_train_samples", type=int, default=20000,
                        help="Subsample training set for large datasets")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/eegfm_benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("actsnet_bench", os.path.join(args.output_dir, "benchmark.log"))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"ACTSNet EEG-FM-Bench Benchmark")
    logger.info(f"Datasets: {args.datasets}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    logger.info(f"Device: {device}")
    logger.info(f"Pretrained: {args.pretrained_path or 'None (from scratch)'}")
    logger.info(f"Max time len: {args.max_time_len}, Max train samples: {args.max_train_samples}")

    all_results = {}
    for ds in args.datasets:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# Dataset: {ds}")
        logger.info(f"{'#'*60}")
        try:
            eegfm_root = get_eegfm_root(ds)
            result = run_single_dataset(
                dataset_name=ds,
                eegfm_root=eegfm_root,
                seeds=args.seeds,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
                output_dir=args.output_dir,
                logger=logger,
                max_time_len=args.max_time_len,
                max_train_samples=args.max_train_samples,
                pretrained_path=args.pretrained_path,
            )
            all_results[ds] = result
        except Exception as e:
            logger.error(f"Failed on {ds}: {e}", exc_info=True)

    # Final summary table
    logger.info(f"\n{'#'*60}")
    logger.info("# FINAL SUMMARY")
    logger.info(f"{'#'*60}")
    logger.info(f"{'Dataset':<15} {'Bal Acc':>18} {'Kappa':>18} {'W-F1':>18}")
    logger.info("-" * 72)
    for ds, metrics in all_results.items():
        ba = metrics.get("balanced_accuracy", (0, 0))
        kp = metrics.get("cohen_kappa", (0, 0))
        f1 = metrics.get("weighted_f1", (0, 0))
        logger.info(f"{ds:<15} {ba[0]:.4f}±{ba[1]:.4f}    {kp[0]:.4f}±{kp[1]:.4f}    {f1[0]:.4f}±{f1[1]:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
