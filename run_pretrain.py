"""ACTSNet self-supervised pretraining on EEG-FM-Bench data.

Usage:
    python run_pretrain.py [--epochs 50] [--batch_size 64]
"""
from __future__ import annotations

import argparse
import os
import sys
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pyarrow.ipc as ipc

sys.path.insert(0, os.path.dirname(__file__))
from actsnet.config import ACTSNetConfig
from actsnet.pretrain import ACTSNetPretrainEncoder, PretrainEEGDataset, pretrain


EEGFM_ROOT_CANDIDATES = [
    "/media/meow/Transcend/time_series_benchmark/eegfm_data_cache",
    "/media/meow/Elements/EEG-FM-Bench-data/processed/fs_256",
    "/media/meow/Transcend/time_series_benchmark/EEG-FM-Bench/assets/data/processed/fs_256",
]


def get_eegfm_root(dataset_name: str):
    for p in EEGFM_ROOT_CANDIDATES:
        if os.path.isdir(os.path.join(p, dataset_name)):
            return p
    raise FileNotFoundError(f"Data root not found for {dataset_name}")


def load_arrow_data(root, dataset, version, split, max_samples=None, max_time_len=1024):
    """Load arrow shards, return numpy array (N, n_ch, T)."""
    base_dir = os.path.join(root, dataset, "finetune", version)
    if not os.path.isdir(base_dir):
        # Try pretrain split
        base_dir = os.path.join(root, dataset, "pretrain", version)

    prefix = f"{dataset}-{split}-"
    shards = sorted([
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(prefix) and f.endswith(".arrow")
    ])
    if not shards:
        raise FileNotFoundError(f"No shards for {dataset}/{split}")

    # First pass: count
    total = 0
    for s in shards:
        reader = ipc.open_stream(s)
        total += len(reader.read_all())

    # Subsample indices
    if max_samples and total > max_samples:
        rng = np.random.RandomState(42)
        keep = set(rng.choice(total, max_samples, replace=False).tolist())
    else:
        keep = None

    # Second pass: load
    all_data = []
    idx = 0
    for s in shards:
        reader = ipc.open_stream(s)
        table = reader.read_all()
        for i in range(len(table)):
            if keep is None or idx in keep:
                d = np.array(table.column("data")[i].as_py(), dtype=np.float32)
                if max_time_len and d.shape[-1] > max_time_len:
                    d = d[:, :max_time_len]
                all_data.append(d)
            idx += 1

    return np.stack(all_data, axis=0)


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
    parser = argparse.ArgumentParser(description="ACTSNet Self-Supervised Pretraining")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_channels", type=int, default=64,
                        help="Max channels (pad/truncate all inputs to this)")
    parser.add_argument("--max_time_len", type=int, default=1024)
    parser.add_argument("--max_samples_per_dataset", type=int, default=30000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--prototype_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("actsnet_pretrain", os.path.join(args.output_dir, "pretrain.log"))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load pretrain data from multiple datasets ──
    pretrain_datasets_info = [
        # (dataset_name, version, split, max_samples)
        ("tuab", "3.0.1", "train", args.max_samples_per_dataset),
        ("tuev", "2.0.0", "train", args.max_samples_per_dataset),
        ("bcic_2a", "1.0.0", "train", None),  # small, load all
        ("seed_iv", "1.0.0", "train", args.max_samples_per_dataset),  # cached on Transcend (bad sector)
        ("siena_scalp", "1.0.0", "train", args.max_samples_per_dataset),  # cached on Transcend (bad sector)
    ]

    datasets = []
    total_samples = 0
    for ds_name, version, split, max_s in pretrain_datasets_info:
        try:
            root = get_eegfm_root(ds_name)
            logger.info(f"Loading {ds_name}/{split} from {root} ...")
            data = load_arrow_data(root, ds_name, version, split,
                                   max_samples=max_s, max_time_len=args.max_time_len)
            ds = PretrainEEGDataset(data, max_channels=args.max_channels,
                                    max_time_len=args.max_time_len)
            datasets.append(ds)
            total_samples += len(ds)
            logger.info(f"  {ds_name}: {len(ds)} samples, shape={data.shape}")
        except Exception as e:
            logger.warning(f"  Skipping {ds_name}: {e}")

    if not datasets:
        logger.error("No datasets loaded!")
        return

    combined = ConcatDataset(datasets)
    dataloader = DataLoader(combined, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, num_workers=4)
    logger.info(f"Total pretrain samples: {total_samples}, Batches: {len(dataloader)}")

    # ── Build model ──
    config = ACTSNetConfig(
        n_channels=args.max_channels,
        n_classes=2,  # placeholder, not used in pretrain
        n_timesteps=args.max_time_len,
        n_groups=min(3, args.max_channels),
        prototype_dim=args.prototype_dim,
        seed=args.seed,
    )

    model = ACTSNetPretrainEncoder(config, max_channels=args.max_channels)
    logger.info(f"Config: max_ch={args.max_channels}, T={args.max_time_len}, "
                f"proto_dim={args.prototype_dim}")

    # ── Pretrain ──
    pretrain(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        logger=logger,
        checkpoint_dir=args.output_dir,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
