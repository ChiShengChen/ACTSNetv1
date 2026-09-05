"""DEPRECATED for cross-subject evaluation -- do not use to reproduce any published result.

This script partitions the data with `random_split` over *windows*, so segments from
the same subject land on both sides of the split. Its `evaluate()` has been corrected to
use the training set as the prototypical support (it no longer scores the validation set
against its own labels), but the window-level split remains, and the script performs no
subject-grouped cross-validation.

It is retained as a single-file training demo. The protocol that produced the published
results lives in https://github.com/ChiShengChen/ACTSNet-EEG-sample-efficiency
(run_loso.py / run_loso_baseline.py).
"""
import warnings

warnings.warn(
    "actsnet.train is deprecated for cross-subject work: it splits over windows, not "
    "subjects. Use run_loso.py in ChiShengChen/ACTSNet-EEG-sample-efficiency.",
    DeprecationWarning, stacklevel=2)

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
import argparse
import time

from .config import ACTSNetConfig
from .model import ACTSNet
from .dataset import EEGDataset


def safe_auc(labels, probs):
    """Multi-class-safe AUC over a full (N, n_classes) probability matrix.

    Binary -> AUC on the positive-class column; multi-class -> one-vs-rest
    macro AUC. Returns 0.0 when AUC is undefined (e.g. only one class present
    in the query labels), mirroring the previous behaviour.
    """
    labels = np.asarray(labels)
    n_classes = probs.shape[1]
    if len(np.unique(labels)) < 2:
        return 0.0
    try:
        if n_classes == 2:
            return roc_auc_score(labels, probs[:, 1])
        return roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
            labels=np.arange(n_classes),
        )
    except ValueError:
        return 0.0


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Use batch itself as support set
        log_probs = model(x, support_x=x, support_labels=y)
        loss = nn.NLLLoss()(log_probs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = log_probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.append(torch.exp(log_probs).detach().cpu().numpy())

    n = len(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)  # (N, n_classes)
    metrics = {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "auc": safe_auc(all_labels, all_probs),
    }
    return metrics


@torch.no_grad()
def evaluate(model, train_loader, val_loader, device, max_support=5000):
    """Inductive evaluation: build class prototypes from the TRAIN set
    (train-as-support), then classify the val/query set against them.

    This avoids the transductive label leakage of using the val set as its
    own support: prototypes here only ever see TRAIN labels, matching the
    protocol in run_eegfm_benchmark.evaluate_batched. For very large train
    sets the support is capped at `max_support` samples.
    """
    model.eval()

    # --- Support set from TRAIN (only train labels are ever used here) ---
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

    # Encode support in chunks (memory-safe on large train sets)
    chunk_size = 256
    support_embs = []
    for i in range(0, len(support_x), chunk_size):
        support_embs.append(model.encode(support_x[i:i + chunk_size]))
    support_emb = torch.cat(support_embs, dim=0)

    # --- Query = val set ---
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        query_emb = model.encode(x)
        log_probs = model.proto(query_emb, support_emb, support_y)
        loss = nn.NLLLoss()(log_probs, y)

        total_loss += loss.item() * x.size(0)
        all_preds.extend(log_probs.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.append(torch.exp(log_probs).cpu().numpy())

    n = len(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)  # (N, n_classes)
    metrics = {
        "loss": total_loss / max(1, n),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "auc": safe_auc(all_labels, all_probs),
    }
    return metrics


def train(config: ACTSNetConfig, data_dir: str, output_dir: str = "checkpoints"):
    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and split
    dataset = EEGDataset(data_dir=data_dir)
    n_train = int(len(dataset) * config.train_ratio)
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed)
    )

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    # Infer timesteps from data
    sample_x, _ = dataset[0]
    config.n_timesteps = sample_x.shape[-1]

    model = ACTSNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_balacc = 0.0
    print(f"Training ACTSNet for {config.epochs} epochs...")
    print(f"Train: {n_train}, Val: {n_val}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, train_loader, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{config.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"BalAcc: {train_metrics['balanced_accuracy']:.4f} "
            f"F1: {train_metrics['f1']:.4f} AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"BalAcc: {val_metrics['balanced_accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["balanced_accuracy"] > best_val_balacc:
            best_val_balacc = val_metrics["balanced_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model (val balanced acc: {best_val_balacc:.4f})")

    # Save final model
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, output_dir / "final_model.pt")
    print(f"Training complete. Best val balanced accuracy: {best_val_balacc:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ACTSNet")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory with data.npy and labels.npy")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_groups", type=int, default=3)
    parser.add_argument("--prototype_dim", type=int, default=128)
    parser.add_argument("--latent_dim_u", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = ACTSNetConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_groups=args.n_groups,
        prototype_dim=args.prototype_dim,
        latent_dim_u=args.latent_dim_u,
        dropout=args.dropout,
        seed=args.seed,
        device=args.device,
    )
    train(config, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
