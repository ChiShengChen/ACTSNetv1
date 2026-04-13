import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
import time

from .config import ACTSNetConfig
from .model import ACTSNet
from .dataset import EEGDataset


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
        probs = torch.exp(log_probs)[:, 1]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    n = len(all_labels)
    metrics = {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0
    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    # Collect all support data for prototype computation
    all_x, all_y = [], []
    for x, y in dataloader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).to(device)

    # Encode all support embeddings once
    support_emb = model.encode(all_x)

    # Evaluate in batches
    offset = 0
    for x, y in dataloader:
        batch_size = x.size(0)
        x, y = x.to(device), y.to(device)
        query_emb = support_emb[offset:offset + batch_size]
        log_probs = model.proto(query_emb, support_emb, all_y)
        loss = nn.NLLLoss()(log_probs, y)

        total_loss += loss.item() * batch_size
        preds = log_probs.argmax(dim=1)
        probs = torch.exp(log_probs)[:, 1]

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        offset += batch_size

    n = len(all_labels)
    metrics = {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.0
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

    best_val_acc = 0.0
    print(f"Training ACTSNet for {config.epochs} epochs...")
    print(f"Train: {n_train}, Val: {n_val}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d}/{config.epochs} ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"F1: {train_metrics['f1']:.4f} AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "val_metrics": val_metrics,
            }, output_dir / "best_model.pt")
            print(f"  → Saved best model (val acc: {best_val_acc:.4f})")

    # Save final model
    torch.save({
        "epoch": config.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }, output_dir / "final_model.pt")
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
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
