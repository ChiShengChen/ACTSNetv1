import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
import argparse
import json

from .config import ACTSNetConfig
from .model import ACTSNet
from .dataset import EEGDataset


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()

    # Collect all data
    all_x, all_y = [], []
    for x, y in dataloader:
        all_x.append(x)
        all_y.append(y)
    all_x = torch.cat(all_x, dim=0).to(device)
    all_y = torch.cat(all_y, dim=0).to(device)

    # Compute embeddings and prototypes on full set
    embeddings = model.encode(all_x)
    log_probs = model.proto(embeddings, embeddings, all_y)

    probs = torch.exp(log_probs)
    preds = log_probs.argmax(dim=1)

    y_true = all_y.cpu().numpy()
    y_pred = preds.cpu().numpy()
    y_prob = probs[:, 1].cpu().numpy()

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    try:
        results["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results["auc_roc"] = 0.0

    # Per-sample predictions
    per_sample = []
    for i in range(len(y_true)):
        per_sample.append({
            "index": i,
            "true_label": int(y_true[i]),
            "predicted_label": int(y_pred[i]),
            "prob_responder": float(y_prob[i]),
        })
    results["per_sample"] = per_sample

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACTSNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = ACTSNet(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load test data
    dataset = EEGDataset(data_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    results = evaluate_model(model, dataloader, device)

    # Print results
    print("=" * 50)
    print("ACTSNet Evaluation Results")
    print("=" * 50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    print(f"AUC-ROC:   {results['auc_roc']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
