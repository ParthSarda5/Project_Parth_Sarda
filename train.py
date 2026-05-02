# train.py
# ─────────────────────────────────────────────────────────────────────────────
# Training loop for the gravitational lens CNN.
#
# Main entry point (programmatic):
#   from train import train_model
#   train_model(model, num_epochs, train_loader, loss_fn, optimizer)
#
# Command-line entry point:
#   python train.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

import config


# ── per-batch helpers ─────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
        preds   = (torch.sigmoid(logits) > 0.5).long().view(-1)   # safe for batch_size=1
        correct += (preds == labels.long().view(-1)).sum().item()
        total   += len(imgs)
    return total_loss / total, correct / total


@torch.no_grad()
def _eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs    = imgs.to(device)
        labels_t = labels.float().unsqueeze(1).to(device)
        logits  = model(imgs)
        loss    = loss_fn(logits, labels_t)
        probs   = torch.sigmoid(logits).view(-1).cpu()   # safe for batch_size=1
        preds   = (probs > 0.5).long()
        total_loss += loss.item() * len(imgs)
        correct    += (preds == labels).sum().item()
        total      += len(imgs)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
    auc = roc_auc_score(all_labels, all_probs)
    return total_loss / total, correct / total, auc


# ── main training function ────────────────────────────────────────────────────

def train_model(model, num_epochs, train_loader, loss_fn, optimizer,
                val_loader=None, device=None, verbose=True):
    """
    Run the full training loop with cosine LR annealing and early stopping.

    Args:
        model        : nn.Module — the GravLensNet (or any binary classifier)
        num_epochs   : int       — maximum training epochs
        train_loader : DataLoader — training split
        loss_fn      : loss function (BCEWithLogitsLoss recommended)
        optimizer    : torch optimiser (Adam recommended)
        val_loader   : DataLoader — validation split for AUC tracking / early stopping.
                       If None, no early stopping is applied.
        device       : torch.device — defaults to CUDA if available, else CPU.
        verbose      : bool — print per-epoch metrics.

    Returns:
        history (dict): keys train_loss, train_acc, val_loss, val_acc, val_auc
                        each a list of per-epoch values.

    Side effect:
        Saves the best checkpoint (by val AUC) to config.weights_path.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    history = {k: [] for k in
               ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_auc']}
    best_auc         = 0.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)

        if val_loader is not None:
            vl_loss, vl_acc, vl_auc = _eval_epoch(model, val_loader, loss_fn, device)
            history['val_loss'].append(vl_loss)
            history['val_acc'].append(vl_acc)
            history['val_auc'].append(vl_auc)

            if verbose:
                marker = ' ←' if vl_auc > best_auc else ''
                print(f"Epoch {epoch:03d}/{num_epochs}  "
                      f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                      f"val loss={vl_loss:.4f} acc={vl_acc:.4f} AUC={vl_auc:.4f}{marker}")

            if vl_auc > best_auc:
                best_auc = vl_auc
                patience_counter = 0
                torch.save(model.state_dict(), config.weights_path)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no AUC improvement for {config.patience} epochs).")
                    break
        else:
            # No validation — always save latest
            torch.save(model.state_dict(), config.weights_path)
            if verbose:
                print(f"Epoch {epoch:03d}/{num_epochs}  "
                      f"train loss={tr_loss:.4f} acc={tr_acc:.4f}")

        scheduler.step()

    if verbose:
        print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
        print(f"Weights saved → {config.weights_path}")

    return history


# ── command-line entry point ───────────────────────────────────────────────────

if __name__ == '__main__':
    import torch.nn as nn
    from torch.optim import Adam
    from model   import GravLensNet
    from dataset import get_dataloader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, val_loader, _ = get_dataloader()

    model     = GravLensNet(pretrained=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate,
                     weight_decay=config.weight_decay)
    loss_fn   = nn.BCEWithLogitsLoss()

    train_model(model, config.epochs, train_loader, loss_fn, optimizer,
                val_loader=val_loader, device=device)
