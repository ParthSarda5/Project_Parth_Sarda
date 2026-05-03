# interface.py
# ─────────────────────────────────────────────────────────────────────────────
# Standardised interface file for the grading program.
# ─────────────────────────────────────────────────────────────────────────────

# The model class
from model import GravLensNet as TheModel

# The function inside train.py that runs the training loop
from train import train_model as the_trainer

# The function inside predict.py that generates inference on a list of image paths
from predict import predict_lenses as the_predictor

# The custom Dataset class
from dataset import LensDataset as TheDataset

# The factory function that returns (train_loader, val_loader, test_loader)
from dataset import get_dataloader as the_dataloader

# Hyperparameters from config.py
from config import batch_size as the_batch_size
from config import epochs     as total_epochs


# ─────────────────────────────────────────────────────────────────────────────
# Quick usage reference for the grading script:
#
#   model        = TheModel(pretrained=True)
#
#   train_loader, val_loader, test_loader = the_dataloader()
#
#   import torch, torch.nn as nn
#   from torch.optim import Adam
#   optimizer = Adam(model.parameters(), lr=1e-4)
#   loss_fn   = nn.BCEWithLogitsLoss()
#   history   = the_trainer(model, total_epochs, train_loader, loss_fn, optimizer,
#                           val_loader=val_loader)
#
#   results = the_predictor(['data/img01.npy', 'data/img02.npy'])
#   # results is a list of dicts: {'path', 'prediction', 'probability'}
# ─────────────────────────────────────────────────────────────────────────────
