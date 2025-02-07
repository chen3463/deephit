import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

# Define the DeepHit model
class DeepHit(nn.Module):
    def __init__(self, input_dim, num_events, hidden_dim=128, num_layers=3):
        super(DeepHit, self).__init__()
        self.num_events = num_events

        # Shared subnetwork
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.shared_net = nn.Sequential(*layers)

        # Cause-specific subnetworks
        self.cause_specific_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Predict log-risk score
            ) for _ in range(num_events)
        ])

    def forward(self, x):
        shared_output = self.shared_net(x)
        cause_specific_outputs = [net(shared_output) for net in self.cause_specific_nets]
        return torch.cat(cause_specific_outputs, dim=1)  # (batch_size, num_events)

# Loss function
def deephit_loss(preds, targets, masks, alpha=0.5):
    """
    Compute the DeepHit loss.
    """
    preds = F.log_softmax(preds, dim=1)  # Log-softmax for numerical stability

    # Negative log-likelihood loss
    nll_loss = -torch.mean(torch.sum(masks * preds, dim=1))

    # Ranking loss (pairwise)
    ranking_loss = torch.tensor(0.0)
    n = preds.shape[0]
    if n > 1:
        pairwise_diffs = (targets.unsqueeze(1) - targets.unsqueeze(0))  # (n, n)
        valid_pairs = pairwise_diffs > 0  # Only consider pairs where targets[i] < targets[j]

        if valid_pairs.sum() > 0:
            risk_diffs = preds.unsqueeze(1) - preds.unsqueeze(0)  # (n, n, num_events)
            ranking_loss = torch.mean(F.softplus(risk_diffs[valid_pairs]))  # Smooth ranking loss

    # Combined loss
    loss = nll_loss + alpha * ranking_loss
    return loss

# C-index calculation
def calculate_c_index(preds, targets, masks):
    """
    Calculate the concordance index (C-index) for survival predictions.
    """
    preds = preds.detach().numpy()
    targets = targets.numpy()
    masks = masks.numpy()

    # Compute C-index for each event type
    c_indices = []
    for i in range(masks.shape[1]):
        event_indicator = masks[:, i].astype(bool)
        if event_indicator.sum() > 1:  # Ensure at least two events occurred
            c_index = concordance_index_censored(event_indicator, targets, preds[:, i])[0]
            c_indices.append(c_index)

    return np.mean(c_indices) if c_indices else 0.0  # Return 0 if no valid pairs

# Training function
def train_deephit(model, dataloader, optimizer, num_epochs=10, alpha=0.5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        c_index_scores = []

        for batch in dataloader:
            x, times, event_indicators = batch  # No GPU usage
            optimizer.zero_grad()
            preds = model(x)
            loss = deephit_loss(preds, times, event_indicators, alpha)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate C-index for the batch
            c_index = calculate_c_index(preds, times, event_indicators)
            c_index_scores.append(c_index)

        # Average C-index over all batches
        avg_c_index = np.mean(c_index_scores)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}, C-index: {avg_c_index:.4f}")

# Test function
def test_deephit(model, dataloader):
    model.eval()
    c_index_scores = []

    with torch.no_grad():
        for batch in dataloader:
            x, times, event_indicators = batch  # No GPU usage
            preds = model(x)

            # Calculate C-index for the batch
            c_index = calculate_c_index(preds, times, event_indicators)
            c_index_scores.append(c_index)

    # Average C-index over all batches
    avg_c_index = np.mean(c_index_scores)
    print(f"Test C-index: {avg_c_index:.4f}")
