import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

# Load the CSV data
data = pd.read_csv("/content/sample_data/synthetic_comprisk.csv")

# Extract features, time, and labels
features = data.iloc[:, 4:].values  # feature1 to feature12
times = data["time"].values
labels = data["label"].values

# Convert to PyTorch tensors (CPU version)
features = torch.tensor(features, dtype=torch.float32)
times = torch.tensor(times, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.int64)  # Ensure labels are integers

# Number of competing events
num_events = 2

# Create event indicators per event type
event_indicators = torch.zeros((len(labels), num_events))
for i in range(num_events):
    event_indicators[:, i] = (labels == (i + 1)).float()

# Combine into a dataset
dataset = TensorDataset(features, times, event_indicators)

# Split into training and testing sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize the model
input_dim = features.shape[1]
model = DeepHit(input_dim, num_events)  # No CUDA

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_deephit(model, train_loader, optimizer, num_epochs=10)

# Test the model
test_deephit(model, test_loader)
