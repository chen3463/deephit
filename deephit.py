import torch
import torch.nn as nn
import torch.nn.functional as F

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
