import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, List, Union


class RescaleHeadWrapper(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model    # This is the model that we want to wrap
        self.year_nn = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()    # TODO: Optimal choice?
        )

    def forward(self, x, year):
        out = self.base_model.forward(x)

        # Process coordinates
        rescale_factor = self.year_nn(year).squeeze(-1)
        rescale_factor = 0.8 + 0.4 * rescale_factor  # Rescale to [0.8, 1.2]

        # Apply rescaling
        out = out * rescale_factor.view(-1, 1, 1, 1)
        return out
