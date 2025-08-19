# Copyright 2025 Bui, William

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, init: str, num_control_points: int):
        super().__init__()
        self.in_features, self.out_features, self.num_control_points = in_features, out_features, num_control_points
        self.kan_weight = nn.Parameter(torch.zeros(in_features, num_control_points, out_features))

        self.register_buffer("local_bias", torch.arange(num_control_points).view(1, -1, 1))
        self.register_buffer("feature_offset", torch.arange(in_features).view(1, -1) * num_control_points)

        centered_bias = self.local_bias.float() - (num_control_points - 1) / 2.0 # type: ignore

        if init == 'random_normal':
            slopes = torch.randn(in_features, out_features, device=self.kan_weight.device)
            slopes /= slopes.norm(dim=0, keepdim=True).clamp_min(1e-12)
        elif init == 'identity':
            if in_features != out_features:
                raise ValueError("'identity' init requires in_features == out_features.")
            slopes = torch.eye(in_features, device=self.kan_weight.device)
        elif init != 'zero':
            raise ValueError("init must be 'random_normal', 'identity', or 'zero'.")
        else:
            return

        with torch.no_grad():
            self.kan_weight.copy_(centered_bias * slopes.unsqueeze(1))

    def forward(self, x):
        # x: (batch_size, in_features)
        
        lower_indices_float = x.floor().clamp(0, self.num_control_points - 2) # (batch_size, in_features)
        lower_indices = lower_indices_float.long() + self.feature_offset # (batch_size, in_features)

        indices = torch.stack((lower_indices, lower_indices + 1), dim=-1) # (batch_size, in_features, 2)
        vals = F.embedding(indices, self.kan_weight.view(-1, self.out_features)) # (batch_size, in_features, 2, out_features)

        lower_val, upper_val = vals.unbind(dim=2) # each: (batch_size, in_features, out_features)
        return torch.lerp(lower_val, upper_val, (x - lower_indices_float).unsqueeze(-1)).sum(dim=1) # (batch_size, out_features)

    def visualize_all_mappings(self, save_path=None):
        interp_tensor = self.kan_weight.detach().cpu().view(self.in_features, self.num_control_points, self.out_features)

        fig, axes = plt.subplots(
            self.in_features,
            self.out_features,
            figsize=(4 * self.out_features, 3 * self.in_features)
        )

        axes = np.array(axes, dtype=object).reshape(self.in_features, self.out_features)

        for i in range(self.in_features):
            for j in range(self.out_features):
                ax = axes[i, j]
                ax.plot(interp_tensor[i, :, j])
                ax.set_title(f'In {i} → Out {j}')
                ax.set_xlabel('Control Points')
                ax.set_ylabel('Value')
                ax.grid(True)

        fig.suptitle("KAN Layer Mappings", fontsize=16, y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()