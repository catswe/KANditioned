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

def _ensure_positive_int(name: str, value: int): # type: ignore
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value}")

def _ensure_in_set(name: str, value: str, allowed: set[str]):
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value}")

class KANLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            init: str,
            num_control_points: int = 32,
            spline_width: float = 4.0,
            variant: str="B-spline",
            impl: str="embedding_bag",
    ) -> None:
        super().__init__()

        _ensure_positive_int("in_features", in_features)
        _ensure_positive_int("out_features", out_features)
        _ensure_in_set("init", init, {"random_normal", "identity", "zero"})
        if init == 'identity' and in_features != out_features:
            raise ValueError("'identity' init requires in_features == out_features.")
        _ensure_positive_int("num_control_points", num_control_points)
        if not (isinstance(spline_width, (int, float)) and spline_width > 0):
            raise ValueError(f"spline_width must be > 0, got {spline_width}")
        _ensure_in_set("variant", variant, {"B-spline", "parallel_scan", "DCT"})
        if variant == "DCT":
            raise NotImplementedError("DCT variant is not yet implemented.")
        _ensure_in_set("impl", impl, {"embedding_bag", "embedding"})

        self.in_features = in_features
        self.out_features = out_features
        self.init = init
        self.num_control_points = num_control_points
        self.spline_width = spline_width
        self.variant = variant
        self.impl = impl

        self.register_buffer("local_bias", torch.arange(num_control_points).view(1, -1, 1)) # TODO:fix this so that it's only registered for parallel scan variant
        self.register_buffer("feature_offset", torch.arange(in_features).view(1, -1) * num_control_points)
        self.init_tensor()

    def get_interp_tensor(self):
        if self.variant == "B-spline":
            return self.kan_weight.view(-1, self.out_features)
        elif self.variant == "parallel_scan":
            cs_r_weight = torch.cumsum(self.r_weight, dim=1) # (in_features, num_control_points, out_features)
            cs_l_weight = torch.cumsum(self.l_weight, dim=1) # (in_features, num_control_points, out_features)

            cs_r_weight_bias_prod = torch.cumsum(self.r_weight * self.local_bias, dim=1) # (in_features, num_control_points, out_features)
            cs_l_weight_bias_prod = torch.cumsum(self.l_weight * self.local_bias, dim=1) # (in_features, num_control_points, out_features)

            r_interp = (self.local_bias * cs_r_weight - cs_r_weight_bias_prod) # type: ignore (in_features, num_control_points, out_features)
            l_interp = (cs_l_weight_bias_prod[:, -1:, :] - cs_l_weight_bias_prod) - self.local_bias * (cs_l_weight[:, -1:, :] - cs_l_weight) # type: ignore (in_features, num_control_points, out_features)
            return (r_interp + l_interp).view(-1, self.out_features) # (in_features * num_control_points, out_features)
        elif self.variant == "DCT":
            raise NotImplementedError("DCT variant is not yet implemented.")

        raise ValueError("Variant must be either 'B-spline' or 'parallel_scan'")

    def init_tensor(self):
        if self.variant == "B-spline":
            centered_bias = self.local_bias.float() - (self.num_control_points - 1) / 2.0

            if self.init == 'random_normal':
                slopes = F.normalize(torch.randn(self.in_features, self.out_features), dim=0)
            elif self.init == 'identity':
                slopes = torch.eye(self.in_features)
            elif self.init == 'zero':
                slopes = torch.zeros(self.in_features, self.out_features)

            self.kan_weight = nn.Parameter(centered_bias * slopes.unsqueeze(1))

        elif self.variant == "parallel_scan":
            self.r_weight = nn.Parameter(torch.zeros(self.in_features, self.num_control_points, self.out_features))
            self.l_weight = nn.Parameter(torch.zeros(self.in_features, self.num_control_points, self.out_features))
            print("Parallel scan custom init not yet supported. Please initialize weights manually.")

    def forward(self, x):
        # x: (batch_size, in_features)
        x = (x + self.spline_width / 2) * (self.num_control_points - 1) / self.spline_width

        lower_indices_float = x.floor().clamp(0, self.num_control_points - 2) # (batch_size, in_features)
        lower_indices = lower_indices_float.long() + self.feature_offset # (batch_size, in_features)

        if self.impl == "embedding_bag":
            t = x - lower_indices_float # (batch_size, in_features)
            return F.embedding_bag(
                torch.stack((lower_indices, lower_indices + 1), dim=2).reshape(x.size(0), -1), # (batch_size, in_features * 2)
                self.get_interp_tensor(), # (in_features * num_control_points, out_features)
                per_sample_weights=torch.stack((1.0 - t, t), dim=2).reshape(x.size(0), -1), # (batch_size, in_features * 2)
                mode='sum',
            ) # (batch_size, out_features)
        elif self.impl == "embedding":
            indices = torch.stack((lower_indices, lower_indices + 1), dim=-1) # (batch_size, in_features, 2)
            vals = F.embedding(indices, self.get_interp_tensor()) # (batch_size, in_features, 2, out_features)

            lower_val, upper_val = vals.unbind(dim=2) # each: (batch_size, in_features, out_features)
            return torch.lerp(lower_val, upper_val, (x - lower_indices_float).unsqueeze(-1)).sum(dim=1) # (batch_size, out_features)

    def visualize_all_mappings(self, save_path=None):
        interp_tensor = self.get_interp_tensor().detach().cpu().view(self.in_features, self.num_control_points, self.out_features)

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