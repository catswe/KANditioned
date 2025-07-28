import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_control_points: int = 1000, eps: float = 1e-6):
        super(KANLayer, self).__init__()
        self.register_buffer("mins", None)
        self.register_buffer("maxs", None)

        self.in_features = in_features
        self.out_features = out_features
        
        self.num_control_points = num_control_points
        self.eps = eps

        self.r_weight = nn.Parameter(torch.zeros(in_features, num_control_points, out_features)) # (in_features, num_control_points, out_features)
        self.l_weight = nn.Parameter(torch.zeros(in_features, num_control_points, out_features)) # (in_features, num_control_points, out_features)

        self.register_buffer("local_bias", torch.arange(num_control_points).view(1, num_control_points, 1)) # (1, num_control_points, 1)
        self.register_buffer("feature_offset", torch.arange(in_features).view(1, -1) * self.num_control_points) # (1, in_features)

    def forward(self, x):
        # x: (batch_size, in_features)

        if self.training or self.mins is None or self.maxs is None:
            self.mins = x.amin(dim=0, keepdim=True)  # (1, in_features)
            self.maxs = x.amax(dim=0, keepdim=True)  # (1, in_features)

        x = (x - self.mins) / (self.maxs - self.mins + self.eps) * (self.num_control_points - 1)  # (batch_size, in_features)

        lower_indices_float = x.floor().clamp(0, self.num_control_points - 2) # (batch_size, in_features)
        lower_indices = lower_indices_float.long() + self.feature_offset # (batch_size, in_features)

        indices = torch.stack((lower_indices, lower_indices + 1), dim=-1) # (batch_size, in_features, 2)
        vals = F.embedding(indices, self.get_interp_tensor()) # (batch_size, in_features, 2, out_features)

        lower_val, upper_val = vals.unbind(dim=2) # each: (batch_size, in_features, out_features)
        return torch.lerp(lower_val, upper_val, (x - lower_indices_float).unsqueeze(-1)).sum(dim=1) # (batch_size, out_features)

    def get_interp_tensor(self):
        cs_r_weight = torch.cumsum(self.r_weight, dim=1) # (in_features, num_control_points, out_features)
        cs_l_weight = torch.cumsum(self.l_weight, dim=1) # (in_features, num_control_points, out_features)

        cs_r_weight_bias_prod = torch.cumsum(self.r_weight * self.local_bias, dim=1) # type: ignore (in_features, num_control_points, out_features)
        cs_l_weight_bias_prod = torch.cumsum(self.l_weight * self.local_bias, dim=1) # type: ignore (in_features, num_control_points, out_features)

        r_interp = (self.local_bias * cs_r_weight - cs_r_weight_bias_prod) # type: ignore (in_features, num_control_points, out_features)
        l_interp = (cs_l_weight_bias_prod[:, -1:, :] - cs_l_weight_bias_prod) - self.local_bias * (cs_l_weight[:, -1:, :] - cs_l_weight) # type: ignore (in_features, num_control_points, out_features)
        return (r_interp + l_interp).view(-1, self.out_features) # (in_features * num_control_points, out_features)

    def visualize_all_mappings(self: KANLayer):
        interp_tensor = self.get_interp_tensor().detach().cpu()
        interp_tensor = interp_tensor.view(self.in_features, self.num_control_points, self.out_features)

        fig, axes = plt.subplots(self.in_features, self.out_features, figsize=(4 * self.out_features, 3 * self.in_features))

        for i in range(self.in_features):
            for j in range(self.out_features):
                ax = axes[i, j] if self.in_features > 1 and self.out_features > 1 else axes[max(i, j)]
                ax.plot(interp_tensor[i, :, j])
                ax.set_title(f'In {i} → Out {j}')
                ax.set_xlabel('Control Points')
                ax.set_ylabel('Value')
                ax.grid(True)
        
        plt.tight_layout()
        plt.show()
