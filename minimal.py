import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class KANLayer(nn.Linear):
    feature_offset: Tensor
    """Zero-centered spline version. Minimal version with no spline visualization, custom init, or error checking.
    TODO: Implement custom backward pass to reduce memory pressure and 2:4 sparse semi-structured tensor."""

    def __init__(self, in_features: int, out_features: int, num_control_points: int = 5, spline_width: float = 2.0, bias: bool=True):
        super().__init__(in_features, out_features, bias=bias)
        if num_control_points % 2 == 0 or num_control_points < 3:
            raise ValueError("num_control_points must be an odd value greater than or equal to 3.")

        self.num_control_points = num_control_points
        self.spline_width = spline_width

        self.register_buffer("feature_offset", torch.arange(in_features).view(1, -1) * (num_control_points - 1))
        self.weight = nn.Parameter(torch.zeros(out_features, in_features * (num_control_points - 1)))

    def forward(self, input):
        x = (input + self.spline_width / 2) * (self.num_control_points - 1) / self.spline_width

        idx_float = x.floor().clamp_(0, self.num_control_points - 2)
        interpolation_weight = x - idx_float

        dense_matrix = torch.zeros(x.size(0), self.weight.size(1), device=x.device)

        mid_index = self.num_control_points // 2
        idx = idx_float.long() - (idx_float >= mid_index).long() + self.feature_offset

        dense_matrix.scatter_(1, idx, (1 - interpolation_weight) * (idx_float != mid_index))
        dense_matrix.scatter_(1, idx + 1, interpolation_weight * (idx_float != (mid_index - 1)))

        return F.linear(dense_matrix, self.weight, self.bias)