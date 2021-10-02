import torch
import torch.nn as nn
from torch import Tensor

_eps = 1e-7

class BezierLayer(nn.Module):
    r"""Produces the data points on the Bezier curve, together with coefficients 
        for regularization purposes.

    Args:
        in_features: size of each input sample.
        n_control_points: number of control points.
        n_data_points: number of data points to be sampled from the Bezier curve.

    Shape:
        - Input: 
            - Input Features: `(N, H)` where H = in_features.
            - Control Points: `(N, D, CP)` where D stands for the dimension of Euclidean space, 
            and CP is the number of control points. For 2D applications, D = 2.
            - Weights: `(N, 1, CP)` where CP is the number of control points. 
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """

    def __init__(self, in_features: int, n_control_points: int, n_data_points: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points-1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d([1,0], 0)
        )

    def forward(self, input: Tensor, control_points: Tensor, weights: Tensor) -> Tensor:
        cp, w = self._check_consistency(control_points, weights) # [N, d, n_cp], [N, 1, n_cp]
        bs, pv, intvls = self.generate_bernstein_polynomial(input) # [N, n_cp, n_dp]
        dp = (cp * w) @ bs / (w @ bs) # [N, d, n_dp]
        return dp, pv, intvls
    
    def _check_consistency(self, control_points: Tensor, weights: Tensor) -> Tensor:
        assert control_points.shape[-1] == self.n_control_points, 'The number of control points is not consistent.'
        assert weights.shape[-1] == self.n_control_points, 'The number of weights is not consistent.'
        assert weights.shape[1] == 1, 'There should be only one weight corresponding to each control point.'
        return control_points, weights

    def generate_bernstein_polynomial(self, input: Tensor) -> Tensor:
        intvls = self.generate_intervals(input) # [N, n_dp]
        pv = torch.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1) # [N, 1, n_dp]
        pw1 = torch.arange(0., self.n_control_points, device=input.device).view(1, -1, 1) # [1, n_cp, 1]
        pw2 = torch.flip(pw1, (1,)) # [1, n_cp, 1]
        lbs = pw1 * torch.log(pv+_eps) + pw2 * torch.log(1-pv+_eps) \
            + torch.lgamma(torch.tensor(self.n_control_points, device=input.device)+_eps).view(1, -1, 1) \
            - torch.lgamma(pw1+1+_eps) - torch.lgamma(pw2+1+_eps) # [N, n_cp, n_dp]
        bs = torch.exp(lbs) # [N, n_cp, n_dp]
        return bs, pv, intvls

    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def forward(self, input):
        return self.model(input)

class LinearCombo(_Combo):
    r"""Regular fully connected layer combo.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(alpha)
        )

class Deconv1DCombo(_Combo):
    r"""Regular deconvolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha)
        )

class Deconv2DCombo(_Combo):
    r"""Regular deconvolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

class Conv1DCombo(_Combo):
    r"""Regular convolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2, dropout=0.4
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout)
        )