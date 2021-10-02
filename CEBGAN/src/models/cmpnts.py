import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from . import layers
from .utils import first_element

class MLP(nn.Module):
    """Regular fully connected network generating features.
    
    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        layer_width: The widths of the hidden layers.
        combo: The layer combination to be stacked up.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output: `(N, H_out)` where H_out = out_features.
    """
    def __init__(
        self, in_features: int, out_features:int, layer_width: list, 
        combo = layers.LinearCombo
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.model = self._build_model(layer_width, combo)
    
    def forward(self, input):
        return self.model(input)

    def _build_model(self, layer_width, combo):
        model = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(zip(
            [self.in_features] + layer_width, 
            layer_width + [self.out_features]
            )):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        return model

class Conv1DNetwork(nn.Module):
    """The 1D convolutional front end.

    Args:
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        conv_channels: The number of channels of each conv1d layer.

    Shape:
        - Input: `(N, C, H_in)` where C = in_channel and H_in = in_features.
        - Output: `(N, H_out)` where H_out is calculated based on in_features.
    """
    def __init__(
        self, in_channels: int, in_features: int, conv_channels: list, 
        combo = layers.Conv1DCombo
        ):
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.m_features = self._calculate_m_features(conv_channels)
        self.conv = self._build_conv(conv_channels, combo)

    def forward(self, input):
        return self.conv(input)
    
    def _calculate_m_features(self, channels):
        n_l = len(channels)
        m_features = self.in_features // (2 ** n_l) * channels[-1]
        return m_features

    def _build_conv(self, channels, combo):
        conv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(
            [self.in_channels] + channels[:-1], channels
            )):
            conv.add_module(
                str(idx), combo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
            conv.add_module(str(idx+1), nn.Flatten())
        return conv

class CPWGenerator(nn.Module):
    """Generate given number of control points and weights for Bezier Layer.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output. 
            Should be even.
        dense_layers: The widths of the hidden layers of the MLP connecting 
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Control Points: `(N, 2, H_out)` where H_out = n_control_points.
            - Weights: `(N, 1, H_out)` where H_out = n_control_points.
    """
    def __init__(
        self, in_features: int, n_control_points: int,
        dense_layers: list = [1024,],
        deconv_channels: list = [96*8, 96*4, 96*2, 96],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self._calculate_parameters(n_control_points, deconv_channels)

        self.dense = MLP(in_features, self.in_chnl*self.in_width, dense_layers)
        self.deconv = self._build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 2, 1), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 1, 1), nn.Sigmoid())
    
    def forward(self, input):
        x = self.deconv(self.dense(input).view(-1, self.in_chnl, self.in_width))
        cp = self.cp_gen(x)
        w = self.w_gen(x)
        return cp, w
    
    def _calculate_parameters(self, n_control_points, channels):
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2 ** n_l)
        assert in_width >= 4, 'Too many deconvolutional layers ({}) for the {} control points.'\
            .format(n_l, self.n_control_points)
        return in_chnl, in_width
    
    def _build_deconv(self, channels):
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(channels[:-1], channels[1:])):
            deconv.add_module(
                str(idx), layers.Deconv1DCombo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
        return deconv

class BezierGenerator(nn.Module):
    """Generator for BezierGAN alike projects.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
        n_data_points: The number of data points to output.
        m_features: The number of intermediate features for generating intervals.
        feature_gen_layer: The widths of hidden layers for generating intermediate features.
        dense_layers: The widths of the hidden layers of the MLP connecting 
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Control Points: `(N, 2, CP)` where CP = n_control_points.
            - Weights: `(N, 1, CP)` where CP = n_control_points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """
    def __init__(
        self, in_features: int, n_control_points: int, n_data_points: int, 
        m_features: int = 256,
        feature_gen_layers: list = [1024,],
        dense_layers: list = [1024,],
        deconv_channels: list = [96*8, 96*4, 96*2, 96],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.feature_generator = MLP(in_features, m_features, feature_gen_layers)
        self.cpw_generator = CPWGenerator(in_features, n_control_points, dense_layers, deconv_channels)
        self.bezier_layer = layers.BezierLayer(m_features, n_control_points, n_data_points)
    
    def forward(self, input):
        features = self.feature_generator(input)
        cp, w = self.cpw_generator(input)
        dp, pv, intvls = self.bezier_layer(features, cp, w)
        return dp, cp, w, pv, intvls
    
    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

class Critics1D(Conv1DNetwork):
    """Regular discriminator for GANs.

    Args: 
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        n_critics: The number of critics.
        conv_channels: The number of channels of each conv1d layer.
        crt_layers: The widths of fully connected hidden layers of critics.

    Shape:
        - Input: `(N, C, H)` where C = in_channel and H = in_features.
        - Output: `(N, NC, 2)` where NC is the number of critics.
    """
    def __init__(
        self, in_channels: int, in_features: int, n_critics: int, 
        conv_channels: list, crt_layers: list
        ):
        super().__init__(in_channels, in_features, conv_channels=conv_channels)
        self.n_critics = n_critics
        self.critics = nn.Sequential(
            MLP(self.m_features, crt_layers[-1], crt_layers[:-1]),
            nn.Linear(crt_layers[-1], n_critics)
        )

    def forward(self, input):
        x = super().forward(input)
        critics = self.critics(x)
        return critics

class InfoDiscriminator1D(Critics1D):
    """Discriminator for GANs equiped with mutual information maximization.

    Args: 
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        n_critics: The number of critics.
        latent_dim: The number of latent variables
        conv_channels: The number of channels of each conv1d layer.
        crt_layers: The widths of fully connected hidden layers of critics.
        pred_layers: The widths of fully connected hidden layers of latent code predictor.

    Shape:
        - Input: `(N, C, H)` where C = in_channel and H = in_features.
        - Output: 
            - Critics: `(N, NC)` where NC = n_critics.
            - Latent Code: `(N, NL, 2)` where NL = latent_dim.
    """
    def __init__(
        self, in_channels: int, in_features: int, n_critics: int, latent_dim: int,
        conv_channels: list = [64, 64*2, 64*4, 64*8, 64*16, 64*32],
        crt_layers: list = [1024,],
        pred_layers: list = [512,]
        ):
        super().__init__(in_channels, in_features, n_critics, conv_channels, crt_layers)
        self.latent_dim = latent_dim
        self.latent_predictor = nn.Sequential(
            MLP(self.m_features, pred_layers[-1], pred_layers[:-1]),
            nn.Linear(pred_layers[-1], latent_dim * 2)
        )
    
    def forward(self, input):
        x = self.conv(input)
        critics = self.critics(x)
        latent_code = self.latent_predictor(x).reshape([-1, self.latent_dim, 2])
        return critics, latent_code


class AdaptiveCost(nn.Module):
    def __init__(self, feature_gen, p=2):
        super().__init__()
        self.feature_gen = feature_gen
        self.p = p
    
    def forward(self, x, y):
        ft_x = first_element(self.feature_gen(x))
        ft_y = first_element(self.feature_gen(y))
        return torch.cdist(ft_x, ft_y, p=self.p)