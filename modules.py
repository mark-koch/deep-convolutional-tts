import torch.nn as nn
import torch.nn.functional as F
from config import Config


# TODO: Normalize for Conv1d should default to True, avoid double dropout for HighwayConv!

class Conv1d(nn.Conv1d):
    """ 1d convolution that supports causal padding and layer normalization """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding="same", normalize=False):
        self.pad = padding.lower()
        self.normalize = normalize
        if padding.lower() == "same":
            pad = ((kernel_size - 1) * dilation + 1) // 2
        elif padding.lower() == "causal":
            pad = (kernel_size - 1) * dilation  # Use build in padding and remove right padding later
        else:
            pad = 0
        super(Conv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     dilation=dilation, padding=pad)
        if normalize:
            self.layerNorm = nn.LayerNorm([out_channels], )
        self.drop_out = nn.Dropout(Config.dropout_rate) if Config.dropout_rate > 0 else None

    def forward(self, x):
        y = super(Conv1d, self).forward(x)
        if self.pad == "causal":
            y = y[:, :, :x.size(2)]  # Remove padding on the right
        if self.normalize:
            y = y.transpose(1, 2)
            y = self.layerNorm(y)
            y = y.transpose(1, 2)
        if self.drop_out is not None:
            y = self.drop_out(y)
        return y


class HighwayConv(Conv1d):
    """ 1d convolution followed by highway net like gated activation. """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding="same", normalize=True):
        self._normalize = normalize
        super(HighwayConv, self).__init__(in_channels=in_channels, out_channels=2*out_channels,
                                          kernel_size=kernel_size, dilation=dilation, padding=padding, normalize=False)
        if normalize:
            self.layerNorm1 = nn.LayerNorm([out_channels])
            self.layerNorm2 = nn.LayerNorm([out_channels])
        self.drop_out = nn.Dropout(Config.dropout_rate) if Config.dropout_rate > 0 else None

    def forward(self, x):
        y = super(HighwayConv, self).forward(x)  # [N, 2*out_channels, T]
        H1, H2 = y.chunk(2, dim=1)  # Split along channels, we get [batch, out_channels, time]

        if self._normalize:
            H1 = self.layerNorm1(H1.transpose(1, 2)).transpose(1, 2)
            H2 = self.layerNorm2(H2.transpose(1, 2)).transpose(1, 2)
        H1 = F.sigmoid(H1)
        y = H1*H2 + (1-H1) * x
        if self.drop_out is not None:
            y = self.drop_out(y)
        return y


class ConvTranspose1d(nn.ConvTranspose1d):
    """ 1d transposed convolution that supports layer normalization """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, padding="same", normalize=True):
        self.normalize = normalize
        if padding.lower() == "same":
            pad = max(0, (kernel_size - 2) // 2)
        else:
            pad = 0
        super(ConvTranspose1d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size, dilation=dilation, stride=stride, padding=pad)
        self.layer_norm = nn.LayerNorm([out_channels]) if normalize else None
        self.drop_out = nn.Dropout(Config.dropout_rate) if Config.dropout_rate > 0 else None

    def forward(self, x, output_size=None):
        y = super(ConvTranspose1d, self).forward(x, output_size)
        if self.normalize:
            y = self.layer_norm(y.transpose(1, 2)).transpose(1, 2)
        if self.drop_out is not None:
            y = self.drop_out(y)
        return y


