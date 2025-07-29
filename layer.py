import torch
import torch.nn as nn
import torch.nn.functional as F




class Convolution3DCH(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.kwargs = kwargs
        # self.name = kwargs['name']
        self.filters = kwargs['out_channels']
        ks = (kwargs['kernel_size'],) * 3
        self.activation = kwargs['activation']
        self.in_channels=kwargs['in_channels']


        # if isinstance(kwargs['padding'], str):
        pad1=kwargs['padding']
        pad2=kwargs['padding']
        pad3=kwargs['padding']
        # else:
        #     pad1=(0, kwargs['padding'], kwargs['padding'])
        #     pad2=(kwargs['padding'], 0, kwargs['padding'])
        #     pad3=(kwargs['padding'], kwargs['padding'], 0)

        self.convx = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.filters,
            kernel_size=(1, ks[1], ks[2]),
            padding=pad1,
            stride=kwargs['stride']
        )

        self.convy = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.filters,
            kernel_size=(ks[0], 1, ks[2]),
            padding=pad2,
            stride=kwargs['stride']
        )

        self.convz = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.filters,
            kernel_size=(ks[0], ks[1], 1),
            padding=pad3,
            stride=kwargs['stride']
        )

    def forward(self, x):
        x_axis = self.convx(x)
        y_axis = self.convy(x)
        z_axis = self.convz(x)
        
        # print("x:", x_axis.shape)
        # print("y:", y_axis.shape)
        # print("z:", z_axis.shape)
        out = x_axis + y_axis + z_axis
        if self.activation != 'linear':
            out = getattr(F, self.activation)(out)
        return out

    def get_weights(self):
        return {
            'x_weights': self.convx.state_dict(),
            'y_weights': self.convy.state_dict(),
            'z_weights': self.convz.state_dict()
        }

    def set_weights(self, weights):
        if 'x_weights' in weights:
            self.convx.load_state_dict(weights['x_weights'])
        if 'y_weights' in weights:
            self.convy.load_state_dict(weights['y_weights'])
        if 'z_weights' in weights:
            self.convz.load_state_dict(weights['z_weights'])


class Conv3DFunctional(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation='linear', padding='same', stride=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride = stride)
        self.activation=activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation != 'linear':
            x = getattr(F, self.activation)(x)
        return x