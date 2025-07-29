import numpy as np
from layer import Convolution3DCH
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import Conv3DFunctional





class ConvBlock3D(nn.Module):
  def __init__(self, in_channels, out_channels, cross_hair, bottleneck = False, activation='tanh',  kernel=5) :
    super(ConvBlock3D, self).__init__()
    Conv = Convolution3DCH if cross_hair else Conv3DFunctional

    self.conv1 = Conv(in_channels= in_channels, out_channels=out_channels//2, kernel_size=kernel, activation='linear', padding='same', stride=1)
    self.bn1 = nn.BatchNorm3d(num_features=out_channels//2)
    self.conv2 = Conv(in_channels= out_channels//2, out_channels=out_channels, kernel_size=kernel, activation='linear', padding='same', stride=1)
    self.bn2 = nn.BatchNorm3d(num_features=out_channels)

    if activation != 'linear':
       self.act = getattr(F, activation)

    self.bottleneck = bottleneck
    if not bottleneck:
        #self.subsample = Conv(in_channels= out_channels, out_channels=out_channels, kernel_size=kernel, activation='linear', padding=(kernel - 1) // 2, stride=2)
        self.subsample = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)


  def forward(self, input):
    res = self.act(self.bn1(self.conv1(input)))
    res = self.act(self.bn2(self.conv2(res)))
    out = None
    # print(f'res shape:{res.shape}')
    if not self.bottleneck:
        out = self.subsample(res)
        
    else:
        out = res
    # print(f'out shape:{res.shape}')
    return out, res


class Decoderblock(nn.Module):
  def __init__(self, in_channels, cross_hair, res_channels=0, last_layer=False, num_classes=None, bottleneck = False,  activation='tanh', kernel=5):
    super(Decoderblock, self).__init__()
    assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None)
    Conv = Convolution3DCH if cross_hair else Conv3DFunctional
    self.last_layer=last_layer

    if bottleneck:
      self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=(2, 2, 2), stride=2)
      #self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=2*in_channels, kernel_size=kernel, stride=2, padding=(kernel - 1) // 2, output_padding=1)
      in_channels = in_channels*2
      out_channels = in_channels // 4
    else:
      self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
      #self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel, stride=2, padding=(kernel - 1) // 2, output_padding=1)
      out_channels = in_channels // 2
    
    if activation != 'linear':
      self.act = getattr(F, activation)

    self.conv1 = Conv(in_channels=in_channels+res_channels, out_channels=out_channels, kernel_size=kernel, stride=1, padding='same', activation='linear')
    self.bn = nn.BatchNorm3d(out_channels)
    self.conv2 = Conv(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel, stride=1, padding='same', activation='linear')

    if last_layer:
      self.conv3 = Conv3DFunctional(in_channels=out_channels, out_channels=num_classes, kernel_size=(1,1,1), stride=1, padding='same', activation='linear')

  def forward(self, input, residual=None):

    out = self.upconv1(input)
    out = self.act(out)

    if residual!=None:
      out = torch.cat((out, residual), 1)
    out = self.act(self.bn(self.conv1(out)))
    out = self.act(self.bn(self.conv2(out)))
    if self.last_layer:
      out = self.conv3(out)
      # out = F.softmax(out, dim=1)
    #print(out.shape)
    return out


    

class UNet3D(nn.Module):
  def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512, cross_hair=False) -> None:
    super(UNet3D, self).__init__()
    level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]

    self.a_block1 = ConvBlock3D(in_channels=in_channels, out_channels=level_1_chnls, cross_hair=cross_hair)
    self.a_block2 = ConvBlock3D(in_channels=level_1_chnls, out_channels=level_2_chnls, cross_hair=cross_hair)
    self.a_block3 = ConvBlock3D(in_channels=level_2_chnls, out_channels=level_3_chnls,cross_hair=cross_hair)
    
    self.bottleNeck = ConvBlock3D(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True, cross_hair=cross_hair)

    self.s_block3 = Decoderblock(in_channels=bottleneck_channel, res_channels=level_3_chnls, bottleneck= True, cross_hair=cross_hair)
    self.s_block2 = Decoderblock(in_channels=level_3_chnls, res_channels=level_2_chnls, cross_hair=cross_hair)
    self.s_block1 = Decoderblock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True, cross_hair=cross_hair)


  def forward(self, input):
    #Analysis path forward feed
    out, residual_level1 = self.a_block1(input)
    
    out, residual_level2 = self.a_block2(out)
    out, residual_level3 = self.a_block3(out)
    out, _ = self.bottleNeck(out)

    

    #Synthesis path forward feed
    out = self.s_block3(out, residual_level3)
    out = self.s_block2(out, residual_level2)
    out = self.s_block1(out, residual_level1)
    return out