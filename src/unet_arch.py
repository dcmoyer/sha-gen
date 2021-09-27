

import torch
import torch.nn as nn
import torch.nn.functional as F


class down_block( nn.Module ):
  def __init__(self, n_in, n_out, activation=F.relu):
    super(down_block, self).__init__()
    self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    self.activation = activation

    self.pool = nn.AvgPool3d(2, stride=2, padding=0)

  def forward(self, x):
    return self.pool( self.activation( self.conv2( self.activation( self.conv1(x) ) ) ) )

class up_block( nn.Module ):
  def __init__(self, n_in, n_out, activation=F.relu):
    super(up_block, self).__init__()
    self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    self.activation = activation

  def forward(self, x):
    out = self.activation( self.conv2( self.activation( self.conv1(x) ) ) )

    #TODO:switch to ONNX compliant upsample?
    # see https://github.com/pytorch/pytorch/pull/52855
    #out = torch.repeat_interleave(out, repeats=2, dim=2)
    #out = torch.repeat_interleave(out, repeats=2, dim=3)
    #out = torch.repeat_interleave(out, repeats=2, dim=4)
    #s = out.size()
    #out = out.view(s[0],s[1],-1,1,1,1).repeat(1,1,1,2,2,2).view(s[0],s[1],2*s[2],2*s[3],2*s[4])
    #print(self.conv2.__dict__)
    s = self.conv2.__dict__["out_channels"]
    weights = torch.eye(s).view(s,s, 1, 1, 1).repeat(1,1,2,2,2)
    out = F.conv_transpose3d( out, weights.to(out.device), stride=2, padding=0)
    return out

#TODO: set last layer to have no skip
#full pass through
class skip_block( nn.Module ):
  def __init__(self, n_in, n_out, activation=F.relu, pass_thru=False):
    super(skip_block, self).__init__()

    self.pass_thru = pass_thru
    if not self.pass_thru:
      self.conv = nn.Conv3d(n_in, n_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
      self.activation = activation

  def forward(self, x):
    if not self.pass_thru:
      return self.activation( self.conv(x) )
    return x

##
##
##

##
##len(layer_sizes) should be n_down_blocks + 1
class UNetArch( nn.Module ):
  def __init__(self, input_n_chan, output_n_chan, down_layer_sizes, up_layer_sizes, skip_layer_sizes):
    super(UNetArch, self).__init__()

    self.n_down = len(down_layer_sizes) - 1 

    self.down_layers = torch.nn.ModuleList()
    self.up_layers = torch.nn.ModuleList()
    self.skip_layers = torch.nn.ModuleList()

    for i in range(self.n_down):
      #n_plus_one_up_layer = if i+1 == n_down 0 else up_layer_sizes[i]

      ### FIRST LAYER 
      if i == 0 :
        down_size = input_n_chan 
      else:
        down_size = down_layer_sizes[i]

      ### LAST LAYER
      if i+1 == self.n_down :
        up_size = skip_layer_sizes[i+1]
        self.skip_layers.append( skip_block(skip_layer_sizes[i+1], skip_layer_sizes[i+1], pass_thru=True))
      else:
        up_size = skip_layer_sizes[i+1] + up_layer_sizes[i+1]
        self.skip_layers.append( skip_block(skip_layer_sizes[i+1], skip_layer_sizes[i+1]) )

      self.down_layers.append( down_block(down_size, down_layer_sizes[i+1]) )
      self.up_layers.append( up_block( up_size , up_layer_sizes[i] ) )

    self.zeroth_skip = skip_block(input_n_chan, skip_layer_sizes[0])
    self.output_layers = nn.Conv3d( skip_layer_sizes[0] + up_layer_sizes[0], output_n_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

  def forward(self, x):

    s0 = self.zeroth_skip(x)
    u = x
    list_of_skips = []
    for i in range(self.n_down):
      u = self.down_layers[i](u)
      list_of_skips.append( self.skip_layers[i](u) )

    #d = self.up_layers[-1](list_of_skips[-1])
    d = None
    for i in reversed(range(self.n_down)):
      if d is not None:
        d = torch.cat( (list_of_skips[i],d), axis=1)
      else:
        d = list_of_skips[i]
      d = self.up_layers[i](d)

    d = torch.cat((s0,d), axis=1)
    output = self.output_layers(d)

    return output


if __name__ == "__main__":
  arch = UNetArch( 1, [4, 8, 16, 32, 64], [4, 8, 16, 32, 64], [4, 8, 16, 32, 64] )



