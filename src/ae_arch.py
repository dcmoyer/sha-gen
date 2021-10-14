
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
    s = self.conv2.__dict__["out_channels"]
    up_weights = torch.eye(s).view(s,s, 1, 1, 1).repeat(1,1,2,2,2)
    self.register_buffer(
        "up_weights", up_weights, persistent=False
    )

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
    #out = F.conv_transpose3d( out, weights.to(out.device), stride=2, padding=0)
    out = F.conv_transpose3d( out, self.up_weights, stride=2, padding=0)
    return out

#  def to(self,**kwargs):
#    self.to(**kwargs)
#    self.up_weights = self.up_weights.to(**kwargs)

##
##len(layer_sizes) should be n_down_blocks + 1
class AEArch( nn.Module ):
  def __init__(self, input_n_chan, output_n_chan, down_layer_sizes, up_layer_sizes, first_layer_size):
    super(AEArch, self).__init__()

    self.n_down = len(down_layer_sizes) - 1 

    self.down_layers = torch.nn.ModuleList()
    self.up_layers = torch.nn.ModuleList()

    for i in range(self.n_down):
      #n_plus_one_up_layer = if i+1 == n_down 0 else up_layer_sizes[i]

      ### FIRST LAYER 
      if i == 0 :
        down_size = first_layer_size
      else:
        down_size = down_layer_sizes[i]

      ### LAST LAYER
      if i+1 == self.n_down :
        up_size = down_layer_sizes[i+1]
      else:
        up_size = up_layer_sizes[i+1]

      self.down_layers.append( down_block(down_size, down_layer_sizes[i+1]) )
      self.up_layers.append( up_block( up_size , up_layer_sizes[i] ) )

    self.zeroth_layer = nn.Conv3d( input_n_chan, first_layer_size, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
    self.output_layers = nn.Conv3d( up_layer_sizes[0], output_n_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)

  def forward(self, x):

    x = self.zeroth_layer(x)
    for i in range(self.n_down):
      x = self.down_layers[i](x)

    #d = self.up_layers[-1](list_of_skips[-1])
    for i in reversed(range(self.n_down)):
      x = self.up_layers[i](x)

    output = self.output_layers(x)

    return output


if __name__ == "__main__":
  arch = AEArch( 1, 1, [32, 32, 32, 32], [32, 32, 32, 32], 16 )




