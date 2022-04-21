import os

import torch

import fetal_loader
import unet_arch
import ae_arch
import re

import example_plotting

#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

import toml_utils
from torch.profiler import profile, record_function, ProfilerActivity

torch.backends.cudnn.benchmark= True

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--config', help='toml experiment config',default=None)
args = parser.parse_args()

config = args.config

if config is not None:
  print(config)
  config = toml_utils.recursive_load(config)
  print(config)
  out_path = config["out_path"] if "out_path" in config.keys() else "params"
  exp_name = config["exp_name"] if "exp_name" in config.keys() else "test-v2"
  down_layer_sizes = config["down_layer_sizes"] if "down_layer_sizes" in config.keys() else [1,32,64,128,256]
  up_layer_sizes = config["up_layer_sizes"] if "up_layer_sizes" in config.keys() else [32,64,128,256,512]
  skip_layer_sizes = config["skip_layer_sizes"] if "skip_layer_sizes" in config.keys() else [32,32,64,128,256]
  LR = float(config["LR"]) if config is not None and "LR" in config.keys() else 1e-5

  arch_type = config["arch_type"] if "arch_type" in config.keys() else "unet"

# else:
  # out_path = "params"
  # exp_name = "test-v2"
  # down_layer_sizes = [1,32,64,128,256]
  # up_layer_sizes = [32,64,128,256,512]
  # skip_layer_sizes = [32,32,64,128,256]
  # LR=1e-5
  # arch_type = "unet"

batch_size = 2
n_epochs = 10000

train_iterator, val_iterator = fetal_loader.get_fetal_torch_iterator(80)
train_loader = torch.utils.data.DataLoader(
  train_iterator,
  batch_size=batch_size,
  shuffle=True,
  pin_memory=True,
  num_workers=16
  #prefetch_factor=10
)
val_loader = torch.utils.data.DataLoader(
  val_iterator,
  batch_size=batch_size,
  shuffle=True,
  pin_memory=True,
  num_workers=32
)



# sets the device for torch, we will use device object later to send both
# model and data to device
if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'


##
## TODO: select arch
##
#net_obj = unet_arch.UNetArch(
#  input_n_chan = 1,
#  down_layer_sizes = [1,32,64,64,64],
#  up_layer_sizes = [32,64,64,64,64],
#  skip_layer_sizes = [1,32,64,64,64]
#)

if arch_type == "unet":
  net_obj = unet_arch.UNetArch(
    input_n_chan = 1,
    output_n_chan = 1,
    down_layer_sizes = down_layer_sizes,
    up_layer_sizes = up_layer_sizes,
    skip_layer_sizes = skip_layer_sizes
  )

elif arch_type == "autoencoder":
  net_obj = ae_arch.AEArch(
    input_n_chan = 1,
    output_n_chan = 1,
    down_layer_sizes = down_layer_sizes,
    up_layer_sizes = up_layer_sizes,
    first_layer_size = 64 #TODO move to config
  )

else:
  print(f"arch type {arch_type} not recognized")
  exit(1)

# print(net_obj)
net_obj = net_obj.to(device)

optimizer = torch.optim.Adam(net_obj.parameters(), lr=LR)

loss_func = torch.nn.MSELoss()

##
## where to save params
##
# os.makedirs( out_path, exist_ok=True )
# os.makedirs( f"{out_path}/{exp_name}", exist_ok=True )

# previous_files = os.listdir(f"{out_path}/{exp_name}")
# previous_files.sort(key=lambda f: int(re.sub('\D', '', f)))
# if len(previous_files) > 0:
#   last_file = previous_files[-1]
#   print(f"loading from {last_file}")
#   net_obj.load_state_dict(torch.load(f"{out_path}/{exp_name}/{last_file}"))
#   last_epoch = int(last_file[:-4]) + 1
# else:
#   last_epoch = 0
# val_loss = 0
val_loss_list = []
for epoch in range(0, 105, 5):

  print(f"epoch {epoch}", flush=True)

  val_loss = 0
  N = 0

  # net_obj.load_state_dict(torch.load(f"{out_path}/{exp_name}/{epoch}.pth"))

  for d_idx,batch in enumerate(val_loader):

      #inputs, outputs = batch
      inputs = batch[1]
      outputs = batch[0]

      # if epoch == 10:
        # plotting_function(inputs,(shape[0]-1)//2,(shape[1]-1)//2,(shape[2]-1)//2,ax1, exp_name)

      inputs = inputs.to(device,non_blocking=True)
      outputs = outputs.to(device) #this is blocking
      # if epoch == 10:
      #   plotting_function(scan_block,(shape[0]-1)//2,(shape[1]-1)//2,(shape[2]-1)//2,ax1)
      #   isns.imgplot( np.transpose(vol[x,:,:]), cbar=False, gray=True, ax=ax[0])
      #   isns.imgplot( np.transpose(vol[:,y,:]), cbar=False, gray=True, ax=ax[1])
      #   isns.imgplot( vol[:,:,z], cbar=False, gray=True, ax=ax[2])
      #   plt.savefig(f"/data/vision/polina/scratch/haleysan/sha-gen/data_{exp_name}.png")

      optimizer.zero_grad()
      net_obj.zero_grad(set_to_none=True)
      outputs_approx = net_obj.forward(inputs)

      loss_value = loss_func( outputs, outputs_approx ).mean()
      # loss_value.backward(retain_graph=True)
      # optimizer.step()

      val_loss += loss_value.item()*batch[0].size()[0]
      N += batch[0].size()[0]

      val_loss_list.append(val_loss)

      del inputs
      del outputs
      del loss_value

      if d_idx > 1000:
        break
      # print(val_loss/N)



import matplotlib.pyplot as plt
import seaborn_image as isns
import nibabel as nib
import fetal_data_iterator as fdi
import numpy as np

import example_plotting

xpoints = np.array([x for x in range(0,105,5)])
ypoints = np.array(val_loss_list)

plt.plot(xpoints, ypoints)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")

# plt.show()
plt.savefig(f'"/data/vision/polina/scratch/haleysan/sha-gen/MSE_loss_{exp_name}.png"')
