
import torch
from torch.nn import functional as F
import nibabel as nib
import numpy as np

# NCWHD
def pad_to_size( vol, size ):

  pads = [
    0,size[-1] - vol.size()[-1],
    0,size[-2] - vol.size()[-2],
    0,size[-3] - vol.size()[-3],
  ]
  return F.pad( vol, pads, mode="constant", value=0 )

# NCWHD
def unpad( vol, original_size):
  return vol[:,:,0:original_size[0],0:original_size[1],0:original_size[2]]

def quantile_normalizer(vol, q1=0.90, q2=0.99):
  vol = vol.double()
  q = torch.quantile(vol,torch.tensor([q1,q2],dtype=torch.float64))
  vol = torch.clip(vol,min=0,max=q[1]) / q[0]
  return vol

def load_scale_and_pad( vol_path, size, vol_chan_last ):
  img = nib.load(vol_path)
  aff = img.affine
  vol = img.get_fdata()
  print(vol.shape)
  original_size = vol.shape

  if len(vol.shape) == 3:
    vol = vol[np.newaxis,np.newaxis,:,:,:]
  elif vol_chan_last:
    vol = vol[np.newaxis,:,:,:]
    vol = np.transpose(vol,axes=(0,4,1,2,3))
  elif len(vol.shape) == 4:
    vol = vol[np.newaxis,:,:,:]
  else:
    print("BAD VOL INFO?")

  vol = torch.tensor(vol)
  vol = quantile_normalizer(vol)

  vol = pad_to_size(vol, size)
  return vol, original_size, aff

def extract_and_save( out_path, outvol, original_size, affine=np.eye(4) ):
  outvol = unpad( outvol, original_size ).numpy()[0,0,:,:,:]
  nib.save( nib.Nifti1Image( outvol, affine ), out_path )




