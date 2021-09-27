
import torch
from torch.nn import functional as F
import torchio as tio
import fetal_data_iterator as fdi

# local
import spin_history_artifacts as sha

import utils
from torch.utils.data import Dataset


class fetal_torch_iter(Dataset):
  def __init__(self, vol_size, list_of_vol_paths, noise_transformer):

    self.subjects_list = [
      tio.Subject(vol=tio.ScalarImage(vol_path)) \
        for vol_path in list_of_vol_paths
    ]
    self.croporpad = tio.CropOrPad( vol_size )
    self.normalizer = tio.Lambda(utils.quantile_normalizer)
    self.preproc = tio.Compose((self.croporpad, self.normalizer))
    self.subj_dataset = tio.SubjectsDataset(self.subjects_list, transform=self.preproc)

    self.noise_transformer = noise_transformer

  def __len__(self):
    return len(self.subj_dataset)

  def __getitem__(self, index : int):
    subj = self.subj_dataset[index]
    subj.load()
    vol = subj["vol"]
    noised_vol = self.noise_transformer(vol)
    return vol.data.float().detach(), noised_vol.data.float().detach()

def create_L_mask( vol_size, L ):
  mask = torch.zeros(vol_size)
  mask[0:L,:,:] = 0
  mask[-L:,:,:] = 0
  mask[:,0:L,:] = 0
  mask[:,-L:,:] = 0
  mask[:,:,0:L] = 0
  mask[:,:,-L:] = 0
  return mask

def isotropic_resample(vol, scale):
  vol = torch.unsqueeze(vol, 0)
  down_vol = F.interpolate( vol, scale_factor=4.0/3.0, mode="nearest", recompute_scale_factor=False)
  out_vol = F.interpolate( down_vol, size=list(vol.size())[2:], mode="trilinear", align_corners=False)
  return out_vol[0,...]
  #axes=(0,1,2), downsampling=(1,4.0/3.0)

#    sha_tio = tio.Lambda( lambda vol : sha_gen(vol) * vol )
def sha_tio_multivol( vol, sha_gen ):
  sha_output,dp_vol = sha_gen(vol)
  output = torch.cat((sha_output*vol,torch.sigmoid(dp_vol)),axis=0) 
  return output

def get_fetal_torch_iterator(n_train_subj, return_field=False):

  ##PARAMS
  vol_size = [96,96,96]
  mask = create_L_mask(vol_size, 20)

  min_sha = 0.75
  max_sha = 0.85
  sha_width = 1.5
  make_rand_sha = lambda : min_sha + torch.rand(1)*(max_sha - min_sha)
  sha_func = lambda x: torch.clip(1 - make_rand_sha()*torch.exp( - (x*x)/(sha_width) ), min=0.0)
  sigma = 0.05
  #sha_func = lambda x: torch.clip(1 - 0.95*torch.exp( -x*x ), min=0.0)

  ##END PARAMS

  giant_list_of_epi_vols = []
  N_train = 0
  for subj_idx, subj in enumerate(fdi.FDSubjIterator()):
    n_frames_epi = subj.get_EPI_split_nifti_num_frames()

    for frame_idx in range(n_frames_epi):
      epi_path = subj.get_EPI_vol( frame_idx, just_path=True )
      giant_list_of_epi_vols.append(epi_path)

      if subj_idx < n_train_subj:
        N_train += 1

      #print(epi_path)

  #create transformer
  downsample = tio.Lambda(
    lambda vol : isotropic_resample( vol, 4.0/3.0 )
  )
  #downsample = tio.RandomAnisotropy(axes=(0,1,2), downsampling=(1,4.0/3.0))
  #TODO: move affine to preproc?
  #affine = tio.RandomAffine(degrees=90,translation=10)
  rnorm_tio = tio.Lambda(
    lambda vol : torch.clip(vol + sigma * torch.normal(torch.zeros_like(vol)), min=0)
  )

  sha_gen = sha.SHA_Generator_Torch( [1] + vol_size, sha_func, return_field=return_field )

  if return_field:
    sha_tio = tio.Lambda( lambda vol : sha_tio_multivol( vol, sha_gen ) )
  else:
    sha_tio = tio.Lambda( lambda vol : sha_gen(vol) * vol )

  noise_model = tio.Compose((downsample, rnorm_tio, sha_tio))

  return (fetal_torch_iter(vol_size, giant_list_of_epi_vols[:N_train], noise_model),
    fetal_torch_iter(vol_size, giant_list_of_epi_vols[N_train:], noise_model))


if __name__ == "__main__":
  import nibabel as nib
  import numpy as np

  fiter,fiter_val = get_fetal_torch_iterator(10)

  #print(fiter[0][0].size())

  nib.save( nib.Nifti1Image(fiter[0][1][0,...].numpy(), np.eye(4)), "fetal_loader_check.nii.gz" )
  nib.save( nib.Nifti1Image(fiter[1000][1][0,...].numpy(), np.eye(4)), "fetal_loader_check2.nii.gz" )
  nib.save( nib.Nifti1Image(fiter[2000][1][0,...].numpy(), np.eye(4)), "fetal_loader_check3.nii.gz" )



