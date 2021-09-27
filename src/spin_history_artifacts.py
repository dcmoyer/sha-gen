

import numpy as np
from scipy import ndimage as ndi

import joblib

#
# This function assumes that the first block is the actual data, and the second is
# a brain mask. This function ignores other parts of data_block.
#
# This is naive and very slow...chose a better one later?
#
# SHA_func is a scalar function that takes the signed dist to a value in [0,1]
# ...or even [0,+), since the SHA can be brighter than normal
def SpinHistorySim( data_block, SHA_func, mask_dilation=5, cutoff=5, append_mask=False, append_scalar=False, multithreading=True, n_threads=16):

  img = data_block[0]
  mask = data_block[1]

  dilated_mask = ndi.morphology.binary_dilation(mask,iterations=mask_dilation)

  ix,iy,iz,_ = np.where(dilated_mask > 0)

  idx = np.random.choice(ix.shape[0])

  norm_vec = np.random.normal(size=(3,1))
  norm_vec = norm_vec / np.sqrt( np.sum( norm_vec * norm_vec )) 
  nzed_vec = np.array( [ix[idx],iy[idx],iz[idx]] )

  #D_coef = -np.dot(norm_vec, nzed_vec)

  mult_vol = np.ones(img.shape)

  if not multithreading:
    for i in range(img.shape[0]):
      dx = i - nzed_vec[0]

      for j in range(img.shape[1]):
        dy = j - nzed_vec[1]

        for k in range(img.shape[2]):
          dz = k - nzed_vec[2]

          w = np.array( [dx, dy, dz] )

          signed_dist = np.dot(w,norm_vec)
          if signed_dist > cutoff:
            continue
          mult_vol[i,j,k,0] = SHA_func(signed_dist)
  else:
    mult_vol = np.ctypeslib.as_ctypes(mult_vol)

    thread_func = lambda job_list : SHA_helper( job_list, nzed_vec, norm_vec, SHA_func )

    job_idx = [ (i,j,k) \
      for i in range(img.shape[0])
      for j in range(img.shape[1])
      for k in range(img.shape[2])
    ]

    batch_size = img.shape[0]*img.shape[1]
    chunks = [ job_idx[start_idx:(start_idx+batch_size)] for start_idx in range( 0, img.shape[2] ) ]

    #p = Pool()
    #out = p.map(thread_func, job_idx)
    output = joblib.Parallel(n_jobs=32,backend="threading")(
      joblib.delayed(thread_func)(chunk) for chunk in chunks
    )

    #mult_vol[ job_idx ] = output
    print( len(output))
    exit(0)

  if append_scalar:
    return [img*mult_vol] + [np.concatenate((mask,mult_vol),axis=-1)] + [d for d in data_block[2:]]
  elif append_mask:
    sha_mask = (mult_vol != 1.0).astype(np.float32)
    return [img*mult_vol] + [np.concatenate((mask,sha_mask),axis=-1)] + [d for d in data_block[2:]]
  else:
    return [img*mult_vol] + [d for d in data_block[1:]]


import torch
class SHA_Generator_Torch:
  def __init__(self, input_shape, sha_func, return_field=False, seed=1919):

    with torch.no_grad():

      self.input_shape = input_shape[1:]
      #maybe can set dynamically from torchio?
      self.n_chan = input_shape[0]
      self.coord_idx_list = []

      for idx,in_shape in enumerate(self.input_shape):
        coord_idx_tensor = torch.range(0,in_shape-1)
        coord_idx_tensor = torch.reshape(
          coord_idx_tensor,
          [in_shape] + [1]*(len(self.input_shape)-1)
        )

        #coord_idx_tensor = torch.repeat_interleave(
        #  coord_idx_tensor,
        #  torch.tensor([1] + self.input_shape_nb_nc[:idx] + self.input_shape_nb_nc[idx+1:])
        #)
        coord_idx_tensor = coord_idx_tensor.repeat(*([1] + self.input_shape[:idx] + self.input_shape[idx+1:]))

        coord_idx_tensor = coord_idx_tensor.permute(
          *(list(range(1,idx+1)) + [0] + list(range(idx+1,len(self.input_shape))))
        )

        self.coord_idx_list.append(coord_idx_tensor.detach())

        #self.coord_idx_list.append(
        #  torch.reshape(coord_idx_tensor,[-1])
        #)

      self.coord_idx_list = torch.stack(self.coord_idx_list)

      self.sha_func = sha_func
      self.rng = np.random.default_rng(seed)
      self.return_field = return_field

  def __call__(self, data_block):

    with torch.no_grad():
      if isinstance(data_block,tuple) or isinstance(data_block,list):
        vol, mask = data_block

        z = []
        for idx in range(len(self.input_shape)):
          m = torch.masked_select(self.coord_idx_list[idx,...], mask)
          index = torch.ones_like(m).multinomial(num_samples=1, replacement=True)
          z.append(m[index].item())

      else:
        vol = data_block

        z = []
        for idx in range(len(self.input_shape)):
          z.append(self.rng.choice( np.arange(self.input_shape[idx]) ))

        #define mask here
        mask = torch.ones_like(vol) > 0
      norm_vec = torch.normal(mean=torch.zeros((3,)))
      norm_vec = norm_vec / torch.sqrt((norm_vec * norm_vec).sum())

      #TODO: recreate here sha

      dp_vol = torch.zeros_like(vol)
      #for idx in range(len(self.coord_idx_list)):
      for idx in range(len(self.input_shape)):
        dp_vol = dp_vol + norm_vec[idx] * (self.coord_idx_list[idx,...] - z[idx])

      #dp_vol = torch.reshape( vol.size() )

      output = self.sha_func(dp_vol)

    if self.return_field:
      return output,dp_vol
    else:
      return output

if __name__ == "__main__":
  img_file = "/data/vision/polina/scratch/fetal-reorg/data/subj-data/MAP-C508-L/EPI-TE32/split_nifti/vol/MAP-C508-L_0024.nii.gz"
  mask_file = "/data/vision/polina/scratch/fetal-reorg/data/subj-data/MAP-C508-L/EPI-TE32/split_nifti/brain_seg/MAP-C508-L_0024_all_brains.nii.gz"

  import nibabel as nib
  img = nib.load(img_file).get_fdata()
  mask = nib.load(mask_file).get_fdata()

  img = img[np.newaxis,:,:,:]
  mask = mask[np.newaxis,:,:,:]

  data_block = [img,mask]
  sha_func = lambda x: np.max([1 - np.exp( -x*x ), 0.0])

  import time
  #start = time.process_time()
  #for i in range(10):
  #  SpinHistorySim( data_block, sha_func, cutoff=3, append_mask=True)
  #stop = time.process_time()

  #print("%0.2f sec per iteration" % ((stop-start)/10) )

  ##
  ##
  ##

  #output = SpinHistorySim( data_block, sha_func, cutoff=3, append_scalar=True)

  #nib.save( nib.Nifti1Image( output[0][:,:,:,0], np.eye(4) ), "sha_vol.nii.gz" )
  #nib.save( nib.Nifti1Image( output[1][:,:,:,0], np.eye(4) ), "sha_brain_mask.nii.gz" )
  #nib.save( nib.Nifti1Image( output[1][:,:,:,1], np.eye(4) ), "sha_plane_mask_scalar.nii.gz" )

  sha_func_torch = lambda x: torch.clip(1 - torch.exp( -x*x ), min=0.0)

  img_torch = torch.tensor(img)
  mask_torch = torch.tensor(mask) > 0

  print(img_torch.size())

  sha_gen_torch = SHA_Generator_Torch( list(img_torch.size()), sha_func_torch )

  start = time.time()
  for i in range(100):
    vol = sha_gen_torch( [img_torch, mask_torch] )
    print(vol.size())
  stop = time.time()
  print(start)
  print(stop)
  print("%0.2f sec per iteration" % ((stop-start)/100) )

  output = sha_gen_torch( [img_torch, mask_torch] )
  output = output.numpy()

  nib.save( nib.Nifti1Image( output[0,:,:,:], np.eye(4) ), "sha_vol.nii.gz" )
  #nib.save( nib.Nifti1Image( output[1][:,:,:,0], np.eye(4) ), "sha_brain_mask.nii.gz" )
  #nib.save( nib.Nifti1Image( output[1][:,:,:,1], np.eye(4) ), "sha_plane_mask_scalar.nii.gz" )










