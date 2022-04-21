
import matplotlib.pyplot as plt
import seaborn_image as isns
import nibabel as nib
import fetal_data_iterator as fdi
import numpy as np

# subj_obj = nib.load("/data/vision/polina/scratch/fetal-reorg/data/subj-data/MAP-C508-L/EPI-TE32/split_nifti/vol/MAP-C508-L_0000.nii.gz")
# scan_block = subj_obj.get_fdata()

#vol should be a 3-d tensor (so single channel image, with X,Y,Z axes)

def plotting_function(vol,x,y,z,ax,exp):
    # vol is a 3D array and the ones coming out of the nework is a 5D array with batch and channel dims [0][0][h][w] or [0,0,...] gives 3D vol
    isns.imgplot( np.transpose(vol[x,:,:]), cbar=False, gray=True, ax=ax[0])
    isns.imgplot( np.transpose(vol[:,y,:]), cbar=False, gray=True, ax=ax[1])
    isns.imgplot( vol[:,:,z], cbar=False, gray=True, ax=ax[2])
    plt.savefig(f"/data/vision/polina/scratch/haleysan/sha-gen/data_{exp}.png")


giant_list_of_epi_vols = []
N_train = 0
n_train_subj = 10
for subj_idx, subj in enumerate(fdi.FDSubjIterator()):
    n_frames_epi = subj.get_EPI_split_nifti_num_frames()

    for frame_idx in range(n_frames_epi):
        epi_path = subj.get_EPI_vol( frame_idx, just_path=True )
        # scan_block = nib.load(epi_path).get_fdata()
        # break
        giant_list_of_epi_vols.append(epi_path)

        if subj_idx < n_train_subj:
            N_train += 1

print(len(giant_list_of_epi_vols))

subject = giant_list_of_epi_vols[10]
print(subject)
subj_obj = nib.load(subject)
scan_block = subj_obj.get_fdata()
print(scan_block.shape[0])
fig1, ax1 = plt.subplots(1,3)
shape = scan_block.shape
scan_block  = np.log(scan_block+1)
# plotting_function(scan_block,(shape[0]-1)//2,(shape[1]-1)//2,(shape[2]-1)//2,ax1)

# fig1, ax = plt.subplots(1,3)
# isns.imgplot( np.transpose(scan_block[109,:,:]), cbar=False, gray=True, ax=ax[0])
# isns.imgplot( np.transpose(scan_block[:,109,:]), cbar=False, gray=True, ax=ax[1])
# isns.imgplot( scan_block[:,:,88], cbar=False, gray=True, ax=ax[2])
# plt.imshow(np.transpose(scan_block[109,:,:]))
# plt.show()
# plt.savefig("/data/vision/polina/scratch/haleysan/sha-gen/example_plot.png")


#example usage:
# fig1, ax1 = plt.subplots(1,3)
# plotting_function(scan_block,109,109,88,ax1)

# data shape (110, 110, 89)