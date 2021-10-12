
import matplotlib.pyplot as plt

from nilearn import plotting
import seaborn_image as isns

#vol should be a 3-d tensor (so single channel image, with X,Y,Z axes)
def plotting_function(vol,x,y,z,ax):
    isns.imgplot( np.transpose(vol[x,:,:]), cbar=False, gray=True, ax=ax[0])
    isns.imgplot( np.transpose(vol[:,y,:]), cbar=False, gray=True, ax=ax[1])
    isns.imgplot( vol[:,:,z], cbar=False, gray=True, ax=ax[2])
    plt.show()

#example usage:
# fig1, ax1 = plt.subplots(1,3)
# plotting_function(vol,48,48,48,ax1)

