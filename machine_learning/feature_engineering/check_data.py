import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
"""
This script is used to check the reconstruction quality of the signal representation methodology
"""
#%%
fit_dim=3
window_size=20
data_folder = 'fitting_data'
data_dir="power_20.csv"
#%% Read the data
coef = pd.read_csv(os.path.join(data_folder,data_dir), header=None)
#%% transform the data into numpy array and change to float 128
# coef_np = np.array(coef, dtype=np.float128)
coef_np = coef.values
#%% reshape
orig_dim=coef_np.shape
data_reshape=coef_np.reshape(orig_dim[0],orig_dim[1]//fit_dim,fit_dim)
#%% spec_recover
index_range=np.arange(1,window_size+1,1)
spec_recover=[]
for i in index_range:
    # spec_recover.append(data_reshape[:,:,0]*i**2+data_reshape[:,:,1]*i+data_reshape[:,:,2]) # second order polynominal
    # spec_recover.append(data_reshape[:,:,0]*i**3+data_reshape[:,:,1]*i**2+data_reshape[:,:,2]*i+data_reshape[:,:,3]) # third order polynomial
    # spec_recover.append(data_reshape[:,:,0]*i+data_reshape[:,:,1]) # first order polynominal

    # spec_recover.append(data_reshape[:,:,0] + data_reshape[:,:,1] * np.cos(i * data_reshape[:,:,3])
    #                     + data_reshape[:,:,2] * np.sin(i * data_reshape[:,:,3]))  # first Fourier fitting

    # spec_recover.append(data_reshape[:,:,0] * np.exp(data_reshape[:,:,1] * i) + data_reshape[:,:,2] * np.exp(data_reshape[:,:,3] * i))  # second order exponential
    spec_recover.append(data_reshape[:,:,0] * i** data_reshape[:,:,1] + data_reshape[:,:,2]) # second order power
spec_recover_np=np.array(spec_recover)
# change the dimension
spec_recover_np_shift=spec_recover_np.transpose(1,2,0)
# set nan and inf to 0
spec_recover_np_shift[np.isnan(spec_recover_np_shift)]=0
spec_recover_np_shift[np.isinf(spec_recover_np_shift)]=0
#%% change the dimension
spec_recover_np_orig=spec_recover_np_shift.reshape(orig_dim[0],-1)
data_save_dir=os.path.join("fitting_data",data_dir[:-4]+'_fit.csv')
np.savetxt(data_save_dir, spec_recover_np_orig[0], delimiter=",")
#%%
# spec_dir="spectrum_201010_low.csv"
# spec = pd.read_csv(spec_dir, header=None)
# spec_np = spec.values
# spec_np_log=np.log(spec_np)
# #%% plot the data
# plt.figure(figsize=[5,3])
# plt.plot(spec_recover_np_orig[0,:])
# plt.plot(spec_np_log[0,spec_recover_np_orig.shape[1]:])
# plt.xlabel('Wavenumber (cm$^{-1}$)')
# plt.ylabel('Intensity')
# plt.xlim(0,7000)
# plt.xticks(np.arange(0,7001,1000))
# plt.ylim(-10,10)
# plt.legend(['Fitted', 'Original'])
# plt.tight_layout()
# image_name=os.path.join("fitting_image",data_dir[:-4]+'.svg')
# plt.savefig(image_name)
# plt.show()
#%%

#%% read the original data
# spec_dir="spectrum_201010_low.csv"
# spec = pd.read_csv(spec_dir, header=None)
# spec_np = spec.values
# spec_np_log=np.log(spec_np)
# #%%
# plt.plot(spec_np_log[0,:])
# plt.show()
# #%% calculate the mse and pearson correlation
# dim_recover=spec_recover_np_orig.shape
# mse=np.sum((spec_np_log[:2000,:dim_recover[1]]-spec_recover_np_orig[:2000,:dim_recover[1]])**2)/dim_recover[0]/dim_recover[1]
# # calculate the pearson correlation of two matrix
# pearson_corr=stats.pearsonr(spec_np_log[:2000,:dim_recover[1]].flatten(),spec_recover_np_orig[:2000,:dim_recover[1]].flatten())
# # keep four decimal places
# if not os.path.exists('metric'):
#     os.makedirs('metric')
# # create a data frame to save mse and pearson correlation
# metric = {'mse': [mse], 'pearson_corr': [pearson_corr[0]]}
# metric_df = pd.DataFrame(data=metric)
# save_dir=os.path.join('metric',data_dir[:-4]+'_metric.csv')
# metric_df.to_csv(save_dir, index=False)