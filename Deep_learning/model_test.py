import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.metrics import mean_squared_error
from algorithm.resnet import *
# from algorithm.resnet import *
# set seed for numpy and pytorch
torch.manual_seed(1)
np.random.seed(1)

#%% data_reading
smp_dir = "./test_dataset/test_data.txt"
tgt_dir = "./test_dataset/test_temp_normed.txt"
data = pd.read_csv(smp_dir, header=None, dtype=np.float32)
target = pd.read_csv(tgt_dir, header=None, dtype=np.float32)
#%%load normalization
norm_spec_dir="test_dataset/spec_boundary.csv"  # normalization boundary for spectrum
max_norm_temp_dir="test_dataset/temp_max_boundry_standard.txt"  # normalization boundary for temperature
min_norm_temp_dir="test_dataset/temp_min_boundry_standard.txt"  # normalization boundary for temperature
spec_norm=np.loadtxt(norm_spec_dir)
max_temp=np.array(pd.read_csv(max_norm_temp_dir, header=None, dtype=np.float32))
min_temp=np.array(pd.read_csv(min_norm_temp_dir, header=None, dtype=np.float32))
# %% dimension augmentation
data_np = np.array(data)
max_data = np.max(data_np, axis=0)
min_data = np.min(data_np, axis=0)
data_scaled = (data_np - spec_norm[1]) / (spec_norm[0] - spec_norm[1])
data_new = data_scaled[:, ::6]
data_need = data_new[:, 0:1024]
# %%
target_np = np.array(target)
data_np_ext2 = data_need.reshape(-1, 1, 32, 32)

# %% change to dataset
sample_tc = torch.from_numpy(data_np_ext2)
sample_tcc = sample_tc.float()
label_tc = torch.from_numpy(target_np)
label_tcc = label_tc.float()
data_length = len(sample_tcc)
test_start_Index = 0
test_set = sample_tcc[test_start_Index:]
test_label_set = label_tcc[test_start_Index:]
# %% **********************compile model**********************************************
# choose model------------------

# model = VGG(make_layers(cfg['E'], batch_norm=False))
model=ResNet(BasicBlock, [2, 2, 2, 2])
# model=ShuffleNet([4, 8, 4])
# model=InceptionV3()
# model=SqueezeNet(class_num=11)
# model=Xception(MiddleFLowBlock)
# model=InceptionV4(4, 7, 3)

#
model.to("cuda")
folder="./model/"
model_dir="resnet18.pt"
model_perf_dir="perf_resnet18.txt"
model_save_dir = folder+model_dir
model.load_state_dict(torch.load(model_save_dir))

pred_test = np.zeros_like(test_label_set)
for index, test_sample in enumerate(test_set):
    test_sample = test_sample.unsqueeze(0).to("cuda")
    temp_test_pred = model(test_sample)
    pred_test[index] = temp_test_pred.detach().cpu().numpy()

#%% calculate matrix
ori_target=np.array(test_label_set)*(max_temp-min_temp)+min_temp
ori_pred=np.array(pred_test)*(max_temp-min_temp)+min_temp

target_res=np.array(ori_target).reshape(-1)
pred_res=np.array(ori_pred).reshape(-1)
mse_orig=mean_squared_error(np.array(test_label_set).reshape(-1),np.array(pred_test).reshape(-1))
rmse = np.sqrt(mean_squared_error(target_res, pred_res))
abs_err = np.mean(np.abs(target_res - pred_res))
rel_abs_err=abs_err/target_res.mean()
rel_rmse=rmse/target_res.mean()
R = pearsonr(target_res, pred_res)

#%% train_vali performance

train_perf_dir=folder+model_perf_dir
train_perf=np.loadtxt(train_perf_dir)
opt_vali= train_perf.min(0)
train_perf_opt=train_perf[np.where(train_perf[:,2]==opt_vali[2])]
assess_metric=[train_perf_opt.squeeze()[1],mse_orig,rmse,rel_abs_err,rel_rmse,R[0]]
assess_metric_np=np.array(assess_metric)
metric_dir="test_perf/"+model_perf_dir
np.savetxt(metric_dir,assess_metric_np)

#%%
# plt.plot(np.arange(0,510),train_perf[:,1])
# plt.plot(np.arange(0,510),train_perf[:,2])
# plt.show()
plt.figure()
location_opt=np.where(train_perf[:,2]==opt_vali[2])
plt.subplot(2,1,1)
epoch=np.arange(0,510)
plt.plot(epoch, train_perf[:,1], 'b', label = 'Training')
plt.plot(epoch, train_perf[:,2], 'r', label = 'Validation')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim([0,550])
plt.ylim([0.01,0.19])
plt.xticks(np.arange(0,551,50))
plt.yticks(np.arange(0.01,0.191,0.02))
plt.legend(loc = 'upper right')


plt.subplot(2,1,2)
plt.plot(epoch, train_perf[:,0], 'r')
# ax2.scatter(x, P, color = 'r', marker = '*', s = 50)
plt.xlabel('Epoch')
plt.ylabel('Learning rate')
plt.xlim([0,550])
plt.ylim([0,1.e-5])
plt.xticks(np.arange(0,551,50))
plt.yticks(np.arange(0,1.01e-5,2e-6))
#图例
plt.tight_layout()
plt.show()
plt.close()
#%%
pred_test_save=np.array(pred_test)
np.savetxt("pred_test_resnet.txt",pred_test_save)