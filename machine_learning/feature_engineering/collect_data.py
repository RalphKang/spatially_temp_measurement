import os
import numpy as np
import pandas as pd
"""
This script is used to collect data from the raw data folder and save it in a new dataset
"""
#%% read the signal-representation features from the folder
data_folder = 'power_20'
#%% read the data
data_record = []
for i in range(0,24022):
    file_dir = os.path.join(data_folder, str(i)+'.csv')
    data_record.append(pd.read_csv(file_dir, header=None))
#%% transform the data into numpy array
data_record_np = np.array(data_record).squeeze()
# save the data
save_folder = "fitting_data"
save_dir = os.path.join(save_folder, data_folder+'.csv')
np.savetxt(save_dir, data_record_np, delimiter=",")