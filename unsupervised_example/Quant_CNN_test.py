#
# Kang et al., " Unsupervised learning for magnetization transfer contrast MR fingerprinting: Application to CEST and nuclear Overhauser enhancement imaging"
# Magn Reson Med, 2021;85:2040-2054
#

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time
import os
import argparse
import h5py
from torch.utils.data import TensorDataset, DataLoader

from lib.Model_Quant import nnModel

parser = argparse.ArgumentParser(description='Setting')
parser.add_argument('--dir_data', type=str, default='data/')
parser.add_argument('--dir_model', type=str, default='model/')
parser.add_argument('--dir_result', type=str, default='result')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--ds', type=int, default=40)

args = parser.parse_args()
if not os.path.exists(args.dir_result):
    os.mkdir(args.dir_result)

# GPU
GPU_NUM = args.gpu # GPU number
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) # change allocation of current GPU
print ('Current cuda device ', torch.cuda.current_device()) # check
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

## Dataset #########################################################################
dir_data = args.dir_data
dir_model = args.dir_model

X_mat = sio.loadmat(dir_data + "/input_invivo_test_mtcmrf_PR40.mat")
test_X = X_mat['input_invivo_test']

X_test=torch.FloatTensor(test_X)

print(np.shape(test_X))
#####################################################################################
testset = TensorDataset(X_test)
testloader=DataLoader(testset,batch_size=1,shuffle=False)

## Model loading - Trained model #####
cnn = nnModel(args.ds,device)
PATH=dir_model+'/NN_model_UL.pth'
checkpoint=torch.load(PATH,map_location=device)
cnn.load_state_dict(checkpoint)
cnn = cnn.to(device)

TEST_DATASIZE = X_test.shape[0]
quantification_result=torch.zeros([TEST_DATASIZE,4,256,256],device=device)
## testidation
with torch.no_grad(): # important!
    test_loss = 0.0
    for j, data in enumerate(testloader):
        [X_batch]=data
        X_batch = X_batch.to(device)
                        
        x_pred_test = cnn(X_batch)
        quantification_result[j,:,:,:]=x_pred_test

    quantification_result=quantification_result.cpu()
    quantification_result=quantification_result.numpy()
    sio.savemat(args.dir_result+'/quantification_UL.mat',{'result_nn': quantification_result})

        




