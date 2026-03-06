import torch
import numpy as np
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
import itertools
import time
import pickle
import os
from matplotlib import pyplot as plt
from lifelines import CoxPHFitter
from pathlib import Path
import time

# Set the working directory to the Code_ACC project directory
# dirc_server='Code_ACC'
# os.chdir(dirc_server)

# Load local defined functions
from func_acc import CoxSGD

import sys

# sys.argv = ['', '2048', '10', '256', '0' , 'epoch', '200', '2']

n_sample = int(sys.argv[1])
true_beta_dim = int(sys.argv[2])
batch_size = int(sys.argv[3])
lr = float(sys.argv[4])
lr_decrease_flag = sys.argv[5] # decrease the learning rate after epoch or iteration
epoch = int(sys.argv[6])
ite = int(sys.argv[7]) # number of repeated runs

# optimizer = optim.SGD     
dir_save = "../Results/n"+sys.argv[1]+'_dim'+sys.argv[2]+'_batch'+sys.argv[3]+'_lr'+sys.argv[4]+'_d'+sys.argv[5]+'_e'+sys.argv[6]+'_ite'+sys.argv[7]
Path(dir_save).mkdir(parents=True, exist_ok=True)

result = {
    'n_sample' : n_sample,
    'true_beta_dim' : true_beta_dim,
    'batch_size' : batch_size,
    'lr' : lr,
    'lr_decrease_flag' : lr_decrease_flag,
    'optimizer' : "optim.SGD",
    'epoch' : epoch,
    'ite' : ite,
    'true_beta' : [],
    'beta_CoxPH' : [],
    'beta_CoxPH_strata' : [],
    'beta_FBGD' : [],
    'beta_SBGD' : [],
}

for i_ite in range(ite):
    
    simulation = CoxSGD(n_sample,true_beta_dim,epoch,batch_size,optim.SGD,lr,lr_decrease_flag,i_ite*2,beta_init = None)
    simulation.fit()
    
    result['beta_CoxPH'].append(simulation.beta_CoxPH)
    result['beta_CoxPH_strata'].append(simulation.beta_CoxPH_strata)
    result['true_beta'].append(simulation.true_beta)
    
    result_loss = {
        'loss_FBGD' : simulation.list_loss_FBGD,
        'loss_SBGD' : simulation.list_loss_SBGD,
        'loss_beta_FBGD' : simulation.list_beta_FBGD,
        'loss_beta_SBGD' : simulation.list_beta_SBGD
    }
    with open(dir_save+"/loss"+str(i_ite)+".pkl", "wb") as fp:   #Pickling
        pickle.dump(result_loss, fp)

    result['beta_FBGD'].append(simulation.beta_FBGD)
    result['beta_SBGD'].append(simulation.beta_SBGD)

with open(dir_save+"/simulation_result.pkl", "wb") as fp:   #Pickling
    pickle.dump(result, fp)