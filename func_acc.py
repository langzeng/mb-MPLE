import torch
import torch.nn as nn
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
import torchtuples as tt

def get_result(n_sample,true_beta_dim,batch_size,lr,lr_decrease_flag,epoch,ite,sim_type = "Cox",long_format = True,default_loc = "Result/Simulation/n"):
    
    if sim_type == "Cox":
        true_beta = [1 for i in range(true_beta_dim)]
        dir_save = default_loc+str(n_sample)+'_dim'+str(true_beta_dim)+'_batch'+str(batch_size)+'_lr'+str(lr)+'_d'+str(lr_decrease_flag)+'_e'+str(epoch)+'_ite'+str(ite)
        with open(dir_save+"/simulation_result.pkl", "rb") as fp:   # Unpickling
            r = pickle.load(fp)
            
        if not long_format:
            return r

        result_wide = pd.DataFrame({'sSBGD': np.log(np.square(r['beta_SBGDR']-np.array(true_beta)).sum(axis = 1)), 
                                   'sFBGD': np.log(np.square(r['beta_FBGDR']-np.array(true_beta)).sum(axis = 1)),
                                   'SBGD': np.log(np.square(r['beta_SBGD']-np.array(true_beta)).sum(axis = 1)),
                                   'FBGD': np.log(np.square(r['beta_FBGD']-np.array(true_beta)).sum(axis = 1)),
                                   'CoxPH': np.log(np.square(r['beta_CoxPH']-np.array(true_beta)).sum(axis = 1)),
                                   'CoxPH-strata': np.log(np.square(r['beta_CoxPH_strata']-np.array(true_beta)).sum(axis = 1)),
                                   'FBGD-CoxPH_strata':np.log(np.square(np.subtract(r['beta_FBGD'],r['beta_CoxPH_strata'])).sum(axis = 1)),
                                   'n_sample' : n_sample,
                                   'batch_size' : batch_size})

        result_wide = result_wide.reset_index()
        result_long = pd.melt(result_wide, id_vars=['n_sample','batch_size'], value_vars=['sSBGD', 'sFBGD', 'SBGD','FBGD','CoxPH','CoxPH-strata','FBGD-CoxPH_strata'])

        return result_long
    
    if sim_type == "Cov":
        dir_save = "/ix1/yding/laz52/SurvML/Result/MiniBatchSGD_CoxPH/Simulation_Cov/n"+str(n_sample)+'_batch'+str(batch_size)+'_lr'+str(lr)+'_d'+str(lr_decrease_flag)+'_e'+str(epoch)+'_ite'+str(ite)
        with open(dir_save+"/simulation_result.pkl", "rb") as fp:   # Unpickling
            r = pickle.load(fp)
        
        true_beta = np.square(r['true_beta'][0])
        if not long_format:
            return r

        SBGD_ora_var = np.var(r['beta_oracle']-np.array(true_beta))
        FBGD_ora_var = np.var(r['beta_oracle_FBGD']-np.array(true_beta))
        SBGD_ora_bias2 = np.square(np.mean(r['beta_oracle']-np.array(true_beta)))
        FBGD_ora_bias2 = np.square(np.mean(r['beta_oracle_FBGD']-np.array(true_beta)))
        result_wide = pd.DataFrame({'s-SBGD-num': np.log(np.square(np.subtract(np.squeeze(r['beta_SBGDR']),r['beta_oracle']))), 
                                   's-FBGD-num': np.log(np.square(np.subtract(np.squeeze(r['beta_FBGDR']),r['beta_oracle_FBGD']))),
                                   'SBGD-num': np.log(np.square(np.subtract(np.squeeze(r['beta_SBGD']),r['beta_oracle']))),
                                   'FBGD-num': np.log(np.square(np.subtract(np.squeeze(r['beta_FBGD']),r['beta_oracle_FBGD']))),
                                   'SBGD-ora': np.log(np.mean(np.square(r['beta_oracle']-np.array(true_beta)))),
                                   'FBGD-ora': np.log(np.mean(np.square(r['beta_oracle_FBGD']-np.array(true_beta)))),
                                   # 'Oracle': np.log(np.square(r['beta_oracle']-np.array(true_beta))),
                                   'n_sample' : n_sample,
                                   'batch_size' : batch_size})

        result_wide = result_wide.reset_index()
        result_long = pd.melt(result_wide, id_vars=['n_sample','batch_size'], value_vars=['s-SBGD-num', 's-FBGD-num', 'SBGD-num','FBGD-num','SBGD-ora','FBGD-ora'])
        
        output = {
            "result_long": result_long,
            "SBGD-ora-var": SBGD_ora_var,
            "FBGD-ora-var": FBGD_ora_var,
            "SBGD-ora-bias2": SBGD_ora_bias2,
            "FBGD-ora-bias2": FBGD_ora_bias2
        }
        
        return output
    
    if sim_type == "CoxCC":
        dir_save = "Result/Simulation_CoxCC/n"+str(n_sample)+'_dim'+str(true_beta_dim)+'_batch'+str(batch_size)+'_lr'+str(lr)+'_d'+str(lr_decrease_flag)+'_e'+str(epoch)+'_ite'+str(ite)
        with open(dir_save+"/simulation_result.pkl", "rb") as fp:   # Unpickling
            r = pickle.load(fp)
            
        if not long_format:
            return r
        
        result_wide = pd.DataFrame({
                           'SBGD': np.log(np.square(np.array(r['beta_SBGD'])-np.array(r['true_beta'])).sum(axis = 1)),
                           # 'FBGD': np.log(np.square(np.array(r['beta_FBGD'])-np.array(r['true_beta'])).sum(axis = 1)),
                           'n_sample' : n_sample,
                           'batch_size' : batch_size})

        result_wide = result_wide.reset_index()
        result_long = pd.melt(result_wide, id_vars=['n_sample','batch_size'], value_vars=['SBGD'])

        return result_long
    
    
class CoxSGD:
    def __init__(self,n_sample,true_beta_dim,epoch,batch_size,optimizer,lr,lr_decrease_flag,seed,beta_init = None,censor_scale = 0.02):
        self.n_sample = n_sample
        self.true_beta_dim = true_beta_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.optimizer = optimizer
        if lr == 0:
            self.lr = batch_size/32
        else:
            self.lr = lr
        self.lr_decrease_flag = lr_decrease_flag
        self.seed = int(seed)
        self.beta_init = beta_init
        self.censor_scale = censor_scale
        
    def fit(self):
        self.true_beta = [1 for i in range(self.true_beta_dim)]
        
        if self.beta_init is None:
            # self.beta_init = np.random.uniform(0.5, 1.5, len(self.true_beta))
            self.beta_init = [0 for i in range(self.true_beta_dim)]
        
        # fit CoxPH
        surv_data = SurvData(self.n_sample,self.true_beta,seed = self.seed,scale = self.censor_scale) # n_sample true_beta
        cph1 = CoxPHFitter()
        cph2 = CoxPHFitter()

        # self.loss_cph = -cph.log_likelihood_.item()/self.n_sample
        df_x = pd.DataFrame(surv_data.x.numpy().reshape(-1,self.true_beta_dim), columns=['x'+str(i+1) for i in range(self.true_beta_dim)])
        df_y = pd.DataFrame({'time' : surv_data.y.numpy()[:,0],
                             'event' : surv_data.y.numpy()[:,1],
                             'batch' : np.repeat(np.arange(self.n_sample/self.batch_size),self.batch_size)})

        cph1.fit(pd.concat([df_x, df_y.drop(columns=['batch'])],axis = 1), 
                duration_col = 'time', 
                event_col = 'event')
        # loss_cph = -cph1.log_likelihood_.item()/n_sample
        self.beta_CoxPH = cph1.params_

        cph2.fit(pd.concat([df_x, df_y],axis = 1),
                 duration_col = 'time', 
                 event_col = 'event',
                 strata=['batch'])
        self.beta_CoxPH_strata = cph2.params_
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # optimize with FBGD
        start = time.time()
        self.beta_FBGD,self.list_beta_FBGD,self.list_loss_FBGD = one_simulation(surv_data,
                                                                 self.beta_init, # initial parameter
                                                                 self.epoch, # epoch
                                                                 self.batch_size, # batch_size
                                                                 True, # fixed_batch
                                                                 True, # batch_split_data
                                                                 self.optimizer, # optimizer
                                                                 self.lr, # optimizer lr
                                                                 self.lr_decrease_flag) # lr_decrease_flag
        end = time.time()
        self.t_FBGD = end - start

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # optimize with SBGD
        start = time.time()
        self.beta_SBGD,self.list_beta_SBGD,self.list_loss_SBGD = one_simulation(surv_data,
                                                                 self.beta_init, # initial parameter
                                                                 self.epoch, # epoch
                                                                 self.batch_size, # batch_size
                                                                 False, # fixed_batch
                                                                 True, # batch_split_data
                                                                 self.optimizer, # optimizer
                                                                 self.lr, # optimizer lr
                                                                 self.lr_decrease_flag) # lr_decrease_flag
        end = time.time()
        self.t_SBGD = end - start
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # optimize with FBGDR
        start = time.time()
        self.beta_FBGDR,self.list_beta_FBGDR,self.list_loss_FBGDR = one_simulation(surv_data,
                                                                    self.beta_init, # initial parameter
                                                                    self.epoch, # epoch
                                                                    self.batch_size, # batch_size
                                                                    True, # fixed_batch
                                                                    False, # batch_split_data
                                                                    self.optimizer, # optimizer
                                                                    self.lr, # optimizer lr
                                                                    self.lr_decrease_flag) # lr_decrease_flag
        end = time.time()
        self.t_FBGDR = end - start
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # optimize with SBGDR
        start = time.time()
        self.beta_SBGDR,self.list_beta_SBGDR,self.list_loss_SBGDR = one_simulation(surv_data,
                                                                    self.beta_init, # initial parameter
                                                                    self.epoch, # epoch
                                                                    self.batch_size, # batch_size
                                                                    False, # fixed_batch
                                                                    False, # batch_split_data
                                                                    self.optimizer, # optimizer
                                                                    self.lr, # optimizer lr
                                                                    self.lr_decrease_flag) # lr_decrease_flag
        end = time.time()
        self.t_SBGDR = end - start
        
    def plot_loss(self):
        plt.plot(self.list_loss_SBGDR, linestyle = 'dotted', label='SBGD-R')
        plt.plot(self.list_loss_FBGDR,linestyle = 'dashdot', label='FBGD-R')
        plt.plot(self.list_loss_SBGD, linestyle = 'dashed', label='SBGD')
        plt.plot(self.list_loss_FBGD,linestyle = 'solid', label='FBGD')
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel(self.lr_decrease_flag)
        plt.show
        
    def plot_se(self):
        plt.axhline(y = np.log(np.square(np.subtract(self.beta_CoxPH,self.true_beta)).sum()) , color = 'k', linestyle = 'solid', linewidth = '3', label = "CoxPH")
        plt.plot(np.log(np.square(np.subtract(self.list_beta_SBGDR,self.true_beta)).sum(axis=1)), linestyle = 'dotted', label='SBGD-R')
        plt.plot(np.log(np.square(np.subtract(self.list_beta_FBGDR,self.true_beta)).sum(axis=1)),linestyle = 'dashdot', label='FBGD-R')
        plt.plot(np.log(np.square(np.subtract(self.list_beta_SBGD,self.true_beta)).sum(axis=1)), linestyle = 'dashed', label='SBGD')
        plt.plot(np.log(np.square(np.subtract(self.list_beta_FBGD,self.true_beta)).sum(axis=1)),linestyle = 'solid', label='FBGD')
        plt.legend()
        plt.ylabel("log(SE)")
        plt.xlabel(self.lr_decrease_flag)
        plt.show
        
    def plot_beta(self):
        plt.plot([i[0] for i in self.list_beta_SBGDR],[i[1] for i in self.list_beta_SBGDR],marker = "x", label='SGD')
        plt.plot([i[0] for i in self.list_beta_FBGDR],[i[1] for i in self.list_beta_FBGDR],marker = ".", label='FBGD-R')
        plt.plot([i[0] for i in self.list_beta_SBGD],[i[1] for i in self.list_beta_SBGD],marker = "p", label='SBGD')
        plt.plot([i[0] for i in self.list_beta_FBGD],[i[1] for i in self.list_beta_FBGD],marker = "h", label='FBGD')
        plt.plot([self.beta_CoxPH[0]],[self.beta_CoxPH[1]],"ks",label="CoxPH")
        plt.plot([self.true_beta[0]],[self.true_beta[1]],"r*",label="true")
        plt.plot([self.beta_init[0]],[self.beta_init[1]],"kD",label="start")
        plt.xlim(-2, 4)
        plt.ylim(-2, 4)
        plt.legend()
        plt.show()
    
    def plot_all(self):
        plt.rcParams['figure.figsize'] = [10,5]
        # plt.subplot(1,3, 1)
        # self.plot_loss()
        plt.subplot(1,2, 1)
        self.plot_se()
        plt.subplot(1,2, 2)
        self.plot_beta()
        plt.tight_layout()
        
        
class SurvData:
    def __init__(self,n_sample,betax,scale = 0.02,seed = None):
        np.random.seed(seed)
        x_tmp = np.random.uniform(size = (n_sample,len(betax)))
        h = np.exp(np.dot(x_tmp,betax))
        c = np.random.exponential(scale= scale, size= n_sample)
        u = np.random.uniform(0, 1,n_sample)
        # baseline_hazard = 1
        true_t = -np.log(u)/h
        y_tmp = np.column_stack([np.minimum(true_t,c),np.less_equal(true_t,c)])
        
        if len(betax) == 1:
            self.x = torch.tensor(x_tmp)[:,None]
        else:
            self.x = torch.tensor(x_tmp)
        self.y = torch.tensor(y_tmp)
        self.len=self.x.shape[0]
        
        del(x_tmp,h,c,u,true_t,y_tmp)
        
    def __getitem__(self,index):    
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    
    
def one_simulation(surv_data, beta_init, epoch, batch_size, fixed_batch, batch_split_data, optimizer_f,optimizer_lr,lr_decrease_flag = None):
    beta = torch.autograd.Variable(torch.tensor(beta_init,dtype=torch.float64), requires_grad=True)
    optimizer = optimizer_f([beta], lr = optimizer_lr)
    n_sample = len(surv_data)
    n_batch = int(n_sample/batch_size)
    
    if lr_decrease_flag == 'ite':
        lambda1 = lambda ite: 1/(ite+1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    
    if lr_decrease_flag == 'epoch':
        lambda1 = lambda epoch: 1/(epoch+1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
        
    list_beta = [beta_init]
    list_loss = []
    
    ite = 0
    if batch_split_data:
        trainloader=DataLoader(dataset=surv_data,batch_size=batch_size,shuffle=(not fixed_batch))
        for epoch in range(epoch):
            for x,y in trainloader:
                optimizer.zero_grad()
                y_pred = torch.mm(x.view(-1,len(beta)),beta[:,None])
                loss=loss_CoxPH(y_pred,y)
                if not torch.is_nonzero(loss):
                    continue
                loss.backward()
                optimizer.step()
  
                if lr_decrease_flag == 'ite':
                    list_loss.append(loss.detach().clone().item())
                    list_beta.append(beta.detach().clone().numpy()) 
                    scheduler.step()
                    ite += 1
                    
            if lr_decrease_flag != 'ite':
                list_loss.append(loss.detach().clone().item())
                list_beta.append(beta.detach().clone().numpy()) 
            if lr_decrease_flag == 'epoch':
                scheduler.step()
                
    elif fixed_batch: # stochastic fixed_batch SGD
        for epoch in range(epoch):
            for i in range(n_batch):
                optimizer.zero_grad()
                
                k = np.random.choice(range(int(n_batch)))
                k1 = k*batch_size
                k2 = k1+batch_size
                
                batch_x = surv_data.x[k1:k2].view(-1,len(beta))
                batch_y = surv_data.y[k1:k2]
                
                y_pred = torch.mm(batch_x,beta[:,None])
                loss=loss_CoxPH(y_pred,batch_y)
                if not torch.is_nonzero(loss):
                    continue
                loss.backward()
                optimizer.step()
  
                if lr_decrease_flag == 'ite':
                    list_loss.append(loss.detach().clone().item())
                    list_beta.append(beta.detach().clone().numpy()) 
                    scheduler.step()
                    ite += 1
                    
            if lr_decrease_flag != 'ite':
                list_loss.append(loss.detach().clone().item())
                list_beta.append(beta.detach().clone().numpy()) 
            if lr_decrease_flag == 'epoch':
                scheduler.step()
                
    else: # stochastic batch SGD
        train_sampler = RandomSampler(surv_data,replacement=False,num_samples=batch_size)
        trainloader=DataLoader(dataset=surv_data,
                               shuffle=False,
                               batch_size=batch_size,
                               sampler = train_sampler)
        for epoch in range(epoch):
            for i in range(n_batch):
                x,y = next(iter(trainloader))
                optimizer.zero_grad()
                y_pred = torch.mm(x.view(-1,len(beta)),beta[:,None])
                loss=loss_CoxPH(y_pred,y)
                if not torch.is_nonzero(loss):
                    continue
                loss.backward()
                optimizer.step()

                if lr_decrease_flag == 'ite':
                    list_loss.append(loss.detach().clone().item())
                    list_beta.append(beta.detach().clone().numpy()) 
                    scheduler.step()
                    ite += 1
            
            if lr_decrease_flag != 'ite':
                list_loss.append(loss.detach().clone().item())
                list_beta.append(beta.detach().clone().numpy()) 
            if lr_decrease_flag == 'epoch':
                scheduler.step()
                
    # print(scheduler.get_lr())
    output = beta.detach().clone().numpy()
    return output,list_beta,list_loss

def loss_CoxPH(y_pred, y_true):
    y_true = y_true.type(torch.float32) # tstart = 0, tstop, event
    y_pred = y_pred.type(torch.float32)
    y_pred = torch.flatten(y_pred)
    
    time = torch.flatten(y_true[:,0])
    time0 = torch.zeros_like(time)
    event = torch.flatten(y_true[:,1])
    
    n_size = event.shape[0]

    sort_index = torch.argsort(time)
    time0 = time0[sort_index] # ascending order
    time = time[sort_index]
    event = event[sort_index]
    y_pred = y_pred[sort_index]
    
    if torch.sum(event) == 0.: 
        return torch.sum(event)
    else:
        time_event = time * event
        eventtime,tie_count = torch.unique(torch.masked_select(time_event,torch.gt(time_event,0)),
                                           return_counts=True)

        at_risk_index = ((time0 < eventtime[:,None]) & (time >= eventtime[:,None])).type(torch.float32)
        event_index = (time == eventtime[:,None]).type(torch.float32)
        
        # haz = exp(risk)
        tie_haz = torch.mm(event_index, (torch.exp(torch.clip(y_pred,-20,20))*event).unsqueeze(1))
        
        tie_risk = torch.mm(event_index, (y_pred*event).unsqueeze(1))

        cum_haz = torch.mm(at_risk_index, (torch.exp(torch.clip(y_pred,-20,20))).unsqueeze(1))

        mask_tie_haz = torch.arange(torch.max(tie_count)) < (tie_count[:,None]-1)
        mask_tie_risk = torch.arange(torch.max(tie_count)) < (tie_count[:,None])
        out0 = torch.zeros(mask_tie_haz.size(),dtype = torch.float32)
        out1 = (torch.cumsum(torch.ones(mask_tie_haz.size()),1)).type(torch.float32)
        out = torch.where(mask_tie_haz, out1, out0)
        tie_count_matrix = tie_count.type(torch.float32).unsqueeze(1)

        J = torch.divide(out,tie_count_matrix)
        efron_correction = J*tie_haz
        log_sum_haz = torch.where(mask_tie_risk,torch.ones(mask_tie_risk.size(),dtype = torch.float32),out0)*cum_haz
        log_sum_haz = torch.where(mask_tie_risk,torch.log(log_sum_haz-efron_correction+1e-15),out0)
        log_sum_haz = torch.sum(log_sum_haz)
        log_lik = torch.sum(tie_risk)-log_sum_haz
        
        return torch.negative(log_lik/n_size)
    
    
class Net(nn.Module):

    def __init__(self,num_input,num_hidden_node,dropout_rt = None,output_bias = False):
        super(Net, self).__init__()
        self.dropout_rt = dropout_rt
        
        self.hidden1 = torch.nn.Linear(num_input, num_hidden_node)
        self.act1 = nn.ReLU()
        self.hidden2 = torch.nn.Linear(num_hidden_node, num_hidden_node)
        self.act2 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(num_hidden_node)
        self.batchnorm2 = nn.BatchNorm1d(num_hidden_node)
        self.output = nn.Linear(num_hidden_node, 1,output_bias)
        
        if dropout_rt is not None:
            self.dropout1 = nn.Dropout(p=dropout_rt)
            self.dropout2 = nn.Dropout(p=dropout_rt)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        # x = self.batchnorm1(x)
        
        if self.dropout_rt is not None:
            x = self.batchnorm1(x)
            x = self.dropout1(x)
            x = self.act2(self.hidden2(x))
            x = self.batchnorm2(x)
            x = self.dropout2(x)
        else:
            # x = self.batchnorm1(x)
            x = self.act2(self.hidden2(x))
            # x = self.batchnorm2(x)
        
        x = self.output(x)
        return x

def g0(x):
    g = ((x[0]**2)*(x[1]**3)+np.log(x[2]+1)+np.sqrt(x[3]*x[4]+1)+np.exp(x[4]/2))**2 - 8.6
    return g
    
def SurvDataNN(n_sample,scale = 5,seed = 0,torch_flag = True):
    np.random.seed(seed)
    x1 = np.random.uniform(0,1,size = n_sample)
    x2 = np.random.uniform(0,1,size = n_sample)
    x3 = np.random.uniform(0,1,size = n_sample)
    x4 = np.random.uniform(0,1,size = n_sample)
    x5 = np.random.uniform(0,1,size = n_sample)

    g = ((x1**2)*(x2**3)+np.log(x3+1)+np.sqrt(x4*x5+1)+np.exp(x5/2))**2 - 8.6
    g = g - np.mean(g)
    h = np.exp(g)
    
    g = g.astype(np.float32)
    
    c = np.random.exponential(scale= scale, size= n_sample)
    u = np.random.uniform(0, 1, n_sample)
    true_t = -np.log(u)/h
    y = np.column_stack([np.minimum(true_t,c),np.less_equal(true_t,c)]).astype(np.float32)

    x = np.column_stack([x1,x2,x3,x4,x5])
    # x += np.random.normal(0,0.01,size = x.shape)
    x = x.astype(np.float32)
    
    if torch_flag:
        return torch.tensor(x),torch.tensor(y),torch.tensor(g)
    else:
        return x,y,g

class NN_CoxSNN:
    def __init__(self,
                 batch_size,
                 n_sample = 2048,
                 num_hidden_node = 16,
                 dropout_rt = 0.2,
                 lr = 0.1,
                 epochs = 100,
                 torch_seed = 13579,
                 train_data_seed = 1234,
                 test_data_seed = 56789,
                 optimizer_choice = optim.SGD,
                 loss_choice = loss_CoxPH):
        
        x_train,y_train,g_train_true = SurvDataNN(n_sample, scale = 5, seed = train_data_seed)
        x_test,y_test,g_test_true = SurvDataNN(n_sample, scale = 5, seed = test_data_seed)
        in_features = x_train.shape[1]

        self.event_rate = torch.mean(y_train[:,1]).item()

        # Model
        torch.manual_seed(torch_seed)
        NN = Net(in_features,num_hidden_node,dropout_rt)

        # Dataloader
        train_dataloader = DataLoader(traindata_custom(x_train,y_train), batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = optimizer_choice(NN.parameters(), lr = lr)

        # Store Loss
        NN_trainingEpoch_loss = []
        NN_trainingEpoch_FullPL = []
        NN_testEpoch_FullPL = []

        # Train
        NN.eval()

        with torch.no_grad():
            y_pred_test = NN(x_test)
            NN_testEpoch_FullPL.append(loss_choice(y_pred_test,y_test).item())

            y_pred_train = NN(x_train)
            NN_trainingEpoch_FullPL.append(loss_choice(y_pred_train,y_train).item())

        for epoch in range(epochs):
            NN.train()
            for x,y in train_dataloader:
                if torch.sum(y[:,1]) == 0.:
                    continue
                optimizer.zero_grad()
                y_pred = NN(x)
                loss=loss_choice(y_pred,y)
                loss.backward()
                optimizer.step()

            NN_trainingEpoch_loss.append(loss.item())

            NN.eval()
            with torch.no_grad():
                y_pred_test = NN(x_test)
                NN_testEpoch_FullPL.append(loss_choice(y_pred_test,y_test).item())

                y_pred_train = NN(x_train)
                NN_trainingEpoch_FullPL.append(loss_choice(y_pred_train,y_train).item())

        self.MSE100 = torch.mean(torch.square(y_pred_test.squeeze()-y_pred_test.squeeze().mean()-g_test_true)).item()
        self.RE100 = torch.sqrt(torch.mean(torch.square(y_pred_test.squeeze()-y_pred_test.squeeze().mean()-g_test_true))/torch.mean(torch.square(g_test_true))).item()

        self.NN_trainingEpoch_loss = NN_trainingEpoch_loss
        self.NN_trainingEpoch_FullPL = NN_trainingEpoch_FullPL
        self.NN_testEpoch_FullPL = NN_testEpoch_FullPL
        self.NN = NN
        

        
class NN_regression:
    def __init__(self,
                 batch_size,
                 n_sample = 2048,
                 num_hidden_node = 16,
                 dropout_rt = 0.2,
                 lr = 0.1,
                 epochs = 100,
                 torch_seed = 13579,
                 train_data_seed = 1234,
                 test_data_seed = 5678,
                 optimizer_choice = optim.SGD):
        
        x_train,y_train,g_train_true = SurvDataNN(n_sample, scale = 5, seed = train_data_seed)
        x_test,y_test,g_test_true = SurvDataNN(n_sample, scale = 5, seed = test_data_seed)
        in_features = x_train.shape[1]
        
        g_train_true = g_train_true.reshape(-1,1)
        g_test_true = g_test_true.reshape(-1,1)

        # Model
        torch.manual_seed(torch_seed)
        NN = Net(in_features,num_hidden_node,dropout_rt,output_bias = True)

        # Dataloader
        train_dataloader = DataLoader(traindata_custom(x_train,g_train_true), batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optimizer_choice(NN.parameters(), lr = lr)
        
        # Store Loss
        NN_trainingEpoch_loss = []
        NN_trainingEpoch_FullPL = []
        NN_testEpoch_FullPL = []

        # Train
        NN.eval()

        with torch.no_grad():
            y_pred_test = NN(x_test)
            NN_testEpoch_FullPL.append(nn.functional.mse_loss(y_pred_test,g_test_true).item())

            y_pred_train = NN(x_train)
            NN_trainingEpoch_FullPL.append(nn.functional.mse_loss(y_pred_train,g_train_true).item())

        for epoch in range(epochs):
            NN.train()
            for x,y in train_dataloader:
                optimizer.zero_grad()
                y_pred = NN(x)
                loss=nn.functional.mse_loss(y_pred,y)
                loss.backward()
                optimizer.step()

            NN_trainingEpoch_loss.append(loss.item())

            NN.eval()
            with torch.no_grad():
                y_pred_test = NN(x_test)
                NN_testEpoch_FullPL.append(nn.functional.mse_loss(y_pred_test,g_test_true).item())

                y_pred_train = NN(x_train)
                NN_trainingEpoch_FullPL.append(nn.functional.mse_loss(y_pred_train,g_train_true).item())

        self.MSE100 = torch.mean(torch.square(y_pred_test-y_pred_test.mean()-g_test_true)).item()
        self.RE100 = torch.sqrt(torch.mean(torch.square(y_pred_test-y_pred_test.mean()-g_test_true))/torch.mean(torch.square(g_test_true))).item()
        self.NN_trainingEpoch_loss = NN_trainingEpoch_loss
        self.NN_trainingEpoch_FullPL = NN_trainingEpoch_FullPL
        self.NN_testEpoch_FullPL = NN_testEpoch_FullPL
        self.NN = NN
        
class traindata_custom(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.len = y.shape[0]

    def __getitem__(self,index):    
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    