#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import CGConv
from random import shuffle, randint
import networkx as nx
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
from models import simpleNet, comp1Net, comp2Net
from util import get_loss, get_plot
from statistics import stdev


# In[ ]:


parser = argparse.ArgumentParser(description='GNN for coverage effect')

parser.add_argument('--case',     default='noname', type=str)
parser.add_argument('--nepoch',   default=5,        type=int)
parser.add_argument('--mo',       default='simple', type=str)
parser.add_argument('--nconv',    default=1,        type=int)
parser.add_argument('--datatype', default='data',   type=str)
parser.add_argument('--ytype',    default='slab',   type=str)
parser.add_argument('--lr',       default=0.001,    type=float)
parser.add_argument('--save',     default='no',     type=str)

args = parser.parse_args()    # 4. 引数を解析

case = args.case
nepoch = args.nepoch
mo = args.mo
nconv = args.nconv
datatype = args.datatype
ytype = args.ytype
lr = args.lr
save = args.save


# In[ ]:


# case = 'noname'
# # nepoch = 10
# mo = 'simple'
# nconv = 1
# datatype = 'data'
# ytype = 'slab' # 3:slab, 4:space, 5:all
# # lr = 0.001
# save = 'no'


# In[ ]:


import pickle

with open('data.pickle', 'rb') as lb:
    rdata = pickle.load(lb)
    edge_indexes,features,surf_filters,ys_slab,ys_space,ys_all = rdata
    
allinone = []
for feature in features:
    for each in feature:
        allinone.append(each)
allinone = np.array(allinone)
scaler = StandardScaler().fit(allinone[:,:8])


# In[ ]:


with open(datatype+'.pickle', 'rb') as lb:
    rdata = pickle.load(lb)
    edge_indexes,features,surf_filters = rdata[:3]
    ys_slab,ys_space,ys_all = rdata[3:]


# In[ ]:


scfeatures = []

for feature in features:
    feature_sl = scaler.transform(feature[:,:8])
    feature_ad = feature[:,8:]
    scfeatures.append(np.concatenate((feature_sl, feature_ad), axis=1))
    
features = np.array(scfeatures)
# features


# In[ ]:


if ytype=='slab':
    ys = rdata[-3]
elif ytype=='space':
    ys = rdata[-2]
elif ytype=='all':
    ys = rdata[-1]


# In[ ]:


data_list = []
for i in range(len(ys_all)):
    edge_index = torch.tensor(edge_indexes[i], dtype=torch.long)
    x = torch.tensor(features[i], dtype=torch.float)
    surf_filter = torch.tensor(surf_filters[i], dtype=torch.long)
    y = torch.tensor(ys_slab[i], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, surf_filter=surf_filter, y=y)
    data_list.append(data)


# In[ ]:


in_ch = len(features[0][0])
in_ch_sl = 8
in_ch_ad = len(features[0][0]) - 8


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


import copy
data_list_ = copy.deepcopy(data_list) # all


# In[ ]:


if mo == 'simple':
    model = simpleNet(in_ch, nconv).to(device)
elif mo == 'comp1':
    model = comp1Net(in_ch_sl, in_ch_ad, nconv).to(device)
elif mo == 'comp2':
    model = comp2Net(in_ch_sl, in_ch_ad, nconv).to(device)


# In[ ]:


random.shuffle(data_list_)
train_list = data_list_[:int(len(data_list_)*0.67)]
test_list = data_list_[int(len(data_list_)*0.67):]


# In[ ]:


if mo=='simple':
    model = simpleNet(in_ch, nconv).to(device)
elif mo=='comp1':
    model = comp1Net(in_ch_sl, in_ch_ad, nconv).to(device)
elif mo=='comp2':
    model = comp2Net(in_ch_sl, in_ch_ad, nconv).to(device)    
    
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
loss_list = []


# In[ ]:


model.train()

for epoch in range(nepoch):
    for data in train_list:
        optimizer.zero_grad()
        data = data.to(device)
        out = model.train()(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
    loss_list.append(get_loss(model.eval(),train_list))
    if (epoch+1)%5==0:
        print('epoch %d RMSE :%f' % (epoch+1, loss_list[-1]**0.5))
    
torch.save(model.state_dict(), './models/'+case+'.pth')


# In[ ]:


plt.figure()
plt.plot(loss_list)
plt.yscale('log')
plt.savefig('./figs/'+case+'_loss.png')
plt.show()


# In[ ]:


model.eval()


# In[ ]:


get_plot(model,train_list,case+'_train','yes')


# In[ ]:


get_plot(model,test_list,case+'_test','yes')


# In[ ]:


trainloss = get_loss(model,train_list)**0.5
trainloss


# In[ ]:


testloss = get_loss(model,test_list)**0.5
testloss


# In[ ]:


allloss = get_loss(model,data_list)**0.5
allloss


# In[ ]:


with open('result.txt','a') as file:
    file.write('%s %f %f %f\n' % (case, trainloss, testloss, allloss))


# In[ ]:




