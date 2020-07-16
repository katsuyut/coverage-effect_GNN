import torch
import torch.nn.functional as F
from torch_geometric.nn import CGConv

class simpleNet(torch.nn.Module):
    def __init__(self, in_ch, nconv):
        super(Net, self).__init__()
        self.conv1 = CGConv(in_ch)
        self.conv2 = CGConv(in_ch)
        self.lin = torch.nn.Linear(in_ch, 1)
        self.nconv = nconv
    
    def forward(self, data):
        x, edge_index, surf_filter = data.x, data.edge_index, data.surf_filter
        
        if self.nconv==1:
            x = self.conv1(x, edge_index)
        elif self.nconv==2:
            x = self.conv1(x, edge_index)
            x = self.conv2(x, edge_index)
        x = x * surf_filter.reshape(-1,1)
        x = self.lin(x)
        y = torch.sum(x) / torch.sum(surf_filter)
        
        return y


class comp1Net(torch.nn.Module):
    def __init__(self, in_ch_sl, in_ch_ad, nconv):
        super(Net, self).__init__()
        self.conv1_sl = CGConv(in_ch_sl)
        self.conv2_sl = CGConv(in_ch_sl)
        self.conv1_ad = CGConv(in_ch_ad)
        self.conv2_ad = CGConv(in_ch_ad)
        self.lin_sl = torch.nn.Linear(in_ch_sl, 1)
        self.lin_ad = torch.nn.Linear(in_ch_ad, 1)
        self.nconv = nconv
    
    def forward(self, data):
        x, edge_index, surf_filter = data.x, data.edge_index, data.surf_filter
        
        x_sl = x[:,:8]
        x_ad = x[:,8:]
        if self.nconv==1:
            x_sl = self.conv1_sl(x[:,:8], edge_index)
            x_ad = self.conv1_ad(x[:,8:], edge_index)
        elif self.nconv==2:
            x_sl = self.conv2_sl(x_sl, edge_index)
            x_ad = self.conv2_ad(x_ad, edge_index)

        x_sl = self.lin_sl(x_sl)
        x_ad = self.lin_ad(x_ad)
        x = x_sl * x_ad
        x = x * surf_filter.reshape(-1,1)
        y = torch.sum(x) / torch.sum(surf_filter)
        
        return y


class comp2Net(torch.nn.Module):
    def __init__(self, in_ch_sl, in_ch_ad, nconv):
        super(Net, self).__init__()
        self.conv1_sl = CGConv(in_ch_sl)
        self.conv2_sl = CGConv(in_ch_sl)
        self.conv1_ad = CGConv(in_ch_ad)
        self.conv2_ad = CGConv(in_ch_ad)
        self.lin_sl = torch.nn.Linear(in_ch_sl, in_ch_ad)
        self.lin = torch.nn.Linear(in_ch_ad, 1)
        self.nconv = nconv
    
    def forward(self, data):
        x, edge_index, surf_filter = data.x, data.edge_index, data.surf_filter
        
        x_sl = x[:,:8]
        x_ad = x[:,8:]
        if self.nconv==1:
            x_sl = self.conv1_sl(x[:,:8], edge_index)
            x_ad = self.conv1_ad(x[:,8:], edge_index)
        elif self.nconv==2:
            x_sl = self.conv2_sl(x_sl, edge_index)
            x_ad = self.conv2_ad(x_ad, edge_index)

        x_sl = self.lin_sl(x_sl)
        x = x_sl * x_ad
        x = x * surf_filter.reshape(-1,1)
        x = self.lin(x)
        y = torch.sum(x) / torch.sum(surf_filter)
        
        return y