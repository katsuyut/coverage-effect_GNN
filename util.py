import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_loss(model,data_list):
    totalloss = 0
    for data in data_list:
        data = data.to(device)
        out = model(data)
        loss = F.mse_loss(out, data.y) 
        totalloss += loss
    return totalloss/len(data_list)


def get_plot(model,data_list,case='noname',save='no'):
    y = []
    ypred = []
    for data in data_list:
        data = data.to(device)
        out = model(data)
        ypred.append(out)
        y.append(data.y)
    ypred = torch.tensor(ypred)
    y = torch.tensor(y)
    pp = np.linspace(min(y),max(y))
    plt.figure()
    plt.plot(y,ypred,'.')
    plt.plot(pp,pp)
    plt.savefig('./figs/'+case+'.png')
    plt.show()