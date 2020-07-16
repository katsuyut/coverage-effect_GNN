import os

datas = ['data','data_100','data_111','data_OH','data_OH_100','data_OH_111']
ytypes = ['slab', 'space', 'all']
mos = ['simple', 'comp1', 'comp2']
nconvs = [1,2]

nepoch=500
lr = 0.001
i=0

for data in datas:
    for ytype in ytypes:
        for mo in mos:
            for nconv in nconvs:
                message = 'python coverage_GNN.py --case="n%02d" --nepoch=%d --ytype="%s" --mo="%s" --nconv=%d --data="%s" --lr=%f --save="%s"' %(i,nepoch,ytype,mo,nconv,data,lr,'yes')
                print(message)
                os.system(message)
                i+=1