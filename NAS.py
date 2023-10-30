from architectures import *
import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import time
from sklearn.metrics import accuracy_score

archs = [
    # FC
    FC1_Network,
    FC2_Network,
    FC3_Network,
    FC4_Network,
    # CNN1
    CNN1_5_Network,
    CNN1_10_Network,
    CNN1_15_Network,
    CNN1_20_Network,
    # CNN2
    CNN2_5_10_Network,
    CNN2_10_15_Network,
    CNN2_15_20_Network,
    CNN2_20_30_Network,
    CNN2_25_40_Network,
    # CNN3
    CNN3_5_10_20_Network,
    CNN3_10_20_30_Network
]

results = np.full((len(archs), 10, 3), np.nan) # acc, time, support

# DATA
train_data = torchvision.datasets.SVHN('./files/', 
                                  split='test', #Tak.
                                  download=True)

X = (torch.tensor(train_data.data)/255).to(torch.float)
y = train_data.labels
print(X.shape)
print(y.shape)

training_support_level = 0.9
max_training_epochs = 250

# FOLDS
skf = StratifiedKFold(random_state=1223, shuffle=True)

for fold, (train, test) in enumerate(skf.split(np.zeros(len(y)), y)):
    
    dataset = TensorDataset(torch.Tensor(X[train]),torch.Tensor(y[train]))
    dataloader = DataLoader(dataset, batch_size=64)
    
    for a_id, a in enumerate(archs):
        
        clf = CNN(architecure=a(x_input_size=32, img_depth=3))
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(clf.parameters(), lr=1e-2)

        # TRAINING
        for e in range(max_training_epochs):
            if e==0:
                clf.custom_train(dataloader, loss_fn, optimizer)
            else:

                proba = nn.Softmax(dim=1)(clf(X[train]))
                max_proba = torch.max(proba, dim=1)[0] 
                mean_proba = torch.mean(max_proba).detach().numpy() # średnie wsparcie decyzyjne
                
                if e%50==1:
                    print(a_id, e, mean_proba)

                if mean_proba>training_support_level:
                    break
                
                clf.custom_train(dataloader, loss_fn, optimizer)
            
        # TESTING
        st = time.time()
        proba = nn.Softmax(dim=1)(clf(X[test]))
        p = torch.argmax(proba, dim=1)    
        el = time.time() - st
        
        # STORE
        results[a_id, fold, 0] = accuracy_score(y[test], p)
        results[a_id, fold, 1] = el
        results[a_id, fold, 2] = mean_proba
        
        print(a_id, fold, results[a_id, fold])
        
        np.save('nas.npy', results)
        
            



