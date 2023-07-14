"""
Certainty-based Domain Selector
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

class CDoS:
    def __init__(self, clfs, thresholds, max_training_epochs, training_support_level, switch_when=10):
        self.clfs = clfs # lista klasyfikatorów
        self.thresholds = thresholds # lista progów (tej samej długości co lista klasyfiaktorów) i najwyższy próg musi nie być przekraczalny bo się wysypie
        self.switch_when = switch_when # ile pod rząd musi przekroczyć próg
        
        self.training_support_level = training_support_level # do tej wartości wsparcia uczone są wszystkie klasyfikatory
        self.max_training_epochs = max_training_epochs # ale jest limit
        
        self.curr_clf_id = 0 # zaczynamy od najgłupszego
        self.switch_count = 0 # licznik
            
        
    def partial_fit(self, X, y, classes):
        for clf in self.clfs:
            for e in range(self.max_training_epochs):
                try:
                    mean_proba = np.mean(np.max(clf.predict_proba(X), axis=1)) # średnie wsparcie decyzyjne
                    if mean_proba>self.training_support_level:
                        break
                    clf.partial_fit(X,y) 
 
                except: # Wysypie się na 1. chunku
                    clf.partial_fit(X,y,classes)
                    
        return self

    def predict(self, X):
        
        # Check certainty
        proba = self.clfs[self.curr_clf_id].predict_proba(X)
        mean_support = np.mean(np.max(proba, axis=1)) # średnie wsparcie decyzyjne
        
        _curr_clf_id = np.argwhere(self.thresholds>mean_support).flatten()[-1]
        if _curr_clf_id != self.curr_clf_id:
            # Check switch
            if self.switch_count==self.switch_when:
                self.switch_count = 0
                # Up or down
                if _curr_clf_id>self.curr_clf_id:
                    self.curr_clf_id+=1
                else:
                    self.curr_clf_id-=1
            else:
                self.switch_count+=1
                
        return np.argmax(proba, axis=1)


class CDoS_T:
    def __init__(self, clfs, thresholds, max_training_epochs, training_support_level, switch_when=10):
        self.clfs = clfs # lista klasyfikatorów
        self.thresholds = thresholds # lista progów (tej samej długości co lista klasyfiaktorów) i najwyższy próg musi nie być przekraczalny bo się wysypie
        self.switch_when = switch_when # ile pod rząd musi przekroczyć próg
        
        self.training_support_level = training_support_level # do tej wartości wsparcia uczone są wszystkie klasyfikatory
        self.max_training_epochs = max_training_epochs # ale jest limit
        
        self.curr_clf_id = 0 # zaczynamy od najgłupszego
        self.switch_count = 0 # licznik
        
        self._firstchunk = True
            
        
    def partial_fit(self, X, y):
        dataset = TensorDataset(torch.Tensor(X),torch.Tensor(y))
        dataloader = DataLoader(dataset, batch_size=64)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for clf_id, clf in enumerate(self.clfs):
            optimizer = torch.optim.SGD(clf.parameters(), lr=1e-2)

            for e in range(self.max_training_epochs):
                if self._firstchunk:
                    clf.train(dataloader, loss_fn, optimizer)
                    self._firstchunk = False
                else:
                    X = X.to(torch.float)
                    proba = nn.Softmax(dim=1)(clf(X))
                    max_proba = torch.max(proba, dim=1)[0] 
                    mean_proba = torch.mean(max_proba).detach().numpy() # średnie wsparcie decyzyjne
                    # print(clf_id, mean_proba)
                    # exit()
                    if mean_proba>self.training_support_level:
                        break
                    
                    # train
                    clf.train(dataloader, loss_fn, optimizer)
 
                    
                    
        return self

    def predict(self, X):
        
        X = X.to(torch.float)
        
        # Check certainty
        proba = nn.Softmax(dim=1)(self.clfs[self.curr_clf_id](X))
        max_proba = torch.max(proba, dim=1)[0] 
        mean_support = torch.mean(max_proba).detach().numpy() # średnie wsparcie decyzyjne
        
        _curr_clf_id = np.argwhere(self.thresholds>=mean_support).flatten()[-1]
        if _curr_clf_id != self.curr_clf_id:
            # Check switch
            if self.switch_count==self.switch_when:
                self.switch_count = 0
                # Up or down
                if _curr_clf_id>self.curr_clf_id:
                    self.curr_clf_id+=1
                else:
                    self.curr_clf_id-=1
            else:
                self.switch_count+=1
                
        return torch.argmax(proba, dim=1)
    

