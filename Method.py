"""
Difficulty Domain Selector
"""

import numpy as np

class DDoS:
    def __init__(self, clfs, thresholds, max_training_epochs, training_support_level, switch_when=10):
        self.clfs = clfs
        self.thresholds = thresholds
        self.switch_when = switch_when
        
        self.max_training_epochs = max_training_epochs
        self.training_support_level = training_support_level
        
        self.curr_clf_id = 0
        self.switch_count = 0
            
        
    def partial_fit(self, X, y, classes):
        for clf in self.clfs:
            for e in range(self.max_training_epochs):
                try:
                    mean_proba = np.mean(np.max(clf.predict_proba(X), axis=1))
                    if mean_proba>self.training_support_level:
                        break
                    clf.partial_fit(X,y) 
 
                except: # Wysypie się na 1. chunku
                    clf.partial_fit(X,y,classes)
                    
        return self

    def predict(self, X):
        
        # Check certainty
        proba = self.clfs[self.curr_clf_id].predict_proba(X)
        mean_support = np.mean(np.max(proba, axis=1))
        
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
    
