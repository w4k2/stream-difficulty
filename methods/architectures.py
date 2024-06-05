import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, architecure, n_classes=10):
        super(CNN, self).__init__()
        self.architecure = architecure
        self.n_classes = n_classes
        
    def forward(self, X):
        representation = self.architecure(X)              
        return representation
    
    def custom_train(self, dataloader, loss_fn, optimizer):        
        for i, (X, y) in enumerate(dataloader):
            y = y.to(torch.int64)
            onehot_y = torch.nn.functional.one_hot(y, self.n_classes)

            X = X.to(torch.float)
            pred = self(X)
            loss = loss_fn(pred, onehot_y.to(torch.float))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def FC1_Network(x_input_size=28, img_depth=1, n_classes=10):   
    size = int(((x_input_size/2)/2)**2)*img_depth
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(size, n_classes),
        )

def FC2_Network(x_input_size=28, img_depth=1, n_classes=10):
    size = int(((x_input_size/2)/2)**2)*img_depth
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(size, 1000),
        nn.ReLU(),
        nn.Linear(1000, n_classes),
        )
    
def FC3_Network(x_input_size=28, img_depth=1, n_classes=10):         
    size = int(((x_input_size/2)/2)**2)*img_depth
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(size, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100),
        nn.ReLU(),
        nn.Linear(100, n_classes),
        )

def FC4_Network(x_input_size=28, img_depth=1, n_classes=10):         
    size = int(((x_input_size/2)/2)**2)*img_depth
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(size, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, n_classes),
        )

def CNN1_5_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = ((int((x_input_size-4)/2))**2)*5
    return nn.Sequential(
        nn.Conv2d(img_depth, 5, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN1_10_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = ((int((x_input_size-4)/2))**2)*10
    return nn.Sequential(
        nn.Conv2d(img_depth, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN1_15_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = ((int((x_input_size-4)/2))**2)*15
    return nn.Sequential(
        nn.Conv2d(img_depth, 15, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )

def CNN1_20_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = ((int((x_input_size-4)/2))**2)*20
    return nn.Sequential(
        nn.Conv2d(img_depth, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )

def CNN2_5_10_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-4)/2)
    out_size = (out_size**2)*10
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 5, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(5, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN2_10_15_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-4)/2)
    out_size = (out_size**2)*15
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(10, 15, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN2_15_20_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-4)/2)
    out_size = (out_size**2)*20
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 15, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(15, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )

def CNN2_20_30_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-4)/2)
    out_size = (out_size**2)*30
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(20, 30, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN2_25_40_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-4)/2)
    out_size = (out_size**2)*40
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 25, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(25, 40, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN3_5_10_20_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-2)/2)
    out_size = int((out_size-2)/2)
    out_size = (out_size**2)*20
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 5, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(5, 10, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(10, 20, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )
    
def CNN3_10_20_30_Network(x_input_size=28, img_depth=1, n_classes=10):
    out_size = int((x_input_size-4)/2)
    out_size = int((out_size-2)/2)
    out_size = int((out_size-2)/2)
    out_size = (out_size**2)*30
    
    return nn.Sequential(
        nn.Conv2d(img_depth, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(10, 20, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(), 
        nn.Conv2d(20, 30, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(out_size, n_classes),
    )