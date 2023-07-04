import torchvision
from torchvision.transforms import Compose, ToTensor
from sklearn.decomposition import PCA
import numpy as np

root = './files/'
transform = Compose([ToTensor()])

data = torchvision.datasets.MNIST(root, 
                                  train=True, 
                                  download=True, 
                                  transform = transform)

X = data.data.reshape(60000,-1)
y = data.targets.numpy()

pca = PCA(n_components=0.8)
X = pca.fit_transform(X)



#extracted = np.concatenate((X, y[:, None]), axis=1)
#np.save('files/extracted.npy', extracted)

# class MnistStreamGenerator:
#     def __init__(self, chunk_size=100, difficulty_drifts=2):
#         self.chunk_size=chunk_size
#         self.difficulty_drifts = difficulty_drifts
        
        