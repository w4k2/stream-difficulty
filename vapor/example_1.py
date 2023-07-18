import concepts
import numpy as np
import torchvision
from torchvision.transforms import Compose, ToTensor
from sklearn.decomposition import PCA
from ConditionalEvidenceStream import ConditionalEvidenceStream
from utils import make_condition_map, mix_to_factor

np.set_printoptions(precision=3, suppress=True)

root = './files/'
transform = Compose([ToTensor()])

n_chunks = 1410
chunk_size = 666
n_concepts = 520

n_samples = 60000
factor_range = (.1,.9)
n_cycles = 3 # conditional cycles

print('Loading MNIST dataset...')
data = torchvision.datasets.MNIST(root, 
                                  train=True, 
                                  download=True, 
                                  transform = transform)

print('Transforming MNIST dataset...')
X = data.data.reshape(n_samples,-1)
y = data.targets.numpy()

# Mamy prawo użyć PCA, bo MNIST jest już zredukowany do 784 cech
# A tak naprawdę to dlatego, że dla nas to tylko kompresor
print('Applying PCA...')
X = PCA(n_components=0.8).fit_transform(X)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
print(X.shape)

print('Establishing distribution factor (difficulty)...')
factor = mix_to_factor(X)
print(factor.shape)

print('Preparing condition map...')
condition_map = make_condition_map(n_cycles=n_cycles, 
                                   n_concepts=n_concepts, 
                                   factor=factor, 
                                   factor_range=factor_range)
print(condition_map.shape)

print('Generating concept probabilities...')
cp = concepts.concept_proba(n_concepts=n_concepts,
                            n_chunks=n_chunks,
                            mode='normal', 
                            compression=None,#'log', 
                            normalize=True,
                            sigma=1)
print(cp.shape)

print('Initializing stream...')
stream = ConditionalEvidenceStream(X, y,
                                   condition_map.T,
                                   cp,
                                   chunk_size=chunk_size,
                                   fragile=False)

print('Processing evidence...')
while chunk := stream.get_chunk():
    X, y = chunk
    print(stream.chunk_idx, y[:3], X.shape, y.shape)
    