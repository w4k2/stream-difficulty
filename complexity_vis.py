from problexity import ComplexityCalculator
from problexity.classification import *
from sklearn.decomposition import PCA
import numpy as np
from generator.ConditionalEvidenceStream import ConditionalEvidenceStream
from generator.concepts import concept_proba
from generator.utils import make_condition_map, mix_to_factor
import torch
import torchvision
import matplotlib.pyplot as plt


np.random.seed(2137)
torch.manual_seed(2137)
torch.set_num_threads(1)

# Prepare stream data
stream_data = torchvision.datasets.MNIST('./files/', 
                                #   split='train',
                                  train=True,
                                  download=True)

X = torch.tensor(stream_data.data)/255
# y = stream_data.labels
y = stream_data.targets

mask = np.zeros_like(y).astype(bool)
mask[y==2]=1
mask[y==7]=1

X=X[mask]
y=y[mask]

y[y==2]=0
y[y==7]=1


X_pca = PCA(n_components=0.9).fit_transform(X.reshape(X.shape[0],-1))
X_pca -= np.mean(X_pca, axis=0)
X_pca /= np.std(X_pca, axis=0)

factor = mix_to_factor(X_pca)

# setup
n_chunks = 200
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

cc = ComplexityCalculator(metrics=[f1], 
                          colors=['#F76915'], 
                          ranges={'FB': 1}, weights=[1],
                          multiclass_strategy='ovo')

# One stream
condition_map = make_condition_map(n_cycles=1,
                                n_concepts=500,
                                factor=factor,
                                factor_range=(0.1,0.9))

cp = concept_proba(n_concepts=500,
                            n_chunks=n_chunks,
                            normalize=True,
                            **modes['linear'])
                            # **modes['instant'])

stream = ConditionalEvidenceStream(X, y,
                                condition_map.T,
                                cp,
                                chunk_size=50,
                                fragile=False,
                                random_state=1223)

images=[]
complexities = []

ile = 5

for c in range(200):
    _X, y = stream.get_chunk()
    _X = _X.numpy()
    y = y.numpy()
    
    print(_X.shape)
    images.append(_X[:ile])
    
    X = PCA(n_components=0.9).fit_transform(_X.reshape(50, -1), y)
    cc.fit(X, y)
    
    complexities.append(cc.report()['complexities']['f1'])
    print(complexities, len(complexities))
    
    # print(complexities)
    # exit()
        
images = np.array(images)
print(images.shape)

# images = images.swapaxes(2,3).swapaxes(3,4)
images = images.reshape(200,-1,1,28,28)
images = images.swapaxes(2,3).swapaxes(3,4)

minarg = np.argmin(complexities)
maxarg = np.argmax(complexities)

print(complexities[minarg])
print(complexities[maxarg])

fig, ax = plt.subplots(ile, 3, figsize=(8,10), sharex=True, sharey=True)

for i in range(ile):
    ax[i,0].imshow(images[50,i])
    ax[i,1].imshow(images[100,i])
    ax[i,2].imshow(images[150,i])
    
ax[0,0].set_title('C:%0.3f | %i' % (complexities[50], 50))
ax[0,1].set_title('C:%0.3f | %i' % (complexities[100], 100))
ax[0,2].set_title('C:%0.3f | %i' % (complexities[150], 150))
   
plt.tight_layout()
plt.savefig('foo2.png')
        