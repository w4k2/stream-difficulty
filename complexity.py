from problexity import ComplexityCalculator
from problexity.classification import *
from sklearn.decomposition import PCA
import numpy as np
from ConditionalEvidenceStream import ConditionalEvidenceStream
from concepts import concept_proba
from utils import make_condition_map, mix_to_factor
import torch
import torchvision
import json

np.random.seed(1223)
torch.manual_seed(15553)
torch.set_num_threads(1)

# Prepare stream data
stream_data = torchvision.datasets.SVHN('./files/', 
                                  split='train',
                                  download=True)

X = torch.tensor(stream_data.data)/255
y = stream_data.labels

X_pca = PCA(n_components=0.8).fit_transform(X.reshape(X.shape[0],-1))
X_pca -= np.mean(X_pca, axis=0)
X_pca /= np.std(X_pca, axis=0)

factor = mix_to_factor(X_pca)

# setup
n_chunks = 1000
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

cc = ComplexityCalculator(metrics=[f1, n1], 
                          colors=['#F76915', '#EEDE04'], 
                          ranges={'FB': 1, 'NB': 1}, weights=[1,1])

# One stream
condition_map = make_condition_map(n_cycles=5,
                                n_concepts=500,
                                factor=factor,
                                factor_range=(0.1,0.9))

cp = concept_proba(n_concepts=500,
                            n_chunks=n_chunks,
                            normalize=True,
                            **modes['linear'])

stream = ConditionalEvidenceStream(X, y,
                                condition_map.T,
                                cp,
                                chunk_size=250,
                                fragile=False,
                                random_state=1223)

complexities={}

for c in range(1000):
    _X, y = stream.get_chunk()
    _X = _X.reshape(250, -1)
    
    X = PCA(n_components=10).fit_transform(_X, y)
    cc.fit(X, y)
    
    complexities.update({c: cc.report()['complexities']})
    print(len(complexities))
    
    _comp = (json.dumps(complexities, indent=4))

    with open('results/comp.json', 'w') as f:
        f.write(_comp)