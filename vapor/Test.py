from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
import generator.concepts as concepts
import torchvision
from torchvision.transforms import Compose, ToTensor
from generator.ConditionalEvidenceStream import ConditionalEvidenceStream
from generator.utils import make_condition_map, mix_to_factor
from methods.Method import CDoS
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Prepare stream 
n_chunks = 1000

data = torchvision.datasets.MNIST('./files/', 
                                  train=True, 
                                  download=True, 
                                  transform = Compose([ToTensor()]))

X = data.data.reshape(60000,-1)
y = data.targets.numpy()

X = PCA(n_components=0.8).fit_transform(X)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

factor = mix_to_factor(X)
condition_map = make_condition_map(n_cycles=3, 
                                   n_concepts=500, 
                                   factor=factor, 
                                   factor_range=(0.1,0.9))

cp = concepts.concept_proba(n_concepts=500,
                            n_chunks=n_chunks,
                            normalize=True,
                            sigma=1)

stream = ConditionalEvidenceStream(X, y,
                                   condition_map.T,
                                   cp,
                                   chunk_size=200,
                                   fragile=False)


# Prepare parameters and the method

rs = 3423
clfs = [
    MLPClassifier(random_state=rs, hidden_layer_sizes=(10)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(20)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(35)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(40)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(50)),
]
thresholds = [0.9, 0.85, 0.8, 0.75, 0.7]

ddos = CDoS(clfs=clfs,
           thresholds=thresholds,
           max_training_epochs=250,
           training_support_level=0.8)

# Process
acc = np.full((len(clfs)+1, n_chunks), np.nan)

for i in tqdm(range(n_chunks)):
    X, y = stream.get_chunk()
    
    if i<20:
        ddos.partial_fit(X, y, np.arange(10))
        
    else:
        pred = ddos.predict(X)
        acc[-1, i] = accuracy_score(y, pred)
        for c_id, c in enumerate(ddos.clfs):
            acc[c_id, i] = accuracy_score(y, c.predict(X))
            

cols = plt.cm.coolwarm(np.linspace(0,1,6))

s=3
fig, ax = plt.subplots(1,1,figsize=(12,8))
for a_id, a in enumerate(acc):
    ax.plot(gaussian_filter1d(a,s), color=cols[a_id], alpha = 0.5 if a_id<5 else 1 , label='DDos' if a_id==5 else 'clf %i' % a_id)

ax.grid(ls=':')
ax.legend(ncols=2, frameon=False)

plt.tight_layout()
plt.savefig('foo.png')