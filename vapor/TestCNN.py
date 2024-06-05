from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
import generator.concepts as concepts
import torchvision
from torchvision.transforms import Compose, ToTensor
from generator.ConditionalEvidenceStream import ConditionalEvidenceStream
from generator.utils import make_condition_map, mix_to_factor
from methods.Method import CDoS_T
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from methods.architectures import CNN, CNN1_10_Network, CNN1_5_Network, CNN2_10_20_Network, CNN2_5_10_Network, CNN3_5_10_20_Network, FC_Network
import torch
import torch.nn as nn

# Prepare stream 
n_chunks = 1000
chunk_size = 200

stream_data = torchvision.datasets.SVHN('./files/', 
                                  split='train', 
                                  download=True, 
                                  transform = Compose([ToTensor()]))

train_data = torchvision.datasets.SVHN('./files/', 
                                  split='test', 
                                  download=True, 
                                  transform = Compose([ToTensor()]))

X = torch.tensor(stream_data.data)/255
y = stream_data.labels

X_pca = PCA(n_components=0.8).fit_transform(X.reshape(73257,-1))
X_pca -= np.mean(X_pca, axis=0)
X_pca /= np.std(X_pca, axis=0)
print('PCA done')

factor = mix_to_factor(X_pca)
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
                                   chunk_size=chunk_size,
                                   fragile=False)

print('Stream done')
# Prepare parameters and the method

rs = 3423
clfs = [
    CNN(architecure=FC_Network(img_depth=3, x_input_size=32)),
    CNN(architecure=CNN1_5_Network(img_depth=3, x_input_size=32)),
    CNN(architecure=CNN1_10_Network(img_depth=3, x_input_size=32)),
    CNN(architecure=CNN2_5_10_Network(img_depth=3, x_input_size=32)),
    CNN(architecure=CNN2_10_20_Network(img_depth=3, x_input_size=32)),
    CNN(architecure=CNN3_5_10_20_Network(img_depth=3, x_input_size=32)),  
]
thresholds = [1., 0.95, 0.9, 0.85, 0.8, 0.75]

ddos = CDoS_T(clfs=clfs,
           thresholds=thresholds,
           max_training_epochs=250,
           training_support_level=0.8)


# Train on stationary

train_X = torch.tensor(train_data.data)/255
train_y = train_data.labels
ddos.partial_fit(train_X, train_y)


# Process
acc = np.full((len(clfs)+1, n_chunks), np.nan)

for i in tqdm(range(n_chunks)):
    X, y = stream.get_chunk()
    print(X.shape)

    with torch.no_grad():
        pred = ddos.predict(X)
        # print(pred)
        # exit()
        acc[-1, i] = accuracy_score(y, pred)
        for c_id, c in enumerate(ddos.clfs):
            # Check certainty
            X = X.to(torch.float)
            proba = nn.Softmax(dim=1)(c(X))
            pred = torch.argmax(proba,1)            
            acc[c_id, i] = accuracy_score(y, pred)
            

cols = plt.cm.jet(np.linspace(0.2,1,7))

s=3
fig, ax = plt.subplots(1,1,figsize=(12,8))
for a_id, a in enumerate(acc):
    ax.plot(gaussian_filter1d(a,s), color=cols[a_id], alpha = 0.5 if a_id<6 else 1 , label='CDos' if a_id==6 else 'clf %i' % a_id)

ax.grid(ls=':')
ax.legend(ncols=2, frameon=False)

plt.tight_layout()
plt.savefig('foo.png')