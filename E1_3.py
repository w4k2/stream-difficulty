import torchvision
import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from ConditionalEvidenceStream import ConditionalEvidenceStream
from utils import make_condition_map, mix_to_factor
import concepts
import torch
import torchvision
import time
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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

#Load trained classifiers
clfs=[]
for c_id in range(5):
    clfs.append(torch.load('models2/%i.pt' % c_id))
    
    
# Experimental setup
repeats = 5
n_chunks = 1000
chunk_size = [50, 150, 300, 500]
n_cycles = [3, 5, 10, 25]
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

pbar = tqdm(total=repeats*len(chunk_size)*len(n_cycles)*len(modes))

accs = np.zeros((repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks, len(clfs)))
times = np.zeros((repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks, len(clfs)))
supps = np.zeros((repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks, len(clfs)))

r_states = np.random.choice(100000, repeats, replace=False)

for cs_id, cs in enumerate(chunk_size):
    for r_id, rs in enumerate(r_states):
        
        for n_c_id, nc in enumerate(n_cycles):
            for m_id, mode in enumerate(modes):

                condition_map = make_condition_map(n_cycles=nc,
                                                n_concepts=500,
                                                factor=factor,
                                                factor_range=(0.1,0.9))

                cp = concepts.concept_proba(n_concepts=500,
                                            n_chunks=n_chunks,
                                            normalize=True,
                                            **modes[mode])

                stream = ConditionalEvidenceStream(X, y,
                                                condition_map.T,
                                                cp,
                                                chunk_size=cs,
                                                fragile=False,
                                                random_state=rs)
                # Prepare method
                with torch.no_grad():

                    for chunk_id in range(n_chunks):
                        _X, _y = stream.get_chunk()
                        
                        # Regular clfs
                        for c_id, c in enumerate(clfs):
                            start = time.time()
                            proba = nn.Softmax(dim=1)(c(_X))
                            p = torch.argmax(proba, dim=1)    
                            elapsed = time.time()-start
                            
                            accs[r_id, cs_id, n_c_id, m_id, chunk_id, c_id] = accuracy_score(_y, p)
                            supps[r_id, cs_id, n_c_id, m_id, chunk_id, c_id] = torch.mean(torch.max(proba, dim=1)[0])
                            times[r_id, cs_id, n_c_id, m_id, chunk_id, c_id] = elapsed
                            
                    
                    pbar.update(1)
                    
                    np.save('results/e1_accs_3.npy', accs)
                    np.save('results/e1_times_3.npy', times)
                    np.save('results/e1_supps_3.npy', supps)
                
                
