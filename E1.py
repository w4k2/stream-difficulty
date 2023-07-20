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
from Method import CDoS_T
from tqdm import tqdm

def get_th(chunk_size):
    max_probas=[]
    for c in clfs:
        proba = nn.Softmax(dim=1)(c(train_X))
        max_proba = torch.max(proba, dim=1)[0]
        max_probas.append(max_proba.detach().numpy())
        
    mp = np.array(max_probas).flatten()
    aa = int(len(mp)/chunk_size)
    mp = mp[:aa*chunk_size]
    mp = mp.reshape(aa,chunk_size)
    mp = np.mean(mp, axis=1)

    n_bins=7
    bin_size = int(len(mp)/n_bins)

    th = []
    argsort_mp = np.argsort(mp)

    for b in range(n_bins-1):
        th.append(mp[argsort_mp[b*bin_size]])

    th.reverse()
    th[0]=1.
    
    return th

# Prepare trainig data
train_data = torchvision.datasets.SVHN('./files/', 
                                  split='test', #Tak.
                                  download=True)

train_X = (torch.tensor(train_data.data)/255).to(torch.float)


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
for c_id in range(6):
    clfs.append(torch.load('models/%i.pt' % c_id))
    
    
# Experimental setup
repeats = 10
n_chunks = 1000
chunk_size = [50, 150, 300, 500]
n_cycles = [3, 5, 10, 25]
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

pbar = tqdm(total=repeats*len(chunk_size)*len(n_cycles)*len(modes))

accs = np.zeros((repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks, 7))
times = np.zeros((repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks, 7))
cdos_selected = np.zeros((repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))

r_states = np.random.choice(100000, repeats, replace=False)
print(r_states)

for cs_id, cs in enumerate(chunk_size):
    thresholds = get_th(chunk_size=cs)
    print(thresholds)
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
                cdos = CDoS_T(clfs=clfs, thresholds=thresholds)

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
                            times[r_id, cs_id, n_c_id, m_id, chunk_id, c_id] = elapsed
                            
                        #CDos
                        start = time.time()
                        p = cdos.predict(_X)
                        elapsed = time.time()-start
                        
                        accs[r_id, cs_id, n_c_id, m_id, chunk_id, -1] = accuracy_score(_y, p)
                        times[r_id, cs_id, n_c_id, m_id, chunk_id, -1] = elapsed
                        cdos_selected[r_id, cs_id, n_c_id, m_id, chunk_id] = cdos.curr_clf_id
                        
                    
                    pbar.update(1)
                    print(cs, nc, mode, np.unique(cdos_selected[r_id, cs_id, n_c_id, m_id], return_counts=True))
                    
                    np.save('results/e1_accs.npy', accs)
                    np.save('results/e1_times.npy', times)
                    np.save('results/e1_selected.npy', cdos_selected)
                
                
