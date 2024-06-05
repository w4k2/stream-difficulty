import numpy as np
from scipy.stats import norm

N_CONCEPTS = 128
N_CHUNKS = 1000
CHUNK_SIZE = 250
SIGMA = .125
COMPRESSION_FACTOR = 1.25

CONCEPT_MODES = ['instant', 'linear', 'normal']
COMPRESSION_MODES = [None, 'log']

def instant_concept_proba(n_concepts=N_CONCEPTS, n_chunks=N_CHUNKS, sigma=None):
    proba_mat = np.arange(n_concepts)[None,] == np.linspace(0,n_concepts,n_chunks)[:,None].astype(int)
    return proba_mat

def linear_concept_proba(n_concepts=N_CONCEPTS, n_chunks=N_CHUNKS, sigma=None):
    proba_mat = np.clip(-np.abs((-np.linspace(0,n_concepts-1,n_chunks)[:, None] + np.arange(n_concepts)[None,:]))+1,0,1)
    return proba_mat

def normal_concept_proba(n_concepts=N_CONCEPTS, n_chunks=N_CHUNKS, sigma=SIGMA):
    lsa = np.linspace(-sigma*n_concepts,sigma*n_concepts,n_chunks)
    m = norm.pdf(lsa[None,:] - 2*sigma*(np.arange(n_concepts)-((n_concepts-1)/2))[:,None])
    m = m / np.sum(m, axis=0)
    return m.T

def concept_proba(n_concepts=N_CONCEPTS, n_chunks=N_CHUNKS, 
                  mode=CONCEPT_MODES[0], compression=COMPRESSION_MODES[0],
                  sigma=SIGMA, compression_factor=COMPRESSION_FACTOR, normalize=True):
    if mode not in CONCEPT_MODES:
        raise ValueError('mode must be one of {}'.format(CONCEPT_MODES))
    if compression not in COMPRESSION_MODES:
        raise ValueError('compression must be one of {}'.format(COMPRESSION_MODES))
    
    concept_proba = {'instant': instant_concept_proba,
                     'linear': linear_concept_proba,
                     'normal': normal_concept_proba}[mode](n_concepts=n_concepts, 
                                                           n_chunks=n_chunks,
                                                           sigma=sigma)

    if compression is not None:
        space = np.logspace(0,compression_factor,n_chunks) - 1
        space -= np.min(space)
        space /= np.max(space)
        
        concept_proba = concept_proba[((n_chunks-1)*(space)).astype(int)]
        
    if normalize:
        concept_proba = concept_proba / np.sum(concept_proba, axis=1)[:,None]
            
    return concept_proba

def cp2evidence(cp, chunk_size):
    c = np.arange(cp.shape[1])
    evidence = []
    for ccp in cp:
        try:
            ev = np.random.choice(c, size=chunk_size, p=ccp)
        except:
            ev = np.random.choice(c, size=chunk_size)
        evidence.append(ev)
        
    return np.array(evidence)

def main():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(CONCEPT_MODES),3,figsize=(14,10))
    
    fig.suptitle('Concept probabilities for %i concepts on %i chunks (SIGMA%.2f)' % (N_CONCEPTS, N_CHUNKS, SIGMA))

    for concept_mode in CONCEPT_MODES:
        cp = concept_proba(mode=concept_mode, compression='log', normalize=True)

        cps = np.cumsum(cp, axis=0)
        
        evidence = cp2evidence(cp, CHUNK_SIZE).ravel()

        aa = ax[CONCEPT_MODES.index(concept_mode)]
        
        aa[0].plot(cp, color='black', alpha=1)
        aa[1].plot(cps, color='black', alpha=1)
        aa[2].scatter(np.linspace(0,N_CHUNKS-1,N_CHUNKS*CHUNK_SIZE), evidence, color='black', alpha=.1, s=1)
        
        aa[0].set_title('%s mode concept probability' % (concept_mode))
        aa[1].set_title('%s mode concept accumulation' % (concept_mode))
        aa[2].set_title('%s mode evidence' % (concept_mode))

        aa[0].set_ylim(0,1)
        aa[0].set_yticks(np.linspace(0,1,5))
        
        aa[0].set_ylabel('Probability')
        aa[1].set_ylabel('Accumulated probability')
            
        for aaa in aa:
            aaa.grid(ls=":")
            aaa.set_xlim(0,N_CHUNKS)
            aaa.set_xticks(np.linspace(0,N_CHUNKS,5))
            aaa.spines['right'].set_visible(False)
            aaa.spines['top'].set_visible(False)
            aaa.set_xlabel('Chunk')
            
    plt.tight_layout()
    plt.savefig('bar.png')

if __name__ == '__main__':
    main()
