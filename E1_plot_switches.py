import numpy as np
import matplotlib.pyplot as plt

chunk_size = [50, 150, 300, 500]
n_cycles = [3, 5, 10, 25]

#MNIST
sel = np.load('results/e1_selected_m.npy')
print(sel.shape)
# (repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))

counts = np.zeros((10,4,4,3))

for rep in range(10):
    for ch_s in range(4):
        for n_c in range(4):
            for m in range(3):
                for c in range(1,1000):
                    if sel[rep,ch_s,n_c,m,c] != sel[rep,ch_s,n_c,m,c-1]:
                        counts[rep,ch_s,n_c,m] +=1

m_counts = np.mean(counts, axis=0)

fig, ax = plt.subplots(1,3,figsize=(12,4))
plt.suptitle('MNIST | #switches')
for m_id, m in enumerate(['instant', 'linear', 'normal']):
    aa = np.flipud(m_counts[:,:,m_id])
    ax[m_id].imshow(aa, cmap='coolwarm')
    
    ax[m_id].set_xticks(np.arange(4), chunk_size)
    ax[m_id].set_yticks(np.arange(4), n_cycles)
    
    ax[m_id].set_xlabel('chunk size')
    ax[m_id].set_ylabel('cycles')
    
    for _a, __a in enumerate(chunk_size):
        for _b, __b in enumerate(n_cycles):
            ax[m_id].text(_b, _a, "%.1f" % (
                aa[_a, _b]
                ) , va='center', ha='center', c='white', fontsize=11)
        
plt.tight_layout()
plt.savefig('fig/counts_m.png')
    

#SVHN
sel = np.load('results/e1_selected.npy')
print(sel.shape)
# (repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))

counts = np.zeros((5,4,4,3))

for rep in range(5):
    for ch_s in range(4):
        for n_c in range(4):
            for m in range(3):
                for c in range(1,1000):
                    if sel[rep,ch_s,n_c,m,c] != sel[rep,ch_s,n_c,m,c-1]:
                        counts[rep,ch_s,n_c,m] +=1

m_counts = np.mean(counts, axis=0)

fig, ax = plt.subplots(1,3,figsize=(12,4))
plt.suptitle('SVHN | #switches')
for m_id, m in enumerate(['instant', 'linear', 'normal']):
    aa = np.flipud(m_counts[:,:,m_id])
    ax[m_id].imshow(aa, cmap='coolwarm')
    
    ax[m_id].set_xticks(np.arange(4), chunk_size)
    ax[m_id].set_yticks(np.arange(4), n_cycles)
    
    ax[m_id].set_xlabel('chunk size')
    ax[m_id].set_ylabel('cycles')
    
    for _a, __a in enumerate(chunk_size):
        for _b, __b in enumerate(n_cycles):
            ax[m_id].text(_b, _a, "%.1f" % (
                aa[_a, _b]
                ) , va='center', ha='center', c='white', fontsize=11)
        
plt.tight_layout()
plt.savefig('fig/counts.png')
    