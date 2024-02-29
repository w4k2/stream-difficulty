import numpy as np
import matplotlib.pyplot as plt

chunk_size = [50, 150, 300, 500]
n_cycles = [3, 5, 10, 25]
modes = ['instant', 'linear', 'normal']

#SVHN
sel = np.load('results/e1_selected_3.npy')
print(sel.shape)
# (repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))

counts = np.zeros((5,4,4,3,5,5))

for rep in range(5):
    for ch_s in range(4):
        for n_c in range(4):
            for m in range(3):
                for c in range(1,1000):
                    if sel[rep,ch_s,n_c,m,c] != sel[rep,ch_s,n_c,m,c-1]:
                        _from = int(sel[rep,ch_s,n_c,m,c-1])
                        _to = int(sel[rep,ch_s,n_c,m,c])
                        counts[rep,ch_s,n_c,m,_from,_to] +=1

m_counts = np.mean(counts, axis=0)
_max = np.max(m_counts)

for mode_id, mode in enumerate(modes):
    fig, ax = plt.subplots(4,4, figsize=(10,10), sharex=True, sharey=True)
    plt.suptitle('SVHN | mode: %s' % mode)
    
    for n_c_id, n_c in enumerate(n_cycles):
        for c_id, c in enumerate(chunk_size):
                        
            if n_c_id==0:
                ax[n_c_id, c_id].set_title('size:%i' % c)
                ax[-1, c_id].set_xlabel('selected architecture', fontsize=12)

            if c_id==0:
                ax[n_c_id, c_id].set_ylabel('cycles:%i \n selected architecture' % n_c, fontsize=12)
            ax[n_c_id, c_id].grid(ls=':')

            aa = m_counts[c_id, n_c_id, mode_id]

            ax[n_c_id, c_id].imshow(aa, cmap='Reds',vmin=0,vmax=_max)
            
            for _a in range(5):
                for _b in range(5):
                    if aa[_a, _b] !=0:
                        ax[n_c_id, c_id].text(_b, _a, "%i" % (
                            np.round(aa[_a, _b])
                            ) , va='center', ha='center', c='white' if aa[_a, _b]>_max/2 else 'black', fontsize=11)

    plt.tight_layout()            
    plt.savefig('fig/svhn/switches_%s.png' % mode)
    plt.savefig('fig/svhn/switches_%s.eps' % mode)
    