import numpy as np
import matplotlib.pyplot as plt

# Setup
n_chunks = 1000
chunk_size = [50, 150, 300, 500]
n_cycles = [3, 5, 10, 25]
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

# Analyze MNIST
sel = np.load('results/e1_selected_m.npy')
print(sel.shape)

# Selection
for mode_id, mode in enumerate(modes):
    fig, ax = plt.subplots(4,4, figsize=(10,10), sharex=True, sharey=True)
    plt.suptitle('MNIST | mode: %s' % mode)
    
    for n_c_id, n_c in enumerate(n_cycles):
        for c_id, c in enumerate(chunk_size):
                        
            if n_c_id==0:
                ax[n_c_id, c_id].set_title('size:%i' % c)
                ax[-1, c_id].set_xlabel('selected architecture', fontsize=12)

            if c_id==0:
                ax[n_c_id, c_id].set_ylabel('cycles:%i \n chunks processed' % n_c, fontsize=12)
            ax[n_c_id, c_id].grid(ls=':')
            

            temp = sel[:,c_id, n_c_id, mode_id]
            unqs, cnts = np.unique(temp, return_counts=True)
            
            ax[n_c_id, c_id].bar(unqs, cnts/10, color='cornflowerblue')

    plt.tight_layout()            
    plt.savefig('fig/mnist/m_hist_%s.png' % mode)
    plt.savefig('fig/mnist/m_hist_%s.eps' % mode)
