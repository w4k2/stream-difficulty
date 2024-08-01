import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Setup
n_chunks = 1000
chunk_size = np.array([50, 150, 300, 500])
chunk_size_mask = np.array([1,0,1,0]).astype(bool)
n_cycles = [3, 5, 10, 25]
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

# Analyze MNIST
accs = np.load('results/e1_accs_m.npy')
times = np.load('results/e1_times_m.npy')
sel = np.load('results/e1_selected_m.npy')

print(accs.shape)
print(times.shape)
print(sel.shape)

accs_m = np.mean(accs, axis=0)[chunk_size_mask]
times_m = np.mean(times, axis=0)[chunk_size_mask]
# sel_m = np.mean(sel, axis=0)[chunk_size_mask]

# print(accs_m.shape)
# exit()


# Accuracy
s = 3
cols = plt.cm.Blues(np.linspace(0.3,0.9,5))
for mode_id, mode in enumerate(modes):
    fig, ax = plt.subplots(4,2, figsize=(8,7), sharex=True, sharey=True)
    plt.suptitle('MNIST | mode: %s' % mode, fontsize=15)

    for n_c_id, n_c in enumerate(n_cycles):
        for c_id, c in enumerate(chunk_size[chunk_size_mask]):

            if n_c_id==0:
                ax[n_c_id, c_id].set_title('size:%i' % c, fontsize=13)
                ax[-1, c_id].set_xlabel('chunk', fontsize=13)

            if c_id==0:
                ax[n_c_id, c_id].set_ylabel('cycles:%i \naccuracy' % n_c, fontsize=13)
            
            ax[n_c_id, c_id].grid(ls=':')
            ax[n_c_id, c_id].spines['top'].set_visible(False)
            ax[n_c_id, c_id].spines['right'].set_visible(False)

            for method_id in range(6):
                temp = accs_m[c_id, n_c_id, mode_id, :, method_id]

                if method_id==5:
                    temp = accs_m[c_id, n_c_id, mode_id, :, -1]
                    ax[n_c_id, c_id].plot(gaussian_filter1d(temp, s), c='r', alpha=.7)
                else:
                    ax[n_c_id, c_id].plot(gaussian_filter1d(temp, s), c=cols[method_id], alpha=0.5)
    
    plt.tight_layout()            
    plt.savefig('fig/mnist/m_acc_%s.png' % mode)
    plt.savefig('fig/mnist/m_acc_%s.eps' % mode)
    plt.savefig('foo.png')

# Time

for mode_id, mode in enumerate(modes):
    fig, ax = plt.subplots(4,2, figsize=(8,7), sharex=True, sharey=False)
    plt.suptitle('MNIST | mode: %s' % mode)

    for n_c_id, n_c in enumerate(n_cycles):
        for c_id, c in enumerate(chunk_size[chunk_size_mask]):
            
            if n_c_id==0:
                ax[n_c_id, c_id].set_title('size:%i' % c, fontsize=13)
                ax[-1, c_id].set_xlabel('chunk', fontsize=13)

            if c_id==0:
                ax[n_c_id, c_id].set_ylabel('cycles:%i \ntime' % n_c, fontsize=13)
            
            ax[n_c_id, c_id].grid(ls=':')
            ax[n_c_id, c_id].spines['top'].set_visible(False)
            ax[n_c_id, c_id].spines['right'].set_visible(False)

            for method_id in range(6):
                temp = times_m[c_id, n_c_id, mode_id, :, method_id]

                if method_id==5:
                    temp = times_m[c_id, n_c_id, mode_id, :, -1]
                    ax[n_c_id, c_id].plot(gaussian_filter1d(temp, s), c='r', alpha=.7)
                else:
                    ax[n_c_id, c_id].plot(gaussian_filter1d(temp, s), c=cols[method_id], alpha=0.5)
    
    plt.tight_layout()                          
    plt.savefig('fig/mnist/m_time_%s.png' % mode)
    plt.savefig('fig/mnist/m_time_%s.eps' % mode)
    plt.savefig('foo.png')
    
    
# Selection

for mode_id, mode in enumerate(modes):
    fig, ax = plt.subplots(4,2, figsize=(8,7), sharex=True, sharey=True)
    plt.suptitle('MNIST | mode: %s' % mode)
    
    for n_c_id, n_c in enumerate(n_cycles):
        for c_id, c in enumerate(chunk_size[chunk_size_mask]):
            
            if n_c_id==0:
                ax[n_c_id, c_id].set_title('size:%i' % c, fontsize=13)
                ax[-1, c_id].set_xlabel('chunk', fontsize=13)

            if c_id==0:
                ax[n_c_id, c_id].set_ylabel('cycles:%i \nclf index' % n_c, fontsize=13)
            ax[n_c_id, c_id].grid(ls=':')
            ax[n_c_id, c_id].spines['top'].set_visible(False)
            ax[n_c_id, c_id].spines['right'].set_visible(False)

            temp = sel[:,chunk_size_mask][:,c_id, n_c_id, mode_id]
            print(temp.shape)
            
            temp_m = gaussian_filter1d(np.mean(temp, axis=0),2)
            temp_std = np.std(temp, axis=0)
            ax[n_c_id, c_id].plot(np.arange(1000), temp_m, c='r', alpha=1)
            ax[n_c_id, c_id].fill_between(np.arange(1000), temp_m-temp_std, temp_m+temp_std, color='r', alpha=0.15, edgecolor='white')
            
            
            ax[n_c_id, c_id].set_ylim(0,4)
            # ax[n_c_id, c_id].scatter(np.arange(n_chunks), temp, c='r', alpha=0.25, s=5)

    plt.tight_layout()            
    plt.savefig('fig/mnist/m_sel_%s.png' % mode)
    plt.savefig('fig/mnist/m_sel_%s.eps' % mode)
    plt.savefig('foo.png')

