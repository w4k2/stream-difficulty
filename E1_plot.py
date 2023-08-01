import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
accs = np.load('results/e1_accs_m.npy')
times = np.load('results/e1_times_m.npy')
sel = np.load('results/e1_selected_m.npy')

print(accs.shape)
print(times.shape)
print(sel.shape)

accs_m = np.mean(accs, axis=0) 
times_m = np.mean(times, axis=0)
sel_m = np.mean(sel, axis=0) 



# Accuracy
s = 3
cols = plt.cm.Blues(np.linspace(0.3,0.9,5))

for c_id, c in enumerate(chunk_size):
    fig, ax = plt.subplots(4,3, figsize=(18,10), sharex=True, sharey=True)
    plt.suptitle('MNIST | chunk size: %i' % c)

    for n_c_id, n_c in enumerate(n_cycles):
        for mode_id, mode in enumerate(modes):
            
            ax[n_c_id, mode_id].set_title('cycles:%i mode:%s' % (n_c, mode))
            ax[n_c_id, mode_id].grid(ls=':')
            for method_id in range(6):
                temp = accs_m[c_id, n_c_id, mode_id, :, method_id]

                if method_id==5:
                    temp = accs_m[c_id, n_c_id, mode_id, :, -1]
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c='r', alpha=.7)
                else:
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c=cols[method_id], alpha=0.5)
    
    plt.tight_layout()            
    plt.savefig('fig/m_acc_cs%i.png' % c)

# Time

for c_id, c in enumerate(chunk_size):
    fig, ax = plt.subplots(4,3, figsize=(18,10), sharex=True, sharey=True)
    plt.suptitle('MNIST | chunk size: %i' % c)

    for n_c_id, n_c in enumerate(n_cycles):
        for mode_id, mode in enumerate(modes):
            
            ax[n_c_id, mode_id].set_title('cycles:%i mode:%s' % (n_c, mode))
            ax[n_c_id, mode_id].grid(ls=':')
            for method_id in range(6):
                temp = times_m[c_id, n_c_id, mode_id, :, method_id]

                if method_id==5:
                    temp = times_m[c_id, n_c_id, mode_id, :, -1]
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c='r', alpha=.7)
                else:
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c=cols[method_id], alpha=0.5)
    
    plt.tight_layout()                          
    plt.savefig('fig/m_time_cs%i.png' % c)
    
    
# Selection

for c_id, c in enumerate(chunk_size):
    fig, ax = plt.subplots(4,3, figsize=(18,10), sharex=True, sharey=True)
    plt.suptitle('MNIST | chunk size: %i' % c)
    
    for n_c_id, n_c in enumerate(n_cycles):
        for mode_id, mode in enumerate(modes):
            
            ax[n_c_id, mode_id].set_title('chunks:%i mode:%s' % (n_c, mode))
            ax[n_c_id, mode_id].grid(ls=':')

            temp = sel_m[c_id, n_c_id, mode_id]
            ax[n_c_id, mode_id].scatter(np.arange(n_chunks), temp, c='cornflowerblue', alpha=0.5)

    plt.tight_layout()            
    plt.savefig('fig/m_sel_cs%i.png' % c)


### ### ### ### ### ### ### ###
# Analyze SVHN
accs = np.load('results/e1_accs.npy')
times = np.load('results/e1_times.npy')
sel = np.load('results/e1_selected.npy')

print(accs.shape)
print(times.shape)
print(sel.shape)

accs_m = np.mean(accs, axis=0) 
times_m = np.mean(times, axis=0)
sel_m = np.mean(sel, axis=0) 


# Accuracy

for c_id, c in enumerate(chunk_size):
    fig, ax = plt.subplots(4,3, figsize=(18,10), sharex=True, sharey=True)
    plt.suptitle('SVHN | chunk size: %i' % c)

    for n_c_id, n_c in enumerate(n_cycles):
        for mode_id, mode in enumerate(modes):
            
            ax[n_c_id, mode_id].set_title('cycles:%i mode:%s' % (n_c, mode))
            ax[n_c_id, mode_id].grid(ls=':')
            for method_id in range(7):
                temp = accs_m[c_id, n_c_id, mode_id, :, method_id]

                if method_id==6:
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c='r', alpha=.7)
                else:
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c=cols[method_id], alpha=0.5)
    
    plt.tight_layout()                             
    plt.savefig('fig/acc_cs%i.png' % c)

# Time

for c_id, c in enumerate(chunk_size):
    fig, ax = plt.subplots(4,3, figsize=(18,10), sharex=True, sharey=True)
    plt.suptitle('SVHN | chunk size: %i' % c)

    for n_c_id, n_c in enumerate(n_cycles):
        for mode_id, mode in enumerate(modes):
            
            ax[n_c_id, mode_id].set_title('cycles:%i mode:%s' % (n_c, mode))
            ax[n_c_id, mode_id].grid(ls=':')
            for method_id in range(7):
                temp = times_m[c_id, n_c_id, mode_id, :, method_id]

                if method_id==6:
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c='r', alpha=.7)
                else:
                    ax[n_c_id, mode_id].plot(gaussian_filter1d(temp, s), c=cols[method_id], alpha=0.5)
    
    plt.tight_layout()                               
    plt.savefig('fig/time_cs%i.png' % c)
    
    
# Selection

for c_id, c in enumerate(chunk_size):
    fig, ax = plt.subplots(4,3, figsize=(18,10), sharex=True, sharey=True)
    plt.suptitle('SVHN | chunk size: %i' % c)
    
    for n_c_id, n_c in enumerate(n_cycles):
        for mode_id, mode in enumerate(modes):
            
            ax[n_c_id, mode_id].set_title('chunks:%i mode:%s' % (n_c, mode))
            ax[n_c_id, mode_id].grid(ls=':')

            temp = sel_m[c_id, n_c_id, mode_id]
            ax[n_c_id, mode_id].scatter(np.arange(n_chunks), temp, c='cornflowerblue', alpha=0.5)
    
    plt.tight_layout()            
    plt.savefig('fig/sel_cs%i.png' % c)
