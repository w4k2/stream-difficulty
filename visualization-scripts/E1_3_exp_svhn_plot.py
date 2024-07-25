
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

n_chunks = 1000
chunk_size = np.array([50, 150, 300, 500])
chunk_size_mask = np.array([1,0,1,0]).astype(bool)
n_cycles = [3, 5, 10, 25]
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

accs = np.load('results/e1_accs_3.npy')
times = np.load('results/e1_times_3.npy')
supp = np.load('results/e1_supps_3.npy')

print(accs.shape)
print(times.shape)
print(supp.shape)

s = 2
switch_count = 0
switch_when = 10

thresholds_all = [
    [1., 0.95, 0.87, 0.85, 0.84], # chunk size = 50
    [1., 0.92, 0.86, 0.85, 0.84], # chunk size = 150
    [1., 0.91, 0.86, 0.855, 0.85], # chunk size = 300
    [1., 0.89, 0.86, 0.855,  0.85] # chunk size = 500
]
thresholds_all = np.array(thresholds_all)[chunk_size_mask]

res_selected = np.zeros((5, 2, 4, 3, 1000)) #(repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))

for m_id, mode in enumerate(modes):
    
    acc_fig, acc_ax = plt.subplots(4,2, figsize=(8,7), sharex=True, sharey=True)
    acc_fig.suptitle('SVHN | mode: %s' % mode, fontsize=15)
    
    sel_fig, sel_ax = plt.subplots(4,2, figsize=(8,7), sharex=True, sharey=True)
    sel_fig.suptitle('SVHN | mode: %s' % mode, fontsize=15)

    time_fig, time_ax = plt.subplots(4,2, figsize=(8,7), sharex=True)
    time_fig.suptitle('SVHN | mode: %s' % mode, fontsize=15)

    for cs_id, cs in enumerate(chunk_size[chunk_size_mask]):
            
        for n_c_id, nc in enumerate(n_cycles):
            
            thresholds = thresholds_all[cs_id]
            
            selected_all = []
            _this_acc_all = []
            _this_time_all = []
            
            for r_id in range(5):

                selected = []
                _this_acc = []
                _this_time = []
                
                this_accs = accs[r_id, cs_id, n_c_id, m_id]
                this_times = times[r_id, cs_id, n_c_id, m_id]
                this_supps = supp[r_id, cs_id, n_c_id, m_id]
                
                for c in range(n_chunks):
                    mean_supp = this_supps[c, s]
                    _s = np.argwhere(thresholds>mean_supp).flatten()[-1]

                    if _s != s:
                        # Check switch
                        if switch_count==switch_when:
                            switch_count = 0
                            # Up or down
                            if _s>s:
                                s+=1
                            else:
                                s-=1
                        else:
                            switch_count+=1

                    res_selected[r_id, cs_id, n_c_id, m_id, c] = s
                    selected.append(s)
                    _this_acc.append(this_accs[c, s])
                    _this_time.append(this_times[c, s])

                
                _this_acc_all.append(_this_acc)
                _this_time_all.append(_this_time)
                selected_all.append(selected)
                
                print(np.unique(selected_all, return_counts=True))
            
                                
            # Plot accuracy
            sigma = 3
            cols = plt.cm.Blues(np.linspace(0.3,0.9,5))

            if n_c_id==0:
                acc_ax[n_c_id, cs_id].set_title('size:%i' % cs, fontsize=13)
            if cs_id==0:
                acc_ax[n_c_id, cs_id].set_ylabel('cycles:%i \nacc' % nc, fontsize=13)
            
            acc_ax[n_c_id, cs_id].grid(ls=':')
            
            for clf_id in range(5):
                temp = np.mean(accs[:, cs_id, n_c_id, m_id, :, clf_id], axis=0)
                acc_ax[n_c_id, cs_id].plot(gaussian_filter1d(temp, sigma), c=cols[clf_id], alpha=0.5)
             
            
            temp = np.mean(np.array(_this_acc_all), axis=0)
            acc_ax[n_c_id, cs_id].plot(gaussian_filter1d(temp, sigma), c='r', alpha=0.75)
            acc_ax[n_c_id, cs_id].spines['top'].set_visible(False)
            acc_ax[n_c_id, cs_id].spines['right'].set_visible(False)


            acc_fig.tight_layout()            
            acc_fig.savefig('foo.png')                           
            acc_fig.savefig('fig/svhn/acc_%s.png' % mode)
            acc_fig.savefig('fig/svhn/acc_%s.eps' % mode)

            
            # Plot time
            if n_c_id==0:
                time_ax[n_c_id, cs_id].set_title('size:%i' % cs, fontsize=13)
            if cs_id==0:
                time_ax[n_c_id, cs_id].set_ylabel('cycles:%i \nseconds' % nc, fontsize=13)
            
            time_ax[n_c_id, cs_id].grid(ls=':')
            
            for clf_id in range(5):
                temp = np.mean(times[:, cs_id, n_c_id, m_id, :, clf_id], axis=0)
                time_ax[n_c_id, cs_id].plot(gaussian_filter1d(temp, sigma), c=cols[clf_id], alpha=0.5)
                
            temp = np.mean(np.array(_this_time_all), axis=0)
            time_ax[n_c_id, cs_id].plot(gaussian_filter1d(temp, sigma), c='r', alpha=0.75)
            time_ax[n_c_id, cs_id].spines['top'].set_visible(False)
            time_ax[n_c_id, cs_id].spines['right'].set_visible(False)

            time_fig.tight_layout()            
            time_fig.savefig('foo.png')                           
            time_fig.savefig('fig/svhn/time_%s.png' % mode)
            time_fig.savefig('fig/svhn/time_%s.eps' % mode)
            
            # Plot sel
            if n_c_id==0:
                sel_ax[n_c_id, cs_id].set_title('size:%i' % cs, fontsize=13)
            if cs_id==0:
                sel_ax[n_c_id, cs_id].set_ylabel('cycles:%i \nindex' % nc, fontsize=13)
            
            sel_ax[n_c_id, cs_id].grid(ls=':')
            
            temp = np.mean(np.array(selected_all), axis=0)
            sel_ax[n_c_id, cs_id].scatter(np.arange(len(temp)), temp, c='r', alpha=0.25, s=5)
            sel_ax[n_c_id, cs_id].spines['top'].set_visible(False)
            sel_ax[n_c_id, cs_id].spines['right'].set_visible(False)

            sel_fig.tight_layout()            
            sel_fig.savefig('foo.png')                           
            sel_fig.savefig('fig/svhn/sel_%s.png' % mode)
            sel_fig.savefig('fig/svhn/sel_%s.eps' % mode)
    # exit()
    
np.save('results/e1_selected_3.npy', res_selected)