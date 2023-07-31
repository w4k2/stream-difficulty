import numpy as np
import matplotlib.pyplot as plt

res = np.load('nas.npy') # 15 , 10, 3
res = res[1:,:5]
print(res.shape)

res_m = np.mean(res, axis=1)
print(res_m.shape)

labels = [
    'FC2',
    'FC3', 
    'FC4',
    'CNN1_5',
    'CNN1_10',
    'CNN1_15',
    'CNN1_20',
    'CNN2_5_10',
    'CNN2_10_15',
    'CNN2_15_20',
    'CNN2_20_30',
    'CNN2_25_40',
    'CNN3_5_10_20',
    'CNN3_10_20_30'
]

markers= ['o', 'o', 'o', 'X', 'X', 'X', 'X', 'D', 'D', 'D', 'D', 'D', 's', 's']

fig, ax = plt.subplots(1,1, figsize=(7,7))
plt.suptitle('SVHN')

cmap = plt.cm.turbo(np.linspace(0,1,len(labels)))

for i, l in enumerate(labels):
    c = cmap[i]
    ax.scatter(res_m[i,0], res_m[i,1], label=l, color=c, marker=markers[i])
ax.set_xlabel('accuracy')
ax.set_ylabel('inference time')

ax.grid(ls=':')

plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('foo.png')