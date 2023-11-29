import numpy as np
import matplotlib.pyplot as plt

res = np.load('nas_mnist.npy') # 15 , 10, 3
res = res[1:,:5]
print(res.shape)

res_m = np.nanmean(res, axis=1)
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

selected = [
    'CNN1_5',
    'CNN1_10',
    'CNN2_10_15',
    'CNN1_20',
    'CNN2_25_40'
]
markers= ['o', 'o', 'o', 'X', 'X', 'X', 'X', 'D', 'D', 'D', 'D', 'D', 's', 's']

fig, ax = plt.subplots(1,1, figsize=(6,6))
plt.suptitle('MNIST')

cmap = plt.cm.turbo(np.linspace(0,1,len(labels)))
cmap_mono = plt.cm.bone(np.linspace(0,0.8,len(labels)))

for i, l in enumerate(labels):
    c = cmap[i]
    c_m = cmap_mono[i]
    ax.scatter(res_m[i,0], res_m[i,1], label=l, color=c if l in selected else c_m, marker=markers[i], s=100 if l in selected else 50)
ax.set_xlabel('accuracy')
ax.set_ylabel('inference time')

ax.grid(ls=':')

plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/mnist/nas_mnist.eps')
plt.savefig('fig/mnist/nas_mnist.png')