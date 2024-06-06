import numpy as np
import json
import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

complexities_m = json.load(io.open('results/comp_m.json'))
complexities_s = json.load(io.open('results/comp.json'))
print(len(complexities_m))
print(len(complexities_s))

measures = ["f1", "n1"]

fig, ax = plt.subplots(1,2,figsize=(12,4))
cols = plt.cm.turbo(np.linspace(0.1,0.9,len(measures)))
s = 2

for m_id, m in enumerate(measures):
    ee = []
    for c in range(len(complexities_m)):
        ee.append(complexities_m[str(c)][m])
    
    print(ee)
    
    ax[0].plot(gaussian_filter1d(ee, s), c=cols[m_id], label=['F1', 'N1'][m_id])

for m_id, m in enumerate(measures):
    ee = []
    for c in range(len(complexities_s)):
        ee.append(complexities_s[str(c)][m])
    
    print(ee)
    
    ax[1].plot(gaussian_filter1d(ee, s), c=cols[m_id], label=['F1', 'N1'][m_id])
    

ax[0].set_title('MNIST | chunk size: 250 | cycles: 5')
ax[1].set_title('SVHN | chunk size: 250 | cycles: 5')
ax[0].set_ylabel('complexity', fontsize=12)
ax[0].legend(frameon=False)

for aa in ax:
    aa.grid(ls=':')
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)
    aa.set_xlabel('chunk', fontsize=12)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/complexity.png')
plt.savefig('fig/complexity.eps')
