import numpy as np
import json
import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

complexities = json.load(io.open('results/comp.json'))
print(len(complexities))

measures = ['f1', 'n1']

fig, ax = plt.subplots(1,1,figsize=(6,3))
cols = plt.cm.turbo(np.linspace(0.1,0.9,len(measures)))
s = 2

for m_id, m in enumerate(measures):
    ee = []
    for c in range(len(complexities)):
        ee.append(complexities[str(c)][m])
    
    print(ee)
    
    ax.plot(gaussian_filter1d(ee, s), c=cols[m_id], label=['F1', 'N1'][m_id])
    
ax.grid(ls=':')
ax.legend(frameon=False)
ax.set_title('SVHN | chunk size: 250 | cycles: 5')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('chunk', fontsize=12)
ax.set_ylabel('complexity', fontsize=12)

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/svhn/complexity.png')
plt.savefig('fig/svhn/complexity.eps')