import numpy as np
import json
import io
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

complexities = json.load(io.open('results/comp_m.json'))
print(len(complexities))

# measures = ["f1", "f1v", "n1", "n2", "l1"]
measures = ["f1", "n1"]

fig, ax = plt.subplots(1,1,figsize=(10,4))
cols = plt.cm.turbo(np.linspace(0.1,0.9,len(measures)))
s = 2

for m_id, m in enumerate(measures):
    ee = []
    for c in range(len(complexities)):
        ee.append(complexities[str(c)][m])
    
    print(ee)
    
    ax.plot(gaussian_filter1d(ee, s), c=cols[m_id], label=m)
    
ax.grid(ls=':')
ax.legend(frameon=False)
ax.set_title('MNIST | chunk size: 50 | cycles: 5')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('chunk')
ax.set_ylabel('complexity')

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('fig/mnist/m_complexity.png')
plt.savefig('fig/mnist/m_complexity.eps')