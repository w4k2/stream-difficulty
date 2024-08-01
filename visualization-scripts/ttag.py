import numpy as np
import matplotlib.pyplot as plt


q = 100

rt_min, rt_max = 0.3, 0.9
ra_min, ra_max = 0.001, 0.035

r_time_loss = np.linspace(rt_min, rt_max, q)
r_acc_loss = np.linspace(ra_min, ra_max,q)

r_t_a_m = np.load('results/r_t_a_m.npy')
t_m = r_t_a_m[...,0].flatten()
a_m = r_t_a_m[...,1].flatten()


r_t_a_s = np.load('results/r_t_a.npy')
t_s = r_t_a_s[...,0].flatten()
a_s = r_t_a_s[...,1].flatten()

print(t_s, a_s)

ttag = np.zeros((100,100))
for t_id, t in enumerate(r_time_loss):
    for a_id, a in enumerate(r_acc_loss):
        ttag[a_id, t_id] = (t*t)/a
        
xx, yy = np.meshgrid(r_time_loss, r_acc_loss)

fig, ax = plt.subplots(1,1,figsize=(7,7))
plt.suptitle('TTAG')

ax.scatter(xx, yy, c = ttag, cmap='coolwarm')
a = 15
ax.set_xticks(r_time_loss[::a], ['%0.3f' % t for t in r_time_loss][::a])
ax.set_xlabel('relative time loss')
ax.set_yticks(r_acc_loss[::a], ['%0.3f' % t for t in r_acc_loss][::a])
ax.set_ylabel('relative accuracy loss')
ax.set_xlim(0.3, 0.9)
ax.set_ylim(0.001, 0.035)

ax.scatter(t_m, a_m, color='red', marker='x', label='mnist')
ax.scatter(t_s, a_s, color='blue', marker='x', label='svhn')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('foo.png')
plt.close()



fig, ax = plt.subplots(1,1,figsize=(7,7/1.618))
#plt.suptitle('TTAG')

ax.imshow(ttag.reshape(xx.shape[0], yy.shape[0]), origin='lower', cmap='coolwarm',
          aspect=0.618, interpolation='none')

a = 15
# ax.set_xticks(r_time_loss[::a], ['%0.3f' % t for t in r_time_loss][::a])

ax.set_xlabel('relative time loss')
ax.set_xticks(np.linspace(0,q-1,q)[::a], ['%0.3f' % t for t in r_time_loss][::a])

ax.set_ylabel('relative accuracy loss')
ax.set_yticks(np.linspace(0,q-1,q)[::a], ['%0.3f' % t for t in r_acc_loss][::a])

# ax.set_xlim(0.3, 0.9)
# ax.set_ylim(0.001, 0.035)



t_ms = q * (t_m - rt_min) / (rt_max - rt_min)
a_ms = q * (a_m - ra_min) / (ra_max - ra_min)

t_ss = q * (t_s - rt_min) / (rt_max - rt_min)
a_ss = q * (a_s - ra_min) / (ra_max - ra_min)

ax.scatter(t_ms, a_ms, color='white', marker='o', s=50)
ax.scatter(t_ms, a_ms, color='red', marker='o', label='MNIST', s=25)
ax.scatter(t_ss, a_ss, color='white', marker='o', s=50)
ax.scatter(t_ss, a_ss, color='blue', marker='o', label='SVHN', s=25)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(ls=':')
plt.legend()
plt.tight_layout()
plt.savefig('ttag.eps')
plt.savefig('foo.png')