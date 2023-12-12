import numpy as np
import matplotlib.pyplot as plt

r_time_loss = np.linspace(0.3,0.9,100)
r_acc_loss = np.linspace(0.001,0.035,100)

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