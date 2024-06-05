import numpy as np
from tabulate import tabulate

# MNIST
# mn_0_quant.tflite - acc - MACC: 83175 - Flash: 30765 B - RAM: 4144 B - Latency: 10.821 ms
# mn_1_quant.tflite - acc - MACC: 166340 - Flash: 38138 B - RAM: 6304 B - Latency: 15.230 ms
# mn_2_quant.tflite - acc - MACC: 394955 - Flash: 36009 B - RAM: 10800 B - Latency: 25.240 ms
# mn_3_quant.tflite - acc - MACC: 332670 - Flash: 52884 B - RAM: 10624 B - Latency: 25.031 ms
# mn_4_quant.tflite - acc - MACC: 1987995 - Flash: 61997 B - RAM: 15308 B - Latency: 83.646 ms


#(repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))
sel = np.load('results/e1_selected_m.npy')
_accs = np.load('results/e1_accs_m.npy')

maccs = [83175, 166340, 394955, 332670, 1987995]
latencies = np.array([10.821, 15.230, 25.240, 25.031, 83.646])

###
chunk_size = [50, 150, 300, 500]
n_cycles = [3, 5, 10, 25]
modes = {
    'instant': {'mode': 'instant'},
    'linear': {'mode': 'linear'},
    'normal': {'mode': 'normal', 'sigma': 1},
    }

###

rows = []
rows.append(['stream', 'R acc', 'CDoS acc', 'CDos Latency (*1e3)', 'CDoS MACC(*1e9)', 'TTAG'])

r_t_a = np.zeros((3,4,4,2))

for m_id, m in enumerate(modes.keys()):
    for c_s_id, c_s in enumerate(chunk_size):
        for c_id, c in enumerate(n_cycles):
            # Name
            str_name = '%s_cs%i_c%i' % (m, c_s, c)
            
            # Reference
            r_acc = np.mean(_accs[:,c_s_id,c_id,m_id,:,4])
            r_latency = latencies[4]*c_s
            r_macc = maccs[4]*c_s
            
            # CDOS
            sel_temp = sel[:,c_s_id,c_id,m_id]
            
            _cdos_acc = _accs[:,c_s_id,c_id,m_id]
            cdos_acc = []
            for rep in range(5):
                for chunk in range(1000):
                    s = int(sel_temp[rep,chunk])
                    cdos_acc.append(_cdos_acc[rep,chunk,s])
            cdos_acc = np.mean(cdos_acc)
            cdos_letency = np.mean(latencies[sel_temp.astype(int)])*c_s
            cdos_macc = np.mean(np.array(maccs)[sel_temp.astype(int)])*c_s
            
            # Time to accuracy gain
            rel_latency_loss = (r_latency-cdos_letency)/r_latency
            rel_macc_loss = (r_macc-cdos_macc)/r_macc
            rel_acc_loss = (r_acc-cdos_acc)/r_acc
            print(rel_latency_loss, rel_macc_loss, rel_acc_loss)
            ttag = (rel_macc_loss*rel_latency_loss)/rel_acc_loss

            r = [ str_name, 
                 '%0.3f' % r_acc, 
                 '%0.3f (%0.3f)' % (cdos_acc, cdos_acc-r_acc), 
                 '%0.3f (%0.3f)' % (cdos_letency/1e3, (cdos_letency-r_latency)/1e3), 
                 '%0.3f (%0.3f)' % (cdos_macc/1e9, (cdos_macc-r_macc)/1e9),
                 '%0.3f' % ttag]
            rows.append(r)
            
            r_t_a[m_id, c_s_id, c_id, 0] = np.sqrt(rel_macc_loss*rel_latency_loss)
            r_t_a[m_id, c_s_id, c_id, 1] = rel_acc_loss
            
    
with open('tables/mnist.txt', 'w') as f:
    f.write(tabulate(rows, tablefmt='latex'))
    
np.save('results/r_t_a_m.npy', r_t_a)
            