import numpy as np
from tabulate import tabulate

# SVHN
# 0_quant.tflite - acc - MACC: 310741 - Flash: 33786 B - RAM: 8260 B - Latency: 25.012 ms
# 1_quant.tflite - acc - MACC: 429336 - Flash: 28756 B - RAM: 9016 B - Latency: 29.783 ms
# 2_quant.tflite - acc - MACC: 980183 - Flash: 38034 B - RAM: 11536 B - Latency: 49.257 ms
# 3_quant.tflite - acc - MACC: 2708821 - Flash: 54007 B - RAM: 15080 B - Latency: 98.771 ms
# 4_quant.tflite - acc - MACC: 4011611 - Flash: 67022 B - RAM: 16948 B - Latency: 146.475 ms


#(repeats, len(chunk_size), len(n_cycles), len(modes), n_chunks))
sel = np.load('results/e1_selected_3.npy')
_accs = np.load('results/e1_accs_3.npy')

maccs = [310741, 429336, 980183, 2708821, 4011611]
latencies = np.array([25.012, 29.783, 49.257, 98.771, 146.475])

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
rows.append(['stream', 'R acc', 'CDoS acc', 'CDos latency', 'CDoS macc (*1e6)', 'TTAG'])

for m_id, m in enumerate(modes.keys()):
    for c_s_id, c_s in enumerate(chunk_size):
        for c_id, c in enumerate(n_cycles):
            # Name
            str_name = '%s_cs%i_c%i' % (m, c_s, c)
            
            # Reference
            r_acc = np.mean(_accs[:,c_s_id,c_id,m_id,:,4])
            r_latency = latencies[4]
            r_macc = maccs[4]
            
            # CDOS
            sel_temp = sel[:,c_s_id,c_id,m_id]
            
            _cdos_acc = _accs[:,c_s_id,c_id,m_id]
            cdos_acc = []
            for rep in range(5):
                for chunk in range(1000):
                    s = int(sel_temp[rep,chunk])
                    cdos_acc.append(_cdos_acc[rep,chunk,s])
            cdos_acc = np.mean(cdos_acc)
            cdos_letency = np.mean(latencies[sel_temp.astype(int)])
            cdos_macc = np.mean(np.array(maccs)[sel_temp.astype(int)])
            
            # Time to accuracy gain
            rel_latency_loss = np.abs((cdos_letency-r_latency)/r_latency)
            rel_macc_loss = np.abs((cdos_macc-r_macc)/r_macc)
            rel_acc_loss = np.abs((cdos_acc-r_acc)/r_acc)
            ttag = (rel_macc_loss*rel_latency_loss)/rel_acc_loss
            
            r = [ str_name, 
                 '%0.3f' % r_acc, 
                 '%0.3f (%0.3f)' % (cdos_acc, cdos_acc-r_acc), 
                 '%0.3f (%0.3f)' % (cdos_letency, cdos_letency-r_latency), 
                 '%0.3f (%0.3f)' % (cdos_macc/1e6, (cdos_macc-r_macc)/1e6),
                 '%0.3f' % ttag]
            rows.append(r)
            
with open('tables/svhn.txt', 'w') as f:
    f.write(tabulate(rows, tablefmt='latex'))
            