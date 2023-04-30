import matplotlib.pyplot as plt
from StreamGenerator import StreamGenerator
from problexity.classification import f1, f1v

n_chunks=200

stream = StreamGenerator(n_chunks=n_chunks,
        chunk_size=200,
        random_state=None,
        n_drifts=0,
        concept_sigmoid_spacing=None,
        n_classes=2,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        recurring=False,
        weights=None,
        incremental=False,
        y_flip=0.01,
        difficulty_n_drifts=3,
        difficulty_concept_sigmoid_spacing=5,
        difficulty_amplitude=2)

fig, ax = plt.subplots(1,2,figsize=(12,6))
f1_comp = []
f1v_comp = []

for i in range(n_chunks):
    X, y = stream.get_chunk()
    
    f1_comp.append(f1(X, y))
    f1v_comp.append(f1v(X, y))
    
    plt.suptitle('chunk %i' % i)
    ax[0].scatter(X[:,0], X[:,1], c=y)
    ax[0].set_xlim(-7,7)
    ax[0].set_ylim(-7,7)
    ax[1].plot(f1_comp, label='F1')
    ax[1].plot(f1v_comp, label='F1v')
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('temp/%04d.png' % i)
    
    ax[0].cla()
    ax[1].cla()
    # time.sleep(1)

