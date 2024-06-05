import matplotlib.pyplot as plt
from poc.StreamGenerator import StreamGenerator
from problexity.classification import f1
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

n_chunks=400

stream = StreamGenerator(n_chunks=n_chunks,
        chunk_size=200,
        random_state=None,
        n_drifts=0,
        concept_sigmoid_spacing=None,
        n_classes=2,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        recurring=False,
        weights=None,
        incremental=False,
        y_flip=0.01,
        difficulty_n_drifts=4,
        difficulty_concept_sigmoid_spacing=5,
        difficulty_amplitude=1.5,
        class_sep=0.7)

fig, ax = plt.subplots(2,2,figsize=(12,12))
fig2, ax2 = plt.subplots(1,5,figsize=(20,4))

ax = ax.ravel()

f1_comp = []
selected_clf = []

clf_acc = []
rs = 3423
clfs = [
    MLPClassifier(random_state=rs, hidden_layer_sizes=(6)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(10)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(25)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(50)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(100)),
]
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4]

curr_clf_id = 0

for i in range(n_chunks):
    
    X, y = stream.get_chunk()
    
    # Przez pierwsze x chunków tylko szkolenie klasyfikatorów
    if i<20:
        for clf in clfs:
            clf.fit(X,y)
        
        #Plot granice decyzyjne
        space = np.linspace(-1,7,50)
        mesh = np.array([[x,y] for x in space for y in space])

        for clf_id, clf in enumerate(clfs):
            pred = clf.predict_proba(mesh)[:,1]
            ax2[clf_id].scatter(mesh[:,0], mesh[:,1], c=pred, cmap='coolwarm')
        
        fig2.tight_layout()
        fig2.savefig('clfs.png')

    # Przez kolejne estymacja i tylko inferencja
    else:
        diff = f1(X, y)
        f1_comp.append(diff)
        
        curr_clf_id = np.argwhere(thresholds<diff).flatten()[-1]
        selected_clf.append(curr_clf_id)

        clf_acc.append(accuracy_score(y_pred=clfs[curr_clf_id].predict(X), y_true=y))

     
        
    
    
    fig.suptitle('chunk %i' % i)
    ax[0].scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    ax[0].set_xlim(-7,7)
    ax[0].set_ylim(-7,7)
    
    ax[1].plot(f1_comp, label='F1')
    ax[1].set_ylim(0,1)
    ax[1].hlines(thresholds, 0, len(f1_comp), label='thresholds', ls=':', color='r')
    
    ax[2].plot(clf_acc, label='acc', c='tomato')
    ax[2].set_ylim(0.5,1)
    
    ax[3].scatter(np.arange(len(selected_clf)), selected_clf, label='selected clf')
    
    for a in ax:
        a.grid(ls=':')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    
    for a  in ax[1:]:
        a.legend(frameon=False)
    
    fig.tight_layout()
    fig.savefig('foo.png')
    fig.savefig('temp/%04d.png' % i)
    
    for a in ax:
        a.cla()


