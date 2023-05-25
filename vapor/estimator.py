import matplotlib.pyplot as plt
from StreamGenerator import StreamGenerator
from problexity.classification import f1, f1v
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPRegressor
import numpy as np

n_chunks=200

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
        difficulty_n_drifts=5,
        difficulty_concept_sigmoid_spacing=9,
        difficulty_amplitude=1.,
        class_sep=1)

fig, ax = plt.subplots(1,3,figsize=(15,6))

f1_comp = []
f1v_comp = []

clf_acc = []
clf = GaussianNB()

estim_err=[]
reg = MLPRegressor()

for i in range(n_chunks):
    X, y = stream.get_chunk()
    
    meta = np.array([np.mean(X[y==0]),
        np.mean(X[y==1]),
        np.std(X[y==0]),
        np.std(X[y==0])
        ]).reshape(1, -1)
    
    if i!=0:
        #Test
        clf_acc.append(accuracy_score(y_pred=clf.predict(X), y_true=y))
        
     # Train
    clf.fit(X,y)
    
    
    if i>0:
        if i>1:
            #Test
            estimated = reg.predict(meta)
            print(estimated, clf_acc[-1], np.abs(estimated - clf_acc[-1]))
            estim_err.append(np.abs(estimated - clf_acc[-1]))
        
        #Train
        [reg.partial_fit(meta, [clf_acc[-1]]) for ep in range(20)]
        
    f1_comp.append(f1(X, y))
    f1v_comp.append(f1v(X, y))
    
    plt.suptitle('chunk %i' % i)
    
    
    ax[0].scatter(X[:,0], X[:,1], c=y)
    ax[0].set_xlim(-7,7)
    ax[0].set_ylim(-7,7)
    ax[1].plot(f1_comp, label='F1')
    ax[1].plot(f1v_comp, label='F1v')
    ax[1].plot(clf_acc, label='acc')
    ax[1].legend()
    
    ax[2].plot(estim_err, label='estim AE', c='r', ls=':')
    ax[2].legend()

    
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('temp/%04d.png' % i)
    
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

