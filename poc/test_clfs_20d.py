import matplotlib.pyplot as plt
from poc.StreamGenerator import StreamGenerator
from problexity.classification import f1
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

n_chunks=400

stream = StreamGenerator(n_chunks=n_chunks,
        chunk_size=200,
        random_state=None,
        n_drifts=0,
        concept_sigmoid_spacing=None,
        n_classes=2,
        n_features=20,
        n_informative=15,
        n_redundant=5,
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
_selected_clf = []

rs = 3423
clfs = [
    MLPClassifier(random_state=rs, hidden_layer_sizes=(3)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(5)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(10)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(20)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(50)),
]
thresholds = [0.0, 0.2, 0.4, 0.6, 0.8]
switch_when = 5
switch_count = 0

clf_acc = []
all_clf_acc = [[] for i in range(len(clfs))]


curr_clf_id = 0

for i in range(n_chunks):
    
    X, y = stream.get_chunk()
    
    if i==0:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
    else:
        X_pca = pca.transform(X)
        

    # Przez pierwsze x chunków tylko szkolenie klasyfikatorów
    if i<20:
        for clf in clfs:
            clf.fit(X,y)
        
        if i%10==0:

            #Plot granice decyzyjne
            xmin = np.min(X_pca)
            xmax = np.max(X_pca)
            space = np.linspace(xmin,xmax,50)
            mesh = np.array([[x,y] for x in space for y in space])

            proj = pca.inverse_transform(mesh)
        
            for clf_id, clf in enumerate(clfs):
                ax2[clf_id].cla()
                pred = clf.predict_proba(proj)[:,1]
                ax2[clf_id].scatter(mesh[:,0], mesh[:,1], c=pred, cmap='coolwarm')
                ax2[clf_id].scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm')
            
            fig2.tight_layout()
            fig2.savefig('clfs.png')
            

    # Przez kolejne estymacja i tylko inferencja
    else:
        diff = f1(X, y)
        f1_comp.append(diff)
        
        _curr_clf_id = np.argwhere(thresholds<diff).flatten()[-1]
        if _curr_clf_id != curr_clf_id:
            #check switch
            if switch_count==switch_when:
                curr_clf_id = _curr_clf_id
                switch_count = 0
            else:
                switch_count+=1
                
        _selected_clf.append(_curr_clf_id)
        selected_clf.append(curr_clf_id)

        clf_acc.append(accuracy_score(y_pred=clfs[curr_clf_id].predict(X), y_true=y))
        for c_id, c in enumerate(clfs): 
            all_clf_acc[c_id].append(accuracy_score(y_pred=c.predict(X), y_true=y))
     
    if i%10==0:
        
        ax[0].set_title('Complexity')
        ax[0].plot(f1_comp, alpha=0.1, color='b')
        ax[0].plot(gaussian_filter1d(f1_comp,3), label='F1', color='b')
        ax[0].set_ylim(0,1)
        ax[0].hlines(thresholds, 0, len(f1_comp), label='thresholds', ls=':', color='r')
        
        ax[1].set_title('Selected classifier accuracy')
        ax[1].plot(clf_acc, c='tomato', alpha=0.1)
        ax[1].plot(gaussian_filter1d(clf_acc, 3), label='acc', c='tomato')
        ax[1].set_ylim(0.5,1)
        
        ax[2].set_title('Selected classifier')
        ax[2].scatter(np.arange(len(selected_clf)), _selected_clf, c='b', alpha=0.2)
        ax[2].scatter(np.arange(len(selected_clf)), selected_clf, label='Index', c='b')

        ax[3].set_title('All classifiers accuracy')
        all_clf_arr = np.array(all_clf_acc)
        cols = plt.cm.coolwarm(np.linspace(0,1,len(clfs)))
        for c_id in range(len(clfs)):
            temp = all_clf_arr[c_id]
            ax[3].plot(temp, c=cols[c_id], alpha=0.1)
            ax[3].plot(gaussian_filter1d(temp, 3), label=c_id, c=cols[c_id])
        ax[3].set_ylim(0.5,1)
        
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


