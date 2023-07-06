import matplotlib.pyplot as plt
from problexity.classification import f1
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.decomposition import PCA
import concepts
import torchvision
from torchvision.transforms import Compose, ToTensor
from ConditionalEvidenceStream import ConditionalEvidenceStream
from utils import make_condition_map, mix_to_factor
from scipy.ndimage import gaussian_filter1d
from scipy import stats

np.set_printoptions(precision=3, suppress=True)

root = './files/'
transform = Compose([ToTensor()])

n_chunks = 1000
chunk_size = 200
n_concepts = 520

n_samples = 60000
factor_range = (.1,.9)
n_cycles = 3 # conditional cycles

print('Loading MNIST dataset...')
data = torchvision.datasets.MNIST(root, 
                                  train=True, 
                                  download=True, 
                                  transform = transform)

print('Transforming MNIST dataset...')
X = data.data.reshape(n_samples,-1)
y = data.targets.numpy()

# Mamy prawo użyć PCA, bo MNIST jest już zredukowany do 784 cech
# A tak naprawdę to dlatego, że dla nas to tylko kompresor
print('Applying PCA...')
X = PCA(n_components=0.8).fit_transform(X)
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)
print(X.shape)

print('Establishing distribution factor (difficulty)...')
factor = mix_to_factor(X)
print(factor.shape)

print('Preparing condition map...')
condition_map = make_condition_map(n_cycles=n_cycles, 
                                   n_concepts=n_concepts, 
                                   factor=factor, 
                                   factor_range=factor_range)
print(condition_map.shape)

print('Generating concept probabilities...')
cp = concepts.concept_proba(n_concepts=n_concepts,
                            n_chunks=n_chunks,
                            # mode='instant',
                            mode='normal', 
                            # compression='log', 
                            normalize=True,
                            sigma=1)
print(cp.shape)

print('Initializing stream...')
stream = ConditionalEvidenceStream(X, y,
                                   condition_map.T,
                                   cp,
                                   chunk_size=chunk_size,
                                   fragile=False)



fig, ax = plt.subplots(2,3,figsize=(15,12))

ax = ax.ravel()

max_support = []
selected_clf = []
_selected_clf = []

rs = 3423
clfs = [
    MLPClassifier(random_state=rs, hidden_layer_sizes=(5)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(10)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(25)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(50)),
    MLPClassifier(random_state=rs, hidden_layer_sizes=(75)),
]
thresholds = [0.9, 0.85, 0.8, 0.75, 0.7]
switch_when = 10
switch_count = 0

clf_acc = []
all_clf_acc = [[] for i in range(len(clfs))]
all_clf_supp = [[] for i in range(len(clfs))]


curr_clf_id = 0
max_epochs=250

for i in range(n_chunks):
    
    X, y = stream.get_chunk()
    # print(y)      

    # # Przez pierwsze x chunków tylko szkolenie klasyfikatorów
    # if i<20:
    #     for clf in clfs:
    #         clf.partial_fit(X,y,np.arange(10))
    #         for e in range(max_epochs):
    #             #TOSTY 
    #             #H0: Próba pochodzi z populacji o rozkładzie normalnym
    #             proba = clf.predict_proba(X)
    #             max_proba = np.max(proba, axis=1)
    #             stat, p = stats.shapiro(max_proba)
    #             # print(stat, p)
    #             if p>=0.05:
    #                 # H0 potwierdzona = r. normalny
    #                 print('BREAK')
    #                 # exit()
    #                 break
    #             # H0 odrzucona = nienormalny
    #             clf.partial_fit(X,y)
             
    # Przez pierwsze x chunków tylko szkolenie klasyfikatorów
    if i<20:
        for clf in clfs:
            clf.partial_fit(X,y,np.arange(10))
            for e in range(max_epochs):
                #ALE UCZMY JE TAK DŁUGO, ŻEBY MAX. WSPARCIE WYNOSIŁO ŚREDNIO 80% 
                proba = clf.predict_proba(X)
                mean_proba = np.mean(np.max(proba, axis=1))
                if mean_proba>0.8:
                    print('BREAK')
                    break
                clf.partial_fit(X,y)       
        

    # Przez kolejne estymacja i tylko inferencja
    else:
        
        # Check certainty
        proba = clfs[curr_clf_id].predict_proba(X)
        max_proba = np.max(proba, axis=1)
        mean_max_proba = np.mean(max_proba)
        print(mean_max_proba)
        max_support.append(mean_max_proba)
        
        print(np.argwhere(thresholds>mean_max_proba))
        # exit()
        _curr_clf_id = np.argwhere(thresholds>mean_max_proba).flatten()[-1]
        if _curr_clf_id != curr_clf_id:
            #check switch
            if switch_count==switch_when:
                #up or down
                if _curr_clf_id>curr_clf_id:
                    curr_clf_id+=1
                else:
                    curr_clf_id-=1
                switch_count = 0
            else:
                switch_count+=1
                
        _selected_clf.append(_curr_clf_id)
        selected_clf.append(curr_clf_id)

        pred = clfs[curr_clf_id].predict(X)
        clf_acc.append(accuracy_score(y_pred=pred, y_true=y))
        
        for c_id, c in enumerate(clfs): 
            all_clf_acc[c_id].append(accuracy_score(y_pred=c.predict(X), y_true=y))

            mean_max_proba = np.mean(np.max(clfs[c_id].predict_proba(X), axis=1))
            all_clf_supp[c_id].append(mean_max_proba)
        
        
    # PLOT 
    if i%10==0:
        
        ax[0].set_title('Support')
        ax[0].plot(max_support, alpha=0.1, color='b')
        ax[0].plot(gaussian_filter1d(max_support,3), label='mean max proba', color='b')
        ax[0].set_ylim(0.5,1)
        ax[0].hlines(thresholds, 0, len(max_support), label='thresholds', ls=':', color='r')
        
        ax[1].set_title('Selected classifier accuracy')
        ax[1].plot(clf_acc, c='tomato', alpha=0.1)
        ax[1].plot(gaussian_filter1d(clf_acc, 3), label='acc', c='tomato')
        ax[1].set_ylim(0.1,1)
        
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
        ax[3].set_ylim(0.1,1)
        
        ax[4].set_title('All classifiers suport')
        all_clf_arr = np.array(all_clf_supp)
        cols = plt.cm.coolwarm(np.linspace(0,1,len(clfs)))
        for c_id in range(len(clfs)):
            temp = all_clf_arr[c_id]
            ax[4].plot(temp, c=cols[c_id], alpha=0.1)
            ax[4].plot(gaussian_filter1d(temp, 3), label=c_id, c=cols[c_id])
        ax[4].set_ylim(0.1,1)
        
        for a in ax:
            a.grid(ls=':')
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
        
        for a  in ax[1:]:
            a.legend(frameon=False)
        
        fig.tight_layout()
        fig.savefig('foo.png')
        # fig.savefig('temp/%04d.png' % i)
        
        for a in ax:
            a.cla()


