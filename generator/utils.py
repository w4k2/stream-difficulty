from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import quantile_transform
import numpy as np
import torch
import torch.nn as nn

# Maska co najmniej
def make_condition_map(n_cycles, n_concepts, factor, factor_range):
    # Factor must be the size of a problem
    # Decydujemy, ile chcemy uzyskać cyklów warunkowych i wytwarzamy
    # odpowiednią sinusoidę. Dalej normalizujemy ją przedziałowo i kwantyzujemy
    # do liczby koncepcji tak, aby można było ją wykorzystać jako zbiór indeksów
    # wartości warunku.
    condition_base = np.sin(np.linspace(0, np.pi*2*n_cycles, n_concepts))
    condition_base -= np.min(condition_base)
    condition_base /= np.max(condition_base)
    condition_base *= (n_concepts-1)
    condition_base = condition_base.astype(int)

    # Wprowadzamy progi warunków
    conditions = np.linspace(*factor_range, n_concepts)
    conditions_vals = conditions[condition_base]

    # Uzyskujemy mapę warunków przez przyrównanie dające nam macierz
    # Samples X concepts
    condition_map = factor[:,None] > conditions_vals[None,:]

    return condition_map
   
def mix_to_factor(X, n_components=10, covtype='spherical'):
    # Modelujemy miksturę Gaussowską
    mixture = GaussianMixture(n_components=n_components,
                              covariance_type=covtype).fit(X)

    # Pobieramy i normalizujemy wsparcia
    # Dysponujemy macierzą mpp o wymiarach liczba obiektów x liczba centroidów
    # mikstury. Sumujemy więc po drugim wymiarze i kolejny raz normalizujemy
    factor = quantile_transform(mixture.predict_proba(X),
                                output_distribution='uniform')
    factor = quantile_transform(np.sum(factor, axis=1).reshape(-1,1),
                                output_distribution='uniform').ravel()
    
    return factor
    
    

def get_th(clfs, train_X, chunk_size, alpha=0.92):
    # alpha -- na ile trudniejszy będzie zbiór testowy niż treningowy 
    # -- rozciągnięcie th 
    # -- jak większe alpha to mniej rozrzuca
    max_probas=[]
    for c in clfs:
        proba = nn.Softmax(dim=1)(c(train_X))
        max_proba = torch.max(proba, dim=1)[0]
        max_probas.append(max_proba.detach().numpy())
        
    mp = np.array(max_probas).flatten()
    aa = int(len(mp)/chunk_size)
    mp = mp[:aa*chunk_size]
    mp = mp.reshape(aa,chunk_size)
    mp = np.mean(mp, axis=1)

    n_bins=len(clfs)+1
    bin_size = int(len(mp)/n_bins)

    th = []
    argsort_mp = np.argsort(mp)

    for b in range(1,n_bins):
        th.append(mp[argsort_mp[b*bin_size]])


    th.reverse()
    th = np.array(th)*np.linspace(1,alpha,len(th))

    th[0]=1.
        
    return th