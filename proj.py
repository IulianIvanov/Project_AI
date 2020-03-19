# coding: utf-8
# pip install numpy scikit-learn

import numpy as np
import re
import os
from collections import defaultdict
from sklearn import svm
from sklearn.model_selection import KFold
import time

PRIMELE_N_CUVINTE = 1000
time_elapsed = time.time()

def accuracy(y, p):
    return 100 * (y==p).astype('int').mean()

def files_in_folder(mypath):
    fisiere = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            fisiere.append(os.path.join(mypath, f))
    return sorted(fisiere)

def extrage_fisier_fara_extensie(cale_catre_fisier):
    nume_fisier = os.path.basename(cale_catre_fisier)
    nume_fisier_fara_extensie = nume_fisier.replace('.txt', '')
    return nume_fisier_fara_extensie

def citeste_texte_din_director(cale):
    date_text = []
    iduri_text = []
    for fis in files_in_folder(cale):
        id_fis = extrage_fisier_fara_extensie(fis)
        iduri_text.append(id_fis)
        with open(fis, 'r', encoding='utf-8') as fin:
            text = fin.read()

        # incercati cu si fara punctuatie sau cu lowercase
        text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        cuvinte_text = text_fara_punct.split()
        date_text.append(cuvinte_text)
    return (iduri_text, date_text)

### citim datele ###
dir_path = 'E:/Python Projects/project/venv/trainData/trainData'
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))

train_data_path = os.path.join(dir_path, 'trainExamples')
iduri_train, data = citeste_texte_din_director(train_data_path)


print(data[0][:10])
### citim datele ###


### numaram cuvintele din toate documentele ###
contor_cuvinte = defaultdict(int)
for doc in data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1': frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)

# extragem primele 1000 cele mai frecvente cuvinte din toate textele
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])


list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)
### numaram cuvintele din toate documentele ###


def get_bow(text, lista_de_cuvinte):
    '''
    returneaza BoW corespunzator unui text impartit in cuvinte
    in functie de lista de cuvinte selectate
    ''' 
    contor = dict()
    cuvinte = set(lista_de_cuvinte)
    for cuvant in cuvinte:
        contor[cuvant] = 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1
    return contor

def get_bow_pe_corpus(corpus, lista):
    '''
    returneaza BoW normalizat
    corespunzator pentru un intreg set de texte
    sub forma de matrice np.array
    ''' 
    bow = np.zeros((len(corpus), len(lista)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, lista)
        ''' 
            bow e dictionar.
            bow.values() e un obiect de tipul dict_values 
            care contine valorile dictionarului
            trebuie convertit in lista apoi in numpy.array
        '''
        v = np.array(list(bow_dict.values()))
        # incercati si alte tipuri de normalizari
        # v = v / np.sqrt(np.sum(v ** 2))
        bow[idx] = v / np.sqrt(np.sum(v ** 2))
    return bow

data_bow = get_bow_pe_corpus(data, list_of_selected_words)
print ("Data bow are shape: ", data_bow.shape)

nr_exemple_train = 2000
nr_exemple_valid = 500
nr_exemple_test = len(data) - (nr_exemple_train + nr_exemple_valid)

indici_train = np.arange(0, nr_exemple_train)
print("Indici train :", indici_train)
indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
print("Indici valid :", indici_valid)
indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))
print("Indici test :", indici_test)

print ("Histograma cu clasele din train: ", np.histogram(labels[indici_train])[0])
print ("Histograma cu clasele din validation: ", np.histogram(labels[indici_valid])[0])
print ("Histograma cu clasele din test: ", np.histogram(labels[indici_test])[0])
# clasele sunt balansate in cazul asta pentru train, valid si nr_exemple_test


# cu cat creste C, cu atat clasificatorul este mai predispus sa faca overfit
# https://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    clasificator = svm.SVC(C = C, kernel = 'poly')
    clasificator.fit(data_bow[indici_train, :], labels[indici_train])
    predictii = clasificator.predict(data_bow[indici_valid, :])
    print ("Acuratete pe validare cu C =", C, ": ", accuracy(predictii, labels[indici_valid]))


# concatenam indicii de train si validare

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10)

scores_svm = []

for train_index, test_index in skf.split(indici_train, labels[indici_train]):
#    print("TRAIN:", train_index, "TEST:", test_index)

    indici_train = np.arange(0, nr_exemple_train)
    indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
    indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))

#    indici_train_valid = np.concatenate([indici_train, indici_valid])
    clasificator = svm.SVC(C = 100, kernel = 'rbf', gamma= 1.0, random_state=None)
    clasificator.fit(data_bow[train_index, :], labels[train_index])
    predictii = clasificator.predict(data_bow[test_index])
    scores_svm.append(accuracy(predictii, labels[test_index]))
    print("Acuratetea :" , accuracy(predictii, labels[test_index]) )

print(" Media acuratetilor " , sum(scores_svm)/10)




def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("Id,Prediction\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(id_text + ',' + str(int(pred)) + '\n')


cale_data_test = 'E:/Python Projects/project/venv/testData-public/testData-public'
indici_test,date_test = citeste_texte_din_director (cale_data_test)
print('Am citit',len(date_test))
data_bow_test = get_bow_pe_corpus(date_test, list_of_selected_words)
clf = svm.SVC(C = 100, kernel = 'rbf', gamma= 1.0, random_state=None)
clf.fit(data_bow,labels)
predictii = clf.predict(data_bow_test)
scrie_fisier_submission('bun.csv',predictii,indici_test)


indici_train_valid = np.concatenate([indici_train, indici_valid])
clasificator = svm.SVC(C = 100, kernel = 'rbf', gamma= 1.0, random_state=None)
clasificator.fit(data_bow[indici_train_valid, :], labels[indici_train_valid])
predictii = clasificator.predict(data_bow[indici_test])

vec_labels = np.squeeze(np.asarray(labels[indici_test]))
vec_predictii = np.squeeze(np.asarray(predictii))

vec_labels= vec_labels.astype(int)
vec_predictii = vec_predictii.astype(int)

#matricea de confuzie
M = np.zeros((11,11))
for adev,pred in zip (vec_labels, vec_predictii):
    M[adev, pred] = M[adev, pred] +1
print(M)

print("--- %s seconds ---" % (time.time() - time_elapsed))





