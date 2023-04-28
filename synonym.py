from sklearn.cluster import KMeans
import spacy
import torch
import numpy as np
from time import time

from dataset import dataset
from path import Path
from utils import *

def get_synonym(DATASET, m=5):
    '''
    given a dataset,
    return the synonym of each alphabet with number m.
    '''
    start_time = time()
    _dataset = dataset(DATASET, True)
    vocab = _dataset.vocab
    words_number = len(vocab)
    alphabet = range(words_number)
    ori_words = vocab.lookup_tokens(alphabet)
    nlp = spacy.load('en_core_web_lg')

    vector_size = torch.tensor(nlp('Hello').vector).shape[0]
    vectors = torch.zeros((words_number, vector_size))
    has_vector = []

    for idx, word in enumerate(ori_words):
        _word = nlp(word)
        if _word.has_vector and _word.vector_norm:
            vectors[idx] = torch.from_numpy(_word.vector)
            has_vector.append(True)
        else:
            has_vector.append(False)
            #print(word, _word.vector)
    assert len(has_vector) == words_number
    vectors.to(dev())

    all_synonym = []
    for idx, word in enumerate(ori_words):
        if has_vector[idx]:
            synonym = []
            vec = vectors[idx]
            diff = vectors - vec
            diff = (diff * diff).sum(dim = 1)
            sorted, indices = torch.sort(diff)
            index = 0
            while len(synonym)<m:
                if has_vector[indices[index]]:
                    synonym.append(indices[index])
                index += 1
        else:
            synonym = [-1] * m

        synonym = torch.tensor(synonym).to(dev())
        all_synonym.append(synonym)
        if idx % 1000 == 0:
            print(f'{idx} words checked.')
    all_synonym = torch.stack(all_synonym).to(dev())
    
    torch.save(all_synonym, Path+DATASET+'_synonym.pth')
    print(f'{DATASET} synonym ready. Use time:{time()-start_time:.1f}')

def get_alphabet(DATASET, CLUSTER):
    start_time = time()
    _dataset = dataset(DATASET, True)
    vocab = _dataset.vocab
    words_number = len(vocab)
    ori_alphabet = range(words_number)
    ori_words = vocab.lookup_tokens(ori_alphabet)
    nlp = spacy.load('en_core_web_lg')

    vector_size = torch.tensor(nlp('Hello').vector).shape[0]
    vectors = []
    has_vector = []

    for idx, word in enumerate(ori_words):
        _word = nlp(word)
        if _word.has_vector and _word.vector_norm:
            vectors.append(torch.from_numpy(_word.vector))
            has_vector.append(True)
        else:
            has_vector.append(False)
            #print(word, _word.vector)
    assert len(has_vector) == words_number

    vectors = torch.stack(vectors)
    vectors = vectors.numpy()

    print(f'vectors ready. Use time:{time()-start_time:.1f}')
    current_time = time()
    kmeans = KMeans(n_clusters=CLUSTER).fit(vectors)
    print(f'kmeans ready. Use time:{time()-current_time:.1f}')

    alphabet = []
    index = 0
    offset = 0
    for idx, word in enumerate(ori_words):
        if has_vector[idx]:
            alphabet.append(kmeans.labels_[index])
            index += 1
        else:
            alphabet.append(CLUSTER + offset)
            offset += 1 
    alphabet=torch.tensor(alphabet)

    torch.save(alphabet, Path+DATASET+'_alphabet.pth')
    print(f'{DATASET} alphabet ready. Use time:{time()-start_time:.1f}')


if __name__ == '__main__':
    
    alphabet = torch.load(Path+'news_alphabet.pth')
    _dataset = dataset('news', True)
    vocab = _dataset.vocab
    pass