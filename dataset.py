import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


from utils import dev

class dataset(Dataset):
    def __init__(self, type='news', train=True, shuffle=True):
        super().__init__()
        self.type = type
        if type == 'news':
            if train:
                self.filename = 'data/news/train_data_pytorch.csv'
            else:
                self.filename = 'data/news/test_data_pytorch.csv'
            self.data = pd.read_csv(self.filename, names=['index', 'title', 'content', 'class'])
            
            # The vocab of test data should be consist with train data.
            self.vocab_filename = 'data/news/train_data_pytorch.csv'
            self.vocab_data = pd.read_csv(self.vocab_filename, names=['index', 'title', 'content', 'class'])
            
            self.content_index = 1
            self.label_index = 3
            self.max_length = 25
            self.classes = 7

        elif type == 'toxic':
            if train:
                self.filename = 'data/toxic/train.csv'
            else:
                self.filename = 'data/toxic/test.csv'
            self.data = pd.read_csv(self.filename, names=['content', 'class'])
            
            self.vocab_filename = 'data/toxic/train.csv'
            self.vocab_data = pd.read_csv(self.vocab_filename, names=['content', 'class'])
            
            self.content_index = 0
            self.label_index = 1
            self.max_length = 150
            self.classes = 2

        else:
            assert False, 'Invalid dataset type'

        self.tokenizer = get_tokenizer('basic_english')
        def yield_tokens():
            for text in self.vocab_data.iloc[:, self.content_index]:
                try:
                    yield self.tokenizer(text) 
                except:
                    yield []
        self.vocab = build_vocab_from_iterator(yield_tokens(), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

        self.data_num = len(self.data)
        if shuffle:
            order = np.random.permutation(self.data_num)
            self.data = self.data.take(order)

        self.int_data = []

        for idx in range(self.data_num):
            int_data = torch.tensor(self.query_vocab(self.data.iloc[idx, self.content_index]),device=dev()).reshape(1,-1)
            self.int_data.append(int_data)
        self.int_data = torch.concat(self.int_data, dim=0)
        self.int_data.to(dev())
    # transform text into a list of int, with same length(max_length).
    def query_vocab(self, text):
        try:
            vocab = self.vocab(self.tokenizer(text))
        except:
            vocab = []
        
        if len(vocab) > self.max_length:
            vocab = vocab[:self.max_length]
        elif len(vocab) < self.max_length:
            vocab += [0] * (self.max_length - len(vocab))
        return vocab

    def __len__(self):
        return self.data_num
        
    def __getitem__(self, idx):
        text = self.int_data[idx]
        label = int(self.data.iloc[idx, self.label_index])
        return text, label

    # prepare data for training models.
    def batch(self, batch_size = 16):
        batched_data = []

        idx = 1
        while idx * batch_size < self.data_num:
            texts = self.int_data[(idx-1)*batch_size:idx*batch_size]
            labels = self.data.iloc[(idx-1)*batch_size:idx*batch_size, self.label_index]
            batched_data.append([texts, labels])
            idx += 1
        if not self.data_num % batch_size == 0:
            texts = self.int_data[(idx-1)*batch_size:]
            labels = self.data.iloc[(idx-1)*batch_size:, self.label_index]
            batched_data.append([texts, labels])

        return batched_data
    
    