import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 4)

import time
import random

from device import dev
from path import Path, Log_Path
from dataset import dataset
from model import RNN
from preprocess import get_transitions, add_transitions

from util import get_matrices
from util import blank_filling, identical_filling, empirical_filling, weighted_filling, near_filling,uniform_filling
from util import none_regularization, linear_regularization, strong_linear_regularization

from synonym import get_synonym

from evaluation import evaluation

if __name__ == '__main__':

    # select the dataset. options: 'news', 'toxic'
    DATASET = 'toxic'

    # select the clusters number
    CLUSTER = 20

    # select the completion and regularization tactics
    COMPLETION = [weighted_filling, uniform_filling, blank_filling]
    REGULARIZATION = [strong_linear_regularization,linear_regularization ,none_regularization]

    # select the iteration times of using synonym to augmenting dataset
    NUM_EPOCHS = 10
    REPLACE_RATE = 0.4
    DROPOUT = 0.2

    start_time = time.time()
    # load model and dataset
    train_dataset = dataset(DATASET, True)
    test_dataset = dataset(DATASET, False)
    model = torch.load(Path+DATASET+'_model.pth')
    model.eval()
    vocab_num = len(train_dataset.vocab)
    state_num = CLUSTER + 1
    print(f'vocab: {vocab_num}')
    print(f'data number: {len(train_dataset.int_data)}')
    print(f'Model and dataset ready. Use time:{time.time()-start_time:.1f}')

    current_time = time.time()
    # get rnn prediction in test set
    rnn_prediction_container = []
    for idx, data in enumerate(test_dataset.int_data):
        # remove 0 at the end
        while len(data) > 1 and data[-1] == 0:
            data = data[0:len(data)-1]
        data = data.reshape(-1, 1)

        model.clear_output_sequence()
        _ = model(data)
        runtime_predict = model.runtime_predict()
        runtime_data = []
        for step_data in runtime_predict:
            step_data = step_data.flatten().detach()
            runtime_prediction = F.softmax(step_data,dim=0)
            runtime_data.append(runtime_prediction.reshape(1, -1))
        runtime_data = torch.concat(runtime_data, dim=0)
        rnn_prediction = torch.argmax(runtime_data[-1])
        rnn_prediction_container.append(rnn_prediction)

    transition_count, kmeans, state_weightes = get_transitions(model, train_dataset, CLUSTER)
    print(f'Transitions ready. Use time:{time.time()-current_time:.1f}')

    # generate state distance
    state_distance = torch.zeros((state_num, state_num),device=dev())
    for p in range(state_num):
        for q in range(state_num):
            diff = state_weightes[p] - state_weightes[q]
            state_distance[p, q] = (diff * diff).sum()
    state_distance = torch.exp(state_distance)
    state_distance = 1 / state_distance
    

    result = np.zeros((len(COMPLETION), len(REGULARIZATION)))
    completion_names = [c.__name__ for c in COMPLETION]
    regularization_names = [r.__name__ for r in REGULARIZATION]
    for i, completion in enumerate(COMPLETION):
        for j, regularization in enumerate(REGULARIZATION):
            current_time = time.time()

            transition_matrices = get_matrices(transition_count, state_distance, completion, regularization)
            correct_rate = evaluation(test_dataset, transition_matrices, state_weightes, rnn_prediction_container)
            result[i,j] = round(correct_rate*100, 2)
            print(f'{completion.__name__} & {regularization.__name__} : {round(correct_rate*100, 2)}%, {time.time() - current_time:.0f}s')
            
    result = pd.DataFrame(result, columns=regularization_names, index=completion_names)
    print(result)
    print(f'Evaluation done.')
    
    all_synonym = torch.load(Path+DATASET+'_synonym.pth')
    '''
    all_synonym is a tensor with size (vocab_num, m),
    where m is the number of synonym for each word.
    The [i,j]-th item of all_synonym indicates the j-th synonym of i-th word.
    If some word doe NOT have synonym, the i-th row will be filled with -1.
    '''
    current_time = time.time()
    for epoch in range(NUM_EPOCHS):
        current_time = time.time()
        all_data = []
        for id, data in enumerate(train_dataset.int_data):
            # remove 0 at the end
            while len(data) > 1 and data[-1] == 0:
                data = data[0:len(data)-1]
            #ori_data = data.clone()
            for idx, word in enumerate(data):
                if random.random() < REPLACE_RATE and word < (vocab_num/5):
                    i = random.randint(1, 4)
                    if all_synonym[word, 0].item() != -1:
                        data[idx] = all_synonym[word, i].item()
                elif random.random() < DROPOUT:
                    data[idx] = 0
            all_data.append(data)
        transition_count = add_transitions(model,all_data,transition_count,kmeans)
        print(f'new transition count ready. Use time:{time.time()-current_time:.1f}')
        if (epoch+1) % 1 == 0:
            result = np.zeros((len(COMPLETION), len(REGULARIZATION)))
            for i, completion in enumerate(COMPLETION):
                for j, regularization in enumerate(REGULARIZATION):
                    current_time = time.time()

                    transition_matrices = get_matrices(transition_count, state_distance, completion, regularization)
                    correct_rate = evaluation(test_dataset, transition_matrices, state_weightes, rnn_prediction_container)
                    result[i,j] = round(correct_rate*100, 2)  
                    print(f'{completion.__name__} & {regularization.__name__} : {round(correct_rate*100, 2)}%, {time.time() - current_time:.0f}s')

            result = pd.DataFrame(result, columns=regularization_names, index=completion_names)
            print('-'*100)          
            print(f'epoch {epoch+1}: ')
            print(result)
            current_time = time.time()
    
    print(f'Workflow done. Use time:{time.time()-start_time:.1f}')
