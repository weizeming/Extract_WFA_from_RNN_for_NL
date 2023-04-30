import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 4)
import time
import random
import argparse
from tqdm import tqdm
from path import Path
from dataset import dataset
from model import RNN

from utils import *
from transition_filling import *
from context_regularization import *
from synonym import get_synonym

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='news', choices=['toxic', 'news'])
    parser.add_argument('--cluster', default=40, type=int)


    # transition matrix complement
    parser.add_argument('--beta', default=0.3, type=float)
    # context regularization
    parser.add_argument('--alpha', default=0.4, type=float)

    # configurations for data augmentation
    parser.add_argument('--augmentation-epochs', default=5, type=int)
    parser.add_argument('--replace-rate', default=0.4, type=float)
    parser.add_argument('--dropout', default=0.2, type=float)

    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--JS', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    DATASET = args.dataset
    CLUSTER = args.cluster

    BETA = args.beta
    ALPHA = args.alpha

    # select the completion and regularization tactics
    if args.ablation:
        COMPLETION = [weighted_filling, uniform_filling]
        REGULARIZATION = [linear_regularization, none_regularization]
        if not args.JS:
            COMPLETION.append(blank_filling)
    else:
        COMPLETION = [weighted_filling]
        REGULARIZATION = [linear_regularization]
        #COMPLETION = [blank_filling]
        #REGULARIZATION = [none_regularization]        
    # select the iteration times of using synonym to augmenting dataset
    NUM_EPOCHS = args.augmentation_epochs

    REPLACE_RATE = args.replace_rate
    DROPOUT = args.dropout

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
    for idx, data in enumerate(tqdm(test_dataset.int_data)):
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
        rnn_prediction = torch.argmax(runtime_data[-1]) if not args.JS else runtime_data[-1]
        rnn_prediction_container.append(rnn_prediction)

    transition_count, kmeans, state_weightes, all_prediction_container = get_transitions(model, train_dataset, CLUSTER)
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

            transition_matrices = get_matrices(transition_count, state_distance, completion, regularization, BETA, ALPHA)
            correct_rate = evaluation(test_dataset, transition_matrices, state_weightes, rnn_prediction_container, args.JS)
            result[i,j] = round(correct_rate*100, 2)
            print(f'{completion.__name__} & {regularization.__name__} : {round(correct_rate, 4)}, {time.time() - current_time:.0f}s')

    result = pd.DataFrame(result, columns=regularization_names, index=completion_names)
    print(result)
    print(f'Evaluation done.')
    
    all_synonym = torch.load(Path+DATASET+'_synonym.pth')
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

                    transition_matrices = get_matrices(transition_count, state_distance, completion, regularization,BETA,ALPHA)
                    correct_rate = evaluation(test_dataset, transition_matrices, state_weightes, rnn_prediction_container, args.JS)
                    result[i,j] = round(correct_rate*100, 2)  
                    print(f'{completion.__name__} & {regularization.__name__} : {round(correct_rate, 4)}, {time.time() - current_time:.0f}s')

            result = pd.DataFrame(result, columns=regularization_names, index=completion_names)
            print('-'*100)          
            print(f'epoch {epoch+1}: ')
            print(result)
            current_time = time.time()
    
    print(f'Workflow done. Use time:{time.time()-start_time:.1f}')
