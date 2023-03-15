import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import random
import time

from device import dev
from path import Path, Log_Path
from dataset import dataset
from model import RNN


def get_transitions(model, train_dataset, cluster, augment=False, synonym=0.4, dropout=0.2, num_epoch=5):
    current_time = time.time()
    runtime_data_container = []  
    ''' 
    List of runtime_data. 
    Each runtime_data is a 2-dim tensor as step-wise RNN output for a sentence.
    '''
    
    rnn_prediction_container = []
    '''
    List of RNN final prediction for a whole sentence.
    The index is consist with runtime_data_container.
    '''
    
    transition_num = 0
    int_data = train_dataset.int_data
    if augment:
        current_time = time.time()
        print(f'original size: {int_data.shape}')
        all_synonym = torch.load(Path + train_dataset.type + '_synonym.pth')
        vocab_num = len(all_synonym)
        length = int_data.shape[1]
        new_data = []
        for id, data in enumerate(int_data):
            new_data.append(data)
        for epoch in range(num_epoch):
            for id, data in enumerate(int_data):
                while len(data) > 1 and data[-1] == 0:
                    data = data[0:len(data) - 1]
                for idx, word in enumerate(data):
                    if random.random() < synonym and word < (vocab_num / 5):
                        i = random.randint(1, 4)
                        if all_synonym[word, 0].item() != -1:
                            data[idx] = all_synonym[word, i].item()
                    elif random.random() < dropout:
                        data[idx] = 0
                new_data.append(data)
        print(f'augmented data: {len(new_data)}. Use time: {time.time()-current_time:.1f}')
        int_data = new_data

    for idx, data in enumerate(int_data):
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
        transition_num += len(runtime_data)
        rnn_prediction = torch.argmax(runtime_data[-1])
        runtime_data_container.append(runtime_data)
        rnn_prediction_container.append(rnn_prediction)

    
    
    # generate abstract states
    current_time = time.time()
    
    all_prediction_container = torch.concat(runtime_data_container, dim=0).cpu().numpy()
    '''
    stack runtime_data_container into one tensor to get kmeans clusters.
    '''
    
    kmeans = KMeans(n_clusters=cluster,max_iter=500).fit(all_prediction_container)
    index = 0
    abst_state_container = []
    '''
    The list of abstarct state id, consist with rnn_runtime_container.
    '''
    for runtime_data in runtime_data_container:
        abst_states = [0] # the initial state is label 0 
        for data in runtime_data:
            abst_states.append(kmeans.labels_[index] + 1) # states in kmeans is counted from 0
            index += 1
        abst_state_container.append(abst_states)
    assert index == transition_num
    classes = train_dataset.classes
    state_weightes = np.array([1/classes] * classes).reshape(1,-1)
    state_weightes = np.concatenate((state_weightes, kmeans.cluster_centers_), axis=0)
    state_weightes = torch.from_numpy(state_weightes).to(dev())
    state_weightes = state_weightes.to(torch.float32)

    # generate transitions

    vocab_num = len(train_dataset.vocab)
    state_num = cluster + 1
    transition_count = torch.zeros((vocab_num, state_num, state_num), device=dev())
    '''
    transition_count maintains the transition numbers in all training data, 
    including original train set and synonym based augmenting data.
    the [a,p,q] item implies the count of transitions that
    from state p, reading word a then transform to state q.
    '''
    for idx, data in enumerate(int_data):
        # remove 0 at the end
        while len(data) > 1 and data[-1] == 0:
            data = data[0:len(data)-1]
        assert len(data)+1 == len(abst_state_container[idx])
        for offset,word in enumerate(data):
            p = abst_state_container[idx][offset]
            a = word
            q = abst_state_container[idx][offset+1]
            transition_count[a, p, q] += 1
            
    return transition_count, kmeans, state_weightes, all_prediction_container

def add_transitions(model, all_data, matrices, kmeans):
    delta = 0.5
    runtime_data_container = []  
    
    for idx, data in enumerate(all_data):
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
        runtime_data_container.append(runtime_data)
    
    all_prediction_container = torch.concat(runtime_data_container, dim=0).cpu().numpy()
    states_id = kmeans.predict(all_prediction_container)
    index = 0
    abst_state_container = []


    for runtime_data in runtime_data_container:
        abst_states = [0]
        for data in runtime_data:
            abst_states.append(states_id[index] + 1)
            index += 1
        abst_state_container.append(abst_states)
    assert index == len(states_id)

    for idx, data in enumerate(all_data):
        assert len(data)+1 == len(abst_state_container[idx])
        for offset,word in enumerate(data):
            p = abst_state_container[idx][offset]
            a = word
            q = abst_state_container[idx][offset+1]
            matrices[a, p, q] += delta
    
    return matrices
