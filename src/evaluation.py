
import torch
from time import time
from device import dev
from path import Path, Log_Path
from dataset import dataset

def evaluation(dataset, transition_matrices, state_weight, rnn_prediction):
    assert len(dataset.int_data) == len(rnn_prediction)
    assert transition_matrices.shape[1] == state_weight.shape[0]
    assert len(dataset.vocab) == transition_matrices.shape[0]
    assert dataset.classes == state_weight.shape[1]


    consist = 0
    state_number = state_weight.shape[0]

    initial_vector = torch.zeros((1, state_number), device=dev())
    initial_vector[0,0] = 1

    for idx, data in enumerate(dataset.int_data):
        # remove 0 at the end
        while len(data) > 1 and data[-1] == 0:
            data = data[0:len(data)-1]
        
        state_probs = initial_vector.clone().to(dev())

        for word in data:
            state_probs = state_probs @ transition_matrices[word]
            #print(state_probs)
        #print(state_probs)
        prediction = state_probs @ state_weight
        prediction = prediction.flatten()
        if prediction.sum() == 0:
            wfa_prediction = -1
        else:
            wfa_prediction =  torch.argmax(prediction).item()
        if wfa_prediction == rnn_prediction[idx]:
            consist += 1
    correct_rate = consist/len(dataset.int_data)
    
    return correct_rate    
