import torch

import time

from device import dev
from path import Path, Log_Path
from dataset import dataset

def blank_filling(count_matrix, state_distance):
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if row.sum() != 0:
            matrix[i] = row / row.sum()
    return matrix

def identical_filling(count_matrix, state_distance):
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if row.sum() != 0:
            matrix[i] = row / row.sum()
        else:
            matrix[i,i] = 1
    return matrix

def empirical_filling(count_matrix, state_distance):
    beta = 0.2

    if count_matrix.sum() != 0:
        probs = count_matrix.sum(dim=0)/count_matrix.sum()
        probs *= beta
        matrix = torch.zeros(count_matrix.shape, device=dev())
        for i, row in enumerate(count_matrix):
            if row.sum() != 0:
                matrix[i] = row / row.sum()
            else:
                matrix[i] = probs
                matrix[i,i] += (1 - beta)
        return matrix
    else:
        matrix = torch.zeros(count_matrix.shape, device=dev())
        for i, row in enumerate(count_matrix):
            matrix[i,i] += 1
        return matrix

def near_filling(count_matrix, state_distance):
    beta = 0.2
    ori_matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if row.sum() != 0:
            ori_matrix[i] = row / row.sum()
    
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if ori_matrix[i].sum() != 0:
            matrix[i] = ori_matrix[i].clone()
        else:
            matrix[i] = (ori_matrix.t() * state_distance[i]).t().sum(dim=0)
            matrix[i] /= matrix[i].sum()
            matrix[i] *= beta
            matrix[i,i] += (1 - beta)
    #print(f'Time : {time.time()-start_time:.4f}')
    return matrix

def weighted_filling(count_matrix, state_distance):
    beta = 0.3
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if count_matrix[i].sum() != 0:
            matrix[i] = count_matrix[i].clone()
            matrix[i] /= matrix[i].sum()
        else:
            matrix[i] = (count_matrix.t() * state_distance[i]).t().sum(dim=0)
            matrix[i] /= matrix[i].sum()
            matrix[i] *= beta
            matrix[i,i] += (1 - beta)            
    return matrix

def uniform_filling(count_matrix, state_distance):
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if row.sum() != 0:
            matrix[i] = row / row.sum()
        else:
            matrix[i] = torch.ones(matrix[i].shape, device=dev()) / len(matrix[i])
    return matrix


def none_regularization(matrix, count_matrix, state_distance):
    return matrix

def linear_regularization(matrix, count_matrix, state_distance):
    alpha = 0.2
    for i, row in enumerate(matrix):
        matrix[i] *= 1 - alpha
        matrix[i, i] += alpha
    return matrix

def strong_linear_regularization(matrix, count_matrix, state_distance):
    alpha = 0.4
    for i, row in enumerate(matrix):
        matrix[i] *= 1 - alpha
        matrix[i, i] += alpha
    return matrix

def weighted_regularization(matrix, count_matrix, state_distance):
    alpha = 0.4
    gamma = 1
    count = count_matrix.sum(dim=1)
    for i, row in enumerate(matrix):
        t = torch.exp(-1 * alpha * (count[i] + gamma)).item()
        matrix[i] *= 1 - t
        matrix[i, i] += t
    return matrix

def softmax_regularization(matrix, count_matrix, state_distance):
    return torch.softmax(matrix,dim=1)


def get_matrices(transition_count, state_distance, completion, regularization):
    assert transition_count.shape[1] == state_distance.shape[0]
    transition_matrices = []
    for idx, count_matrix in enumerate(transition_count):
        if count_matrix.sum() == 0:
            matrix = torch.zeros(count_matrix.shape, device=dev())
            for i in range(len(matrix)):
                matrix[i,i] = 1
            transition_matrices.append(matrix)
            continue

        matrix = completion(count_matrix, state_distance)
        matrix = regularization(matrix, count_matrix, state_distance)
        transition_matrices.append(matrix)

    transition_matrices = torch.stack(transition_matrices)
    return transition_matrices.to(dev())
    #print(transition_matrices.shape)