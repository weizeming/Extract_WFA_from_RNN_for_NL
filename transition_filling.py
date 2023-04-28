import torch

from utils import *

def blank_filling(count_matrix, state_distance, beta):
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if row.sum() != 0:
            matrix[i] = row / row.sum()
    return matrix

def weighted_filling(count_matrix, state_distance, beta):
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

def uniform_filling(count_matrix, state_distance, beta):
    matrix = torch.zeros(count_matrix.shape, device=dev())
    for i, row in enumerate(count_matrix):
        if row.sum() != 0:
            matrix[i] = row / row.sum()
        else:
            matrix[i] = torch.ones(matrix[i].shape, device=dev()) / len(matrix[i])
    return matrix

