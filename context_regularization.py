import torch

def none_regularization(matrix, count_matrix, state_distance, alpha):
    return matrix

def linear_regularization(matrix, count_matrix, state_distance, alpha):
    for i, row in enumerate(matrix):
        matrix[i] *= 1 - alpha
        matrix[i, i] += alpha
    return matrix
