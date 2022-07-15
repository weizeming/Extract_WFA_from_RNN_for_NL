import torch

def dev():  
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    return torch.device(device_name)


