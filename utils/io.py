"""
Utility for saving and loading

@yuningw
"""

import torch 

def save_checkpoint(state, path_name):

    torch.save(state, path_name)
    print('Saved checkpoint')


def load_checkpoint(model, path_name, optimizer=None):

    print('Loading checkpoint')
    print(path_name)

    checkpoint = torch.load(path_name)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_dict'])

    print('Loaded checkpoint')

