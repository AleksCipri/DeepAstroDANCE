import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy import arcsinh as arcsinh


from typing import Tuple
from torch import Tensor
import torch

def update_sinh(t):
    global_min = np.percentile(t, 0.1)
    global_max = np.percentile(t, 99.9)

    for i in range(0,3):
        c = .85/global_max #gets you close to arcsinh(max_x) = 1, arcsinh(min_x) = 0
        t[:,i] = np.clip(t[:,i], global_min, global_max)
        t[:,i] = arcsinh(c*t[:, i])
        t[:,i] = (t[:,i] + 1.0) / 2.0


def prepare_images(array_path_dict): # -> Tensor

    #UPDATE TODO:
        # Need to differentiate between source, target, test data

    #load numpy data arrays
    source_data = list(np.load(array_path_dict['x_train_y10']))
    target_data = list(np.load(array_path_dict['x_train_y1']))

    truth_label = list(np.load(array_path_dict['truth_label']))

    eval_data = list(np.load(array_path_dict['x_val_y1']))
    eval_label = list(np.load(array_path_dict['eval_label']))

    #normalize x_data
    update_sinh(source_data)
    update_sinh(target_data)
    update_sinh(eval_data)

    #transform the arrays into tensors
    source_tensor = torch.Tensor(source_data)
    target_tensor = torch.Tensor(target_data)

    truth_label = torch.Tensor(truth_label)

    eval_tensor = torch.Tensor(eval_data)
    eval_label_tensor = torch.Tensor(eval_label)
    
    return source_tensor, truth_label, target_tensor, eval_tensor, eval_label_tensor

#  create Dataset class
class AdversariesDataset(data.Dataset[Tuple[Tensor, ...]]):
    
    def __init__(self, *tensors: Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self,index):
        items = (tensor[index] for tensor in self.tensors)
        items.append(index)
        return tuple(items)
    
    def __len__(self):
        return self.tensors[0].size(0)
    
def AdversaryDataLoader(array_paths, batch_size, return_id=False, balanced=False):

    source_data, truth_label, target_data, eval_data, eval_label = prepare_images(array_paths)

    source_loader = torch.utils.data.Dataloader(AdversariesDataset(source_data,truth_label))
    target_loader = torch.utils.data.Dataloader(AdversariesDataset(target_data, truth_label))
    eval_loader = torch.utils.data.Dataloader(AdversariesDataset(eval_data, eval_label))

    return source_loader, target_loader, eval_loader
