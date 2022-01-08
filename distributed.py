import torch
from torch import distributed as D
from torch.utils.data.sampler import Sampler
import math
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_rank():
    if not D.is_available():
        return 0
    if not D.is_initialized():
        return 0
    return D.get_rank()

def synchronize():
    if not D.is_available():
        return 
    if not D.is_initialized():
        return
    world_size = D.get_world_size()

    if world_size == 1:
        return
    D.barrier()

def get_world_size():
    if not D.is_available():
        return 1
    if not D.is_initialized():
        return 1

    return D.get_world_size()

def reduce_sum(tensor):
    if not D.is_available():
        return tensor
    if not D.is_initialized():
        return tensor
    tensor = tensor.clone()
    D.all_reduce(tensor , op=D.ReduceOp.SUM)

def gather_grad(params):
    world_size = get_world_size()

    if world_size == 1:
        return 
    for param in params:
        if param.grad is not None:
            D.all_reduce(param.grad.data , op=D.ReduceOp.SUM)
            param.grad.data.div_(world_size)

def all_gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]
    
    buffer = pickle.dump(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    local_size = torch.IntTensor([tensor.numel()]).to(device)
    size_list = [torch.IntTensor([0]).to(device) for _ in range(world_size)]
    D.all_gather(size_list , local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,))).to(device)
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat([tensor , padding] , 0)
    D.all_gather(tensor_list , tensor)
    data_list = []
    for size , tensor in zip(size_list , tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    
    return data_list


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size == 1:
        return loss_dict
    
    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])
        losses = torch.stack(losses , 0)
        D.reduce(losses , dst=0)

        if D.get_rank() == 0:
            losses /= world_size
        
        reduce_losses = {k: v for k , v in zip(keys , losses)}
    return reduce_losses