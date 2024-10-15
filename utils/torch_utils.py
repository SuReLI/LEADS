import torch

def one_hot_encode(batch_indexes, n):
    """
    one hot encoding
    """
    # print(batch_indexes)
    print(batch_indexes.shape)
    batch_size = batch_indexes.size
    print(batch_size)
    one_hot = torch.zeros(batch_size, n)
    one_hot.scatter_(1, batch_indexes, 1)
    return one_hot

