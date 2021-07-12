import torch

def model_info(net: torch.nn.Module, only_trainable: bool = True):
    """
    Returns:
        int: the total number of parameters used by `net` (only counting shared parameters once);
             if `only_trainable` is True, then only counts parameters with `requires_grad = True`
    """
    net.train()
    parameters = net.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    net.eval()
    return sum(p.numel() for p in unique)
