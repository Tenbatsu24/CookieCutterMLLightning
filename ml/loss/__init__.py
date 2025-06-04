import torch


def get_loss(loss_name):
    if hasattr(torch.nn, loss_name):
        return getattr(torch.nn, loss_name)
    else:
        raise ValueError(f"Loss {loss_name} not found in torch.nn")
