import torch


def get_loss(loss_name, loss_params):
    if hasattr(torch.nn, loss_name):
        return getattr(torch.nn, loss_name)(**loss_params)
    else:
        raise ValueError(f"Loss {loss_name} not found in torch.nn or smp.losses")
