import torch

from .mixup import RandomMixUp
from .cutmix import RandomCutMix


class CutMixUp(torch.nn.Module):

    def __init__(self, mixup_alpha=0.0, cutmix_alpha=0.0, p=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.mixup = RandomMixUp(p=1.0, alpha=mixup_alpha, inplace=True)
        self.cutmix = RandomCutMix(p=1.0, alpha=cutmix_alpha, inplace=True)

    def __call__(self, batch, target):
        # Apply mixup and cutmix
        if torch.rand(1).item() < self.p:
            return self.mixup(batch, target)
        else:
            return self.cutmix(batch, target)


def make_cutmix_mixup(cfg):
    if cfg.dataset.name in ["in", "in100"]:
        if "convnext" in cfg.model.type:
            mix_up_alpha = 0.8
        else:
            mix_up_alpha = 0.2
    else:
        if "swin" in cfg.model.type or "cct" in cfg.model.type:
            mix_up_alpha = 0.8
        else:
            mix_up_alpha = 1.0

    # check if config has mix_params and if so, override the mixup_alpha and cutmix_alpha if provided
    if hasattr(cfg, "mix_params"):
        mix_up_alpha = cfg.mix_params.get("mixup_alpha", mix_up_alpha)
        cutmix_alpha = cfg.mix_params.get("cutmix_alpha", 1.0)
    else:
        mix_up_alpha = mix_up_alpha
        cutmix_alpha = 1.0

    return CutMixUp(
        mixup_alpha=mix_up_alpha, cutmix_alpha=cutmix_alpha, num_categories=cfg.num_classes
    )
