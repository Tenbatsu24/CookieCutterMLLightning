from typing import Dict, Tuple

import torch

from torch import Tensor
from torchvision.transforms import TrivialAugmentWide, AugMix


class TrivialAugment(TrivialAugmentWide):

    def __init__(self, all_ops, to_filter_out=None, **kwargs):
        super().__init__(**kwargs)
        self.all_ops = all_ops

        if to_filter_out is None:
            to_filter_out = []

        if not all_ops:
            to_filter_out += ["Brightness", "Color", "Contrast", "Sharpness"]

        self.to_filter_out = to_filter_out

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        original_space = {**super()._augmentation_space(num_bins)}

        if self.all_ops:
            return original_space

        for key in self.to_filter_out:
            original_space.pop(key)

        return original_space


class TrivialMix(AugMix):

    @classmethod
    def _original_space(
        cls, num_bins: int, image_size: Tuple[int, int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, image_size[1] / 3.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, image_size[0] / 3.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 35.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (
                4 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(),
                False,
            ),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def __init__(self, all_ops=True, to_filter_out=None, **kwargs):
        super().__init__(**kwargs)
        self.all_ops = all_ops

        if to_filter_out is None:
            to_filter_out = []

        if not all_ops:
            to_filter_out += ["Brightness", "Color", "Contrast", "Sharpness"]

        self.to_filter_out = to_filter_out

    def _augmentation_space(
        self, num_bins: int, image_size: Tuple[int, int]
    ) -> Dict[str, Tuple[Tensor, bool]]:
        original_space = {**self._original_space(num_bins, image_size)}

        if self.all_ops:
            return original_space

        for key in self.to_filter_out:
            original_space.pop(key)

        return original_space
