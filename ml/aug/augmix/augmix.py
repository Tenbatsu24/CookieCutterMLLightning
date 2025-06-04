import torch

from kornia.augmentation import (
    RandomAffine,
    RandomPosterize,
    RandomSolarize,
    RandomAutoContrast,
    RandomEqualize,
    RandomBrightness,
    RandomSaturation,
    RandomContrast,
    RandomSharpness,
)

from ..prime import PRIMEAugModule, GeneralizedPRIMEModule


class GPUAugMix(torch.nn.Module):
    r"""AugMix data augmentation method based on
    `"AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty" <https://arxiv.org/abs/1912.02781>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        severity (int): The severity of base augmentation operators. Default is ``3``.
        mixture_width (int): The number of augmentation chains. Default is ``3``.
        chain_depth (int): The depth of augmentation chains. A negative value denotes stochastic depth sampled from the interval [1, 3].
            Default is ``-1``.
        alpha (float): The hyperparameter for the probability distributions. Default is ``1.0``.
        all_ops (bool): Use all operations (including brightness, contrast, color and sharpness). Default is ``True``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        severity: int = 3,
        mixture_width: int = 3,
        chain_depth: int = -1,
        max_depth: int = 3,
        spatial=True,
        color=True,
        all_ops: bool = True,
    ) -> None:
        super().__init__()
        self._PARAMETER_MAX = 10
        if not (1 <= severity <= self._PARAMETER_MAX):
            raise ValueError(
                f"The severity must be between [1, {self._PARAMETER_MAX}]. Got {severity} instead."
            )
        self.severity = severity
        self.mixture_width = mixture_width
        self.chain_depth = chain_depth

        self.spatial = spatial
        self.color = color
        self.all_ops = all_ops

        self.augmentations = self._augmentation_space()

        self._aug = GeneralizedPRIMEModule(
            aug_module=PRIMEAugModule(list(self.augmentations.values())),
            mixture_width=self.mixture_width,
            mixture_depth=self.chain_depth,
            max_depth=max_depth,
        )

    def _augmentation_space(self):
        factor = self.severity / self._PARAMETER_MAX

        augmentations = dict()

        if self.spatial:
            augmentations.update(
                {
                    "ShearX": RandomAffine(
                        degrees=0,
                        translate=None,
                        scale=None,
                        shear=0.3 * factor,
                        p=1.0,
                        resample=1,
                        same_on_batch=False,
                    ),
                    "ShearY": RandomAffine(
                        degrees=0,
                        translate=None,
                        scale=None,
                        shear=(0, 0, -0.3 * factor, 0.3 * factor),
                        p=1.0,
                        resample=1,
                        same_on_batch=False,
                    ),
                    "TranslateX": RandomAffine(
                        degrees=0,
                        translate=(factor / 3.0, 0),
                        scale=None,
                        shear=None,
                        p=1.0,
                        resample=1,
                        same_on_batch=False,
                    ),
                    "TranslateY": RandomAffine(
                        degrees=0,
                        translate=(0, factor / 3.0),
                        scale=None,
                        shear=None,
                        p=1.0,
                        resample=1,
                        same_on_batch=False,
                    ),
                    "Rotate": RandomAffine(
                        degrees=30 * factor,
                        translate=None,
                        scale=None,
                        shear=None,
                        p=1.0,
                        resample=1,
                        same_on_batch=False,
                    ),
                }
            )

        if self.color:
            augmentations.update(
                {
                    "Posterize": RandomPosterize(
                        bits=(4 - round((self.severity - 1) / ((10 - 1) / 4))),
                        p=1.0,
                        same_on_batch=False,
                    ),
                    "Solarize": RandomSolarize(
                        thresholds=(1.0 - factor, 1.0), p=1.0, same_on_batch=False
                    ),
                    "AutoContrast": RandomAutoContrast(p=1.0, same_on_batch=False),
                    "Equalize": RandomEqualize(p=1.0, same_on_batch=False),
                }
            )

        if self.all_ops:
            augmentations.update(
                {
                    "Brightness": RandomBrightness(
                        brightness=(0.0, 0.9 * factor), p=1.0, same_on_batch=False
                    ),
                    "Color": RandomSaturation(
                        saturation=(0.0, 0.9 * factor), p=1.0, same_on_batch=False
                    ),
                    "Contrast": RandomContrast(
                        contrast=(0.0, 0.9 * factor), p=1.0, same_on_batch=False
                    ),
                    "Sharpness": RandomSharpness(
                        sharpness=(0.0, 0.9 * factor), p=1.0, same_on_batch=False
                    ),
                }
            )

        return augmentations

    @torch.no_grad()
    def forward(self, img):
        return self._aug(img)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"severity={self.severity}, mixture_width={self.mixture_width}, "
            f"chain_depth={self.chain_depth} , max_depth={self.max_depth},"
            f"spatial={self.spatial}, color={self.color}, all_ops={self.all_ops},\n"
            f"augmentations={self.augmentations.keys()}\n"
            f")"
        )
        return s

    def __str__(self):
        return self.__repr__()
