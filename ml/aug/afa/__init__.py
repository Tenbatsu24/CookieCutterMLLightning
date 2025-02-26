import torch

from einops import rearrange, repeat

from ml.util import STORE, AUG_TYPE


@STORE.reg(AUG_TYPE, "afa")
class AFA(torch.nn.Module):

    def __init__(self, img_size, min_str=5, mean_str=15, spatial_dims=2, fold_d_into_batch=False):
        super().__init__()

        self.spatial_dims = spatial_dims

        _x = torch.linspace(-img_size / 2, img_size / 2, steps=img_size)
        self._x, self._y = torch.meshgrid(_x, _x, indexing="ij")

        self.img_size = img_size
        self.min_str = min_str
        self.mean_str = mean_str

        self.eps_scale = img_size / 32
        self.fold_d_into_batch = fold_d_into_batch

    def forward(self, x):
        init_shape = x.shape
        if len(x.shape) == 1 + self.spatial_dims:
            x = x.unsqueeze(0)

        if self.spatial_dims == 3:
            d = x.shape[-1]

        if self.spatial_dims == 3 and self.fold_d_into_batch:
            x = rearrange(x, "b c h w d -> (b d) c h w")

        b, c, *_ = x.shape

        freqs = 1 - torch.rand((b, c, 1, 1), device=x.device)
        phases = -torch.pi * torch.rand((b, c, 1, 1), device=x.device)
        strengths = torch.empty_like(phases).exponential_(1 / self.mean_str) + self.min_str
        waves = self.gen_planar_waves(freqs, phases, x.device)

        if self.spatial_dims == 3 and (not self.fold_d_into_batch):
            # repeat the waves for each depth
            waves = repeat(waves, "b c h w -> b c h w d", d=d)
            strengths = repeat(strengths, "b c h w -> b c h w d", d=d)

        _temp = torch.clamp(x + strengths * waves, 0, 1)

        if self.spatial_dims == 3 and self.fold_d_into_batch:
            _temp = rearrange(_temp, "(b d) c h w -> b c h w d", d=d)

        return _temp.reshape(init_shape)

    def gen_planar_waves(self, freqs, phases, device):
        _x, _y = self._x.to(device), self._y.to(device)
        _waves = torch.sin(
            2 * torch.pi * freqs * (_x * torch.cos(phases) + _y * torch.sin(phases))
            - torch.rand(1, device=device) * torch.pi
        )
        _waves.div_(_waves.norm(dim=(-2, -1), keepdim=True))

        return self.eps_scale * _waves

    def __str__(self):
        return f"AFA(image_size={self.img_size}, min_str={self.min_str}, mean_str={self.mean_str})"


@STORE.reg(AUG_TYPE, "f_afa")
class FourierAFA(torch.nn.Module):
    """
    FourierAFA applies a Fourier transform to the input tensor and adds a dirac noise at some uniformly
    chosen random coordinate in the fourier domain. The transformed image is then inverse transformed back.

    This is a multi-dimensional version of the AFA, and has smoother noise patterns.
    """

    def __init__(self, min_str=1, mean_str=5):
        super().__init__()
        self.min_str = min_str
        self.mean_str = mean_str

    def forward(self, x):
        init_shape = x.shape

        if len(x.shape) == 3:  # at least C, H, W
            x = x.unsqueeze(0)

        b, c, *_ = x.shape

        x_ft = torch.fft.fftn(x, dim=tuple(range(2, len(x.shape))))

        max_vals = torch.max(x_ft.abs().view(b, c, -1), dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        strengths = max_vals * (
            torch.empty(x_ft.shape, device=x.device).exponential_(1 / self.mean_str) + self.min_str
        )

        noise_map = torch.zeros_like(x)
        noise_idxs = [torch.randint(0, s, (b * c,), device=x.device) for s in x_ft.shape[2:]]
        noise_map[[torch.arange(b).repeat(c), torch.arange(c).repeat(b), *noise_idxs]] = 1

        noise_map = noise_map * strengths
        x_ft += noise_map
        x = torch.fft.ifftn(x_ft, dim=tuple(range(2, len(x.shape)))).real

        return x.reshape(init_shape)

    def __str__(self):
        return f"FourierAFA(min_str={self.min_str}, mean_str={self.mean_str})"
