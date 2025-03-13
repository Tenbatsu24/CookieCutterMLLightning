import torch


class ScaledNDimFourierAFA(torch.nn.Module):
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
