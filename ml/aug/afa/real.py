import torch


def pad_dims(tensor, n_dims):
    for _ in range(n_dims):
        tensor = tensor.unsqueeze(-1)
    return tensor


def possibly_not_list(x):
    return x if isinstance(x, list) else [x]


class RealNDimFourier(torch.nn.Module):

    def __init__(self, spatial_dims, min_str=1, mean_str=5, clamp=True, clamp_range=(0, 1)):
        super().__init__()
        self.min_str = min_str
        self.mean_str = mean_str
        self.spatial_dims = spatial_dims

        self.clamp = clamp
        self.clamp_range = clamp_range

    @torch.no_grad()
    def forward(self, x):
        init_shape = x.shape

        if len(x.shape) == 0:
            raise ValueError("Input tensor must have at least one dimension.")

        if len(x.shape) == self.spatial_dims:  # at least has spatial dimensions
            x = x.unsqueeze(0)  # add channel dimension
        elif len(x.shape) < self.spatial_dims:
            raise ValueError(
                f"Input tensor must have at least {self.spatial_dims} spatial dimensions."
            )

        if len(x.shape) == self.spatial_dims + 1:  # at least has C, H, W
            x = x.unsqueeze(0)  # add batch dimension
        elif len(x.shape) < self.spatial_dims + 1:
            raise ValueError(
                f"Input tensor must have at least {self.spatial_dims} spatial dimensions and one channel dimensions."
            )

        b, c, *_ = x.shape

        # make an all empty fourier spectrum with the same shape as x
        n_ft = torch.zeros_like(x)

        # for each b, c add a random dirac noise at a random location
        mag_map = torch.zeros_like(x)
        phase_map = torch.zeros_like(x)

        noise_idxs = [torch.randint(0, s, (b * c,), device=x.device) for s in n_ft.shape[2:]]
        mag_map[[torch.arange(b).repeat(c), torch.arange(c).repeat(b), *noise_idxs]] = 1
        phase_map[[torch.arange(b).repeat(c), torch.arange(c).repeat(b), *noise_idxs]] = (
            2 * torch.pi * torch.rand(b * c, device=x.device)
        )

        # create the noise map
        noise_map = torch.complex(mag_map, phase_map)

        # make it symmetric so that the noise is real. We do not know how many dimensions the input has
        noise_map = noise_map + noise_map.flip(tuple(range(2, len(x.shape))))

        # take the inverse fourier transform
        w_ft = torch.fft.ifftn(noise_map, dim=tuple(range(2, len(x.shape)))).real

        # take l2 norm of the noise
        w_ft = w_ft / w_ft.norm(dim=tuple(range(2, len(x.shape))), keepdim=True, p=2)

        # calculate the strength of the noise
        strengths = (
            torch.empty(
                (
                    b,
                    c,
                ),
                device=x.device,
            ).exponential_(1 / self.mean_str)
            + self.min_str
        )

        # add dims of size 1 to make it broadcastable
        strengths = pad_dims(strengths, len(x.shape) - 2)

        # scaling
        scale = max(*possibly_not_list(x.shape[slice(2, 2 + self.spatial_dims)])) / 32

        # add the noise to the image
        x = x + scale * strengths * w_ft

        if self.clamp:
            x = torch.clamp(x, *self.clamp_range)

        return x.reshape(init_shape)

    def __str__(self):
        return f"RealFourier(min_str={self.min_str}, mean_str={self.mean_str})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # create a random image
    img = torch.zeros(3, 256, 256)

    # apply the RealFourier transform
    rft = RealNDimFourier(2, min_str=0, mean_str=10, clamp=True)
    img_rft = rft(img)

    # plot the original and transformed image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.permute(1, 2, 0))
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(img_rft.permute(1, 2, 0))
    ax[1].set_title("RealFourier")
    ax[1].axis("off")
    plt.show()

    one_d_signal = torch.zeros(256)
    rft = RealNDimFourier(1, min_str=0, mean_str=1, clamp=False)
    one_d_signal_rft = rft(one_d_signal)

    _, ax = plt.subplots(1, 2)
    ax[0].plot(one_d_signal.squeeze())
    ax[0].set_title("Original")
    ax[1].plot(one_d_signal_rft.squeeze())
    ax[1].set_title("RealFourier")
    plt.show()

    rft = RealNDimFourier(3, min_str=5, mean_str=15, clamp=False)
    img = torch.zeros(3, 256, 256, 256)
    img_rft = rft(img)

    print(img_rft.shape)
    # plot one slice along each spatial dimension

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(img_rft[0, :, :, 128])
    ax[0].set_title("Slice 1")
    ax[0].axis("off")
    ax[1].imshow(img_rft[0, :, 128, :])
    ax[1].set_title("Slice 2")
    ax[1].axis("off")
    ax[2].imshow(img_rft[0, 128, :, :])
    ax[2].set_title("Slice 3")
    ax[2].axis("off")
    plt.show()
