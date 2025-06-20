from .utils.tokenizer import Tokenizer
from .utils.base import BaseCompactTransformer
from .utils.transformers import TransformerClassifier


class ViTLite(BaseCompactTransformer):
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        in_channels=3,
        kernel_size=16,
        dropout=0.0,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        num_layers=14,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=1000,
        positional_embedding="learnable",
        *args,
        **kwargs,
    ):
        super(ViTLite, self).__init__()
        assert img_size % kernel_size == 0, (
            f"Image size ({img_size}) has to be" f"divisible by patch size ({kernel_size})"
        )
        self.tokenizer = Tokenizer(
            n_input_channels=in_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
            max_pool=False,
            activation=None,
            n_conv_layers=1,
            conv_bias=True,
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=in_channels, height=img_size, width=img_size
            ),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
        )


def _vit_lite(
    arch,
    pretrained,
    progress,
    num_layers,
    num_heads,
    mlp_ratio,
    embedding_dim,
    positional_embedding="learnable",
    kernel_size=4,
    *args,
    **kwargs,
):
    model = ViTLite(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        positional_embedding=positional_embedding,
        *args,
        **kwargs,
    )

    assert not pretrained, "Pretrained models not supported for ViT Lite"

    return model


def vit_2(*args, **kwargs):
    return _vit_lite(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128, *args, **kwargs)


def vit_4(*args, **kwargs):
    return _vit_lite(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128, *args, **kwargs)


def vit_6(*args, **kwargs):
    return _vit_lite(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256, *args, **kwargs)


def vit_7(*args, **kwargs):
    return _vit_lite(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, *args, **kwargs)


def vit_lite_7_4_32(
    pretrained=False,
    progress=False,
    img_size=32,
    positional_embedding="learnable",
    num_classes=10,
    *args,
    **kwargs,
):
    return vit_7(
        "vit_7_4_32",
        pretrained,
        progress,
        kernel_size=4,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        *args,
        **kwargs,
    )
