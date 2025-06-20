import torch.nn as nn

from .utils.tokenizer import Tokenizer
from .utils.base import BaseCompactTransformer
from .utils.transformers import TransformerClassifier


class CCT(BaseCompactTransformer):
    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        in_channels=3,
        kernel_size=7,
        dropout=0.0,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        num_layers=14,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=1000,
        positional_embedding="learnable",
        n_conv_layers=1,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        *args,
        **kwargs,
    ):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(
            n_input_channels=in_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False,
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(
                n_channels=in_channels, height=img_size, width=img_size
            ),
            embedding_dim=embedding_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            **kwargs,
        )


def _cct(
    num_layers,
    num_heads,
    mlp_ratio,
    embedding_dim,
    kernel_size=3,
    stride=None,
    padding=None,
    *args,
    **kwargs,
):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CCT(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        *args,
        **kwargs,
    )
    return model


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256, *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384, *args, **kwargs)


def cct_7_3x1_32(img_size=32, positional_embedding="learnable", num_classes=10, *args, **kwargs):
    return cct_7(
        *args,
        kernel_size=3,
        n_conv_layers=1,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        **kwargs,
    )


def cct_14_7x2_224(
    img_size=224, positional_embedding="learnable", num_classes=1000, *args, **kwargs
):
    return cct_14(
        kernel_size=7,
        n_conv_layers=2,
        img_size=img_size,
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        *args,
        **kwargs,
    )
