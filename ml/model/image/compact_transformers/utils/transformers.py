import torch
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init

from ml.layers import MemEffAttention, DropPath


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path_rate=0.1,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        self.self_attn = MemEffAttention(
            dim=d_model, num_heads=nhead, attn_drop=attention_dropout, proj_drop=dropout
        )

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


class TransformerClassifier(Module):
    def __init__(
        self,
        seq_pool=True,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=None,
        dropout=0.1,
        attention_dropout=0.1,
        stochastic_depth=0.1,
        positional_embedding="learnable",
        sequence_length=None,
        patch_latent=False,
        image_latent=False,
    ):
        super().__init__()
        assert not (
            image_latent and patch_latent
        ), "Image latent and patch latent cannot be used at the same time."
        positional_embedding = (
            positional_embedding
            if positional_embedding in ["sine", "learnable", "none"]
            else "sine"
        )
        dim_feedforward = int(embedding_dim * mlp_ratio)

        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0

        self.patch_latent = patch_latent
        self.image_latent = image_latent

        assert sequence_length is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and"
            f" the sequence length was not specified."
        )

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim), requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = Parameter(
                    torch.zeros(1, sequence_length, embedding_dim), requires_grad=True
                )
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False
                )
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(embedding_dim)

        if num_classes is not None:
            self.fc = Linear(embedding_dim, num_classes)
        else:
            self.fc = None

        self.apply(self.init_weight)

    def forward(self, x, return_latent=False):
        if return_latent:
            assert (
                self.image_latent or self.patch_latent
            ), "return_latent=True is only supported when image_latent or patch_latent is True."

        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if not self.patch_latent:
            if self.seq_pool:
                image_latent = torch.matmul(
                    F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x
                ).squeeze(-2)
            else:
                image_latent = x[:, 0]

            if self.fc is None:
                return image_latent
            else:
                out = self.fc(image_latent)
                if return_latent:
                    return out, image_latent
                else:
                    return out
        else:
            if self.seq_pool:
                patch_latent = x
                image_latent = torch.matmul(
                    F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x
                ).squeeze(-2)
            else:
                patch_latent = x[:, 1:]
                image_latent = x[:, 0]

            if self.fc is None:
                return patch_latent
            else:
                out = self.fc(image_latent)
                if return_latent:
                    return out, patch_latent
                else:
                    return out

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor(
            [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)
