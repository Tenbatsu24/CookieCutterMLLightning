# Adapted from: https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py
# References:
#   https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import math

from functools import partial
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint

from loguru import logger
from torch.nn.functional import interpolate
from torch.nn.init import trunc_normal_

from ml.layers import (
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
    MemEffAttention,
    NestedTensorBlock as Block,
    DINOHead,
)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x, return_attention=False):
        # Adaptation for returing attentions
        for i, b in enumerate(self):
            if i < len(self) - 1:
                x = b(x)
            else:
                return b(x, return_attention=return_attention)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=518,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        # ADDED PARAMETERS
        bottleneck_dim=256,
        hidden_dim=2048,
        n_layers_projection_head=3,
        l2_norm=True,
        prototypes=65536,
        finegrained=False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
            interpolate_offset: (float) work-around offset to apply when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.l2_norm = l2_norm
        self.finegrained = finegrained

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = DINOHead(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            out_dim=prototypes,
            n_layers=max(n_layers_projection_head, 1),
            mlp_bias=True,
            l2_normalize=l2_norm,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Initialize the model's weights
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            # antialias=self.interpolate_antialias,
        )

        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x

    def forward_features_list(self, x_list, masks_list):
        if masks_list is None:
            masks_list = [None for _ in range(len(x_list))]

        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]

        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None, last_self_attention=False):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x = blk(x, return_attention=last_self_attention)

        attn = None
        if last_self_attention:
            x, attn = x
            # Attention is selected from the cls token to the patch tokens only
            # Thus, we ignore the cls from the patch tokens (i.e., start from 1)
            attn = attn[:, :, 0, 1:]

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "last_self_attention": attn,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, inputs, masks=None):
        bs = inputs.shape[0] if isinstance(inputs, torch.Tensor) else inputs[0].shape[0]

        was_list = isinstance(inputs, list)
        list_out_dict: list | dict = self.forward_features(
            inputs, masks=masks, last_self_attention=not was_list
        )

        if was_list:
            image_latents = [out_dict["x_norm_clstoken"] for out_dict in list_out_dict]

            if self.finegrained:
                patch_latents = [
                    out_dict["x_norm_patchtokens"].reshape(-1, self.embed_dim)
                    for out_dict in list_out_dict
                ]
                list_dict_proj_proto_patch = [
                    self.head(patch_latent) for patch_latent in patch_latents
                ]
                patch_projs, patch_logits = zip(
                    *[
                        (dict_proj_proto["proj"], dict_proj_proto["logits"])
                        for dict_proj_proto in list_dict_proj_proto_patch
                    ]
                )

                out = {
                    "latent": [
                        torch.stack(torch.chunk(cls_token, bs)).squeeze(1)
                        for cls_token in image_latents
                    ],
                    "patch_latent": [
                        torch.stack(torch.chunk(latent, bs)) for latent in patch_latents
                    ],
                    "proj": [torch.stack(torch.chunk(proj, bs)) for proj in patch_projs],
                    "logits": [torch.stack(torch.chunk(logit, bs)) for logit in patch_logits],
                }
            else:
                list_dict_proj_proto_cls = [
                    self.head(image_latent) for image_latent in image_latents
                ]
                cls_projs, cls_logits = zip(
                    *[
                        (dict_proj_proto["proj"], dict_proj_proto["logits"])
                        for dict_proj_proto in list_dict_proj_proto_cls
                    ]
                )

                out = {
                    "latent": [
                        torch.stack(torch.chunk(cls_token, bs)).squeeze(1)
                        for cls_token in image_latents
                    ],
                    "proj": [torch.stack(torch.chunk(proj, bs)).squeeze(1) for proj in cls_projs],
                    "logits": [
                        torch.stack(torch.chunk(logit, bs)).squeeze(1) for logit in cls_logits
                    ],
                }
        else:
            attentions = list_out_dict["last_self_attention"]
            image_latents = list_out_dict["x_norm_clstoken"]

            if self.finegrained:
                patch_latents = list_out_dict["x_norm_patchtokens"].reshape(-1, self.embed_dim)
                proj_logits_patch = self.head(patch_latents)
                patch_projs, patch_logits = proj_logits_patch["proj"], proj_logits_patch["logits"]
                out = {
                    "latent": torch.stack(torch.chunk(image_latents, bs)).squeeze(1),
                    "patch_latent": torch.stack(torch.chunk(patch_latents, bs)),
                    "proj": torch.stack(torch.chunk(patch_projs, bs)),
                    "logits": torch.stack(torch.chunk(patch_logits, bs)),
                    "attentions": attentions,
                }
            else:
                proj_logits_cls = self.head(image_latents)
                proj_cls, logits_cls = proj_logits_cls["proj"], proj_logits_cls["logits"]

                out = {
                    "latent": torch.stack(torch.chunk(image_latents, bs)).squeeze(1),
                    "proj": torch.stack(torch.chunk(proj_cls, bs)).squeeze(1),
                    "logits": torch.stack(torch.chunk(logits_cls, bs)).squeeze(1),
                    "attentions": attentions,
                }

        return out

    def get_last_selfattention(self, x, masks=None):
        """
        Adapted from https://gitlab.com/ziegleto-machine-learning/dino/-/tree/main/
        """
        if isinstance(x, list):
            raise NotImplementedError("Not implemented for list of inputs")
            # return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        # Run through model, at the last block just return the attention.
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                break

        _, attn = self.blocks[-1](x, return_attention=True)

        return attn


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        init_values=1.0,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        init_values=1.0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        init_values=1.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


def vit_giant(patch_size=16, num_register_tokens=0, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        init_values=1.0,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model


if __name__ == "__main__":
    # Example usage
    for finegrained in [False, True]:
        print()
        print(f"Testing vit_base with finegrained={finegrained}")

        model = vit_base(patch_size=16, num_register_tokens=0, finegrained=finegrained).to(
            "cuda"
        )  # Move model to GPU
        _x = torch.randn(
            2, 3, 224, 224, dtype=torch.float32, device="cuda"
        )  # Example input tensor
        _output = model(_x)
        for key, value in _output.items():
            print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")

        _xs = [
            torch.randn(2, 3, 224, 224, dtype=torch.float32, device="cuda"),
            torch.randn(2, 3, 96, 96, dtype=torch.float32, device="cuda"),
        ]  # Example input tensor
        _outputs = model(_xs)

        print(len(_outputs))

        for key, value in _outputs.items():
            if isinstance(value, list) or isinstance(value, tuple):
                print(f"{key}: {[v.shape for v in value]}")
            else:
                print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
        print()
