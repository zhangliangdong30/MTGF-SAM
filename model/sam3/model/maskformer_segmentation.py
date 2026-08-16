# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .model_misc import MLP


class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model):
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum(
                    "bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed
                )
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum(
                    "lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )
            else:
                mask_preds = torch.einsum(
                    "lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed
                )

        return mask_preds


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(
                hidden_dim, 1, kernel_size=3, stride=1, padding=1
            )
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _embed_pixels(
        self,
        backbone_feats: List[torch.Tensor],
        image_ids,
        encoder_hidden_states,
    ) -> torch.Tensor:
        # Unwrap NestedTensors to plain tensors if needed (multiplex path)
        from sam3.model.data_misc import NestedTensor

        def _unwrap(x):
            return x.tensors if isinstance(x, NestedTensor) else x

        feature_device = backbone_feats[0].device  # features could be on CPU
        model_device = self.device
        image_ids_ = image_ids.to(feature_device)
        if self.use_encoder_inputs:
            if backbone_feats[0].shape[0] > 1:
                # For bs > 1, we construct the per query backbone features
                backbone_visual_feats = []
                for feat in backbone_feats:
                    # Copy the img features per query (pixel decoder won't share img feats)
                    backbone_visual_feats.append(
                        _unwrap(feat)[image_ids_, ...].to(model_device)
                    )
            else:
                # Bs=1, we rely on broadcasting for query-based processing
                backbone_visual_feats = [
                    _unwrap(bb_feat).clone() for bb_feat in backbone_feats
                ]
            # Extract visual embeddings
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(
                -1, *backbone_feats[-1].shape[1:]
            )

            backbone_visual_feats[-1] = encoder_visual_embed
            if self.act_ckpt:
                pixel_embed = checkpoint.checkpoint(
                    self.pixel_decoder, backbone_visual_feats, use_reentrant=False
                )
            else:
                pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats = [_unwrap(x).to(model_device) for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # For batch_size=1 training, we can avoid the indexing to save memory
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward, mode=compile_mode, dynamic=True, fullgraph=True
            )
            # Needed to make checkpointing happy. But we don't know if the module is checkpointed, so we disable it by default.
            torch._dynamo.config.optimize_ddp = False

    def forward(self, backbone_feats: List[torch.Tensor]):
        # Assumes backbone features are already projected (C == hidden dim)

        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(
                prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode
            )
            if self.shared_conv:
                # only one conv layer
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class DeepSemanticSegHead(nn.Module):
    """Multi-layer conv decode head replacing the single 1x1 semantic head.

    The Jiuzhaigou DEM-SAM inference sweep found w_q~=0 — the SAM3 query-mask
    branch is unused and this semantic head IS the model's output. A single 1x1
    conv is a linear-per-pixel bottleneck; this gives it real capacity (two 3x3
    convs with GroupNorm+GELU, the second dilated for context), analogous to the
    decode head of SegFormer (which outperforms the 1x1-head DEM-SAM).
    """

    def __init__(self, in_dim: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or in_dim

        def _gn(c: int) -> nn.GroupNorm:
            g = 32
            while g > 1 and c % g != 0:
                g //= 2
            return nn.GroupNorm(g, c)

        self.block = nn.Sequential(
            nn.Conv2d(in_dim, h, 3, padding=1, bias=False), _gn(h), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=2, dilation=2, bias=False), _gn(h), nn.GELU(),
            nn.Conv2d(h, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPMSemanticSegHead(nn.Module):
    """PSPNet/UPerNet-style semantic head: a Pyramid Pooling Module for multi-scale
    GLOBAL context on top of the deep conv decode (Scheme A for Jiuzhaigou V4e).

    The FPN pixel-decoder fuses backbone levels but its convs are local; landslides
    span scales and look like background texture locally, so global context helps.
    The PPM pools the pixel embedding at {1,2,3,6}, projects+upsamples each back, and
    concatenates with the input before the deep decode → final 1x1.
    """

    def __init__(self, in_dim: int, pool_scales=(1, 2, 3, 6), hidden: Optional[int] = None):
        super().__init__()
        h = hidden or in_dim

        def _gn(c: int) -> nn.GroupNorm:
            g = 32
            while g > 1 and c % g != 0:
                g //= 2
            return nn.GroupNorm(g, c)

        branch_dim = max(h // 4, 8)
        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, branch_dim, 1, bias=False), _gn(branch_dim), nn.GELU(),
            ) for s in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + branch_dim * len(pool_scales), h, 3, padding=1, bias=False),
            _gn(h), nn.GELU(),
        )
        self.decode = nn.Sequential(
            nn.Conv2d(h, h, 3, padding=1, bias=False), _gn(h), nn.GELU(),
            nn.Conv2d(h, h, 3, padding=2, dilation=2, bias=False), _gn(h), nn.GELU(),
            nn.Conv2d(h, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        feats = [x]
        for branch in self.ppm:
            feats.append(F.interpolate(branch(x), size=(H, W), mode="bilinear", align_corners=False))
        x = self.bottleneck(torch.cat(feats, dim=1))
        return self.decode(x)


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
        dem_decoder_fusion=None,
        deep_seg_head: bool = False,
        ppm_seg_head: bool = False,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, (
                "Specifying a dot product scorer without a presence head is likely a mistake"
            )

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer
                if dot_product_scorer is not None
                else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        if ppm_seg_head:
            self.semantic_seg_head = PPMSemanticSegHead(self.pixel_decoder.out_dim)
        elif deep_seg_head:
            self.semantic_seg_head = DeepSemanticSegHead(self.pixel_decoder.out_dim)
        else:
            self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(
            self.pixel_decoder.out_dim, self.d_model, kernel_size=1
        )
        # Optional DEM-aware refinement of the pixel-decoder output, applied right
        # before the per-pixel semantic / instance heads. With a zero-init residual
        # gate the base SAM3 mask head behaves identically at training step 0.
        self.dem_decoder_fusion = dem_decoder_fusion

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Optional[torch.Tensor]]:
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Optional terrain-aware refinement before the per-pixel heads.
        dem_feat = kwargs.get("dem_feat", None)
        if self.dem_decoder_fusion is not None and dem_feat is not None:
            # Forward path may receive dem_feat as a tensor; slice by image_ids so
            # the per-image DEM map matches the corresponding pixel_embed.
            if dem_feat.shape[0] != pixel_embed.shape[0]:
                dem_feat = dem_feat[image_ids]
            pixel_embed = self.dem_decoder_fusion(pixel_embed, dem_feat)

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }
