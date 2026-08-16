# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    # Pick the largest power-of-2 group count that divides `channels`, capped at max_groups.
    # GroupNorm is preferable to BatchNorm here because SAM3 fine-tunes typically run with
    # small per-GPU batch sizes at high resolution, where BN running statistics are noisy.
    groups = max_groups
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class _SERefine(nn.Module):
    """Depthwise-conv + channel squeeze-excitation residual block (SAM-DEM v3).

    Produces a *refinement* tensor of the same shape as the input, meant to be
    consumed as a gated residual by the caller (``x + tanh(gate) * refine(x)``).
    A zero-initialised gate therefore makes this a strict no-op at training step 0,
    which is what lets v3 modules load ``strict=False`` from a v5.26V2 checkpoint
    without perturbing the warm-started weights.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 8)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.norm = _gn(channels)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, 1)
        self.fc1 = nn.Conv2d(channels, bottleneck, 1)
        self.fc2 = nn.Conv2d(bottleneck, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.act(self.norm(self.dw(x))))
        s = F.adaptive_avg_pool2d(y, output_size=1)
        s = torch.sigmoid(self.fc2(self.act(self.fc1(s))))
        return y * s


class DEMEncoder(nn.Module):
    """Lightweight terrain encoder for DEM-derived channels.

    Expected input is a tensor shaped [B, C, H, W], typically containing
    normalized elevation, slope, and aspect-derived channels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        out_channels: int = 256,
        num_scales: int = 3,
        num_stages: Optional[int] = None,
        v3_refine: bool = False,
    ) -> None:
        """num_stages overrides the legacy 3-stage default; with num_stages=4 the
        encoder gains an extra stride-2 stage (overall stride 2^k for k=1..N),
        giving SAM3-DEM v5.26V2 a deeper terrain feature pyramid at no change
        to the cross-attention / decoder-fusion interfaces (the [0]-th feature
        still feeds DEMDecoderFusion, and the trailing N=len(target_sizes)
        features feed DEMCrossAttentionFusion)."""
        super().__init__()
        self.out_channels = out_channels
        self.num_scales = num_scales
        if num_stages is None:
            num_stages = 3
        self.num_stages = num_stages

        # Channel widths grow by a factor of 2 per stage and cap at 4 × hidden.
        channels = [
            min(hidden_channels * (2 ** i), hidden_channels * 4)
            for i in range(num_stages)
        ]
        layers = []
        current_in = in_channels
        for current_out in channels:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_in,
                        current_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    _gn(current_out),
                    nn.GELU(),
                    nn.Conv2d(
                        current_out,
                        current_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    _gn(current_out),
                    nn.GELU(),
                )
            )
            current_in = current_out

        self.stages = nn.ModuleList(layers)
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1) for ch in channels]
        )

        # SAM-DEM v3: per-scale gated SE refinement of the projected terrain
        # features. New params (absent from v5.26V2 checkpoints) with zero-init
        # gates → strict=False load + no-op at step 0.
        self.v3_refine = v3_refine
        if v3_refine:
            self.refine = nn.ModuleList(
                [_SERefine(out_channels) for _ in channels]
            )
            self.refine_gates = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in channels]
            )

    def forward(
        self,
        dem: torch.Tensor,
        target_sizes: Optional[List[torch.Size]] = None,
        return_global: bool = False,
        return_highres: bool = False,
    ) -> List[torch.Tensor]:
        if dem.dim() == 3:
            dem = dem.unsqueeze(0)
        if dem.dim() != 4:
            raise ValueError(f"DEM tensor must be [B, C, H, W], got {dem.shape}")
        expected_channels = self.stages[0][0].in_channels
        if dem.shape[1] != expected_channels:
            raise ValueError(
                f"DEM tensor has {dem.shape[1]} channels, but this DEMEncoder "
                f"expects {expected_channels}. Set dem_in_channels accordingly."
            )

        x = dem
        features = []
        for i, (stage, projection) in enumerate(zip(self.stages, self.projections)):
            x = stage(x)
            proj = projection(x)
            if self.v3_refine:
                proj = proj + torch.tanh(self.refine_gates[i]) * self.refine[i](proj)
            features.append(proj)

        # The HIGHEST-resolution stage output, irrespective of how many later
        # stages are sliced off by `target_sizes`. v5.26V2 uses this to give
        # DEMDecoderFusion a stride-2 (from input) feature even when the
        # cross-attention is aligned to the deeper SAM3 FPN levels.
        highres_feature = features[0]

        if target_sizes is None:
            selected = features[-self.num_scales :]
            out = [selected]
            if return_global:
                global_vector = F.adaptive_avg_pool2d(
                    selected[-1], output_size=1
                ).flatten(1)
                out.append(global_vector)
            if return_highres:
                out.append(highres_feature)
            return out[0] if len(out) == 1 else tuple(out)

        aligned = []
        source_features = features[-len(target_sizes) :]
        for feat, size in zip(source_features, target_sizes):
            if feat.shape[-2:] != tuple(size):
                feat = F.interpolate(
                    feat,
                    size=tuple(size),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(feat)
        if return_global and return_highres:
            global_vector = F.adaptive_avg_pool2d(aligned[-1], output_size=1).flatten(1)
            return aligned, global_vector, highres_feature
        if return_global:
            global_vector = F.adaptive_avg_pool2d(aligned[-1], output_size=1).flatten(1)
            return aligned, global_vector
        if return_highres:
            return aligned, highres_feature
        return aligned


class DEMPromptFusion(nn.Module):
    """Fuse global DEM terrain + global RGB image context into SAM3 text prompt features.

    In addition to the textual prompt, this module accepts up to two visual prompts:
      • terrain_vectors  — global pooled DEM feature  ∈ ℝ^(B×C)
      • rgb_vectors      — global pooled RGB feature  ∈ ℝ^(B×C)   (optional)

    Each modality is projected via an independent MLP and additively combined with
    the text tokens through a zero-init tanh-gated residual, so the base SAM3
    behaviour is preserved at initialisation.
    """

    def __init__(
        self,
        dim: int = 256,
        mode: str = "add",
        use_rgb_prompt: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"add", "concat"}:
            raise ValueError(f"Unsupported DEM prompt fusion mode: {mode}")
        self.mode = mode
        self.use_rgb_prompt = use_rgb_prompt
        self.terrain_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.rgb_proj = None
        if use_rgb_prompt:
            self.rgb_proj = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        self.concat_proj = None
        if mode == "concat":
            cat_in = dim * (3 if use_rgb_prompt else 2)
            self.concat_proj = nn.Sequential(
                nn.LayerNorm(cat_in),
                nn.Linear(cat_in, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        text_features: torch.Tensor,
        terrain_vectors: torch.Tensor,
        rgb_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multimodal-prompt residual fusion.

        Args:
            text_features:   Text tokens shaped [S, B, C].
            terrain_vectors: Global DEM vectors shaped [B, C].
            rgb_vectors:     Optional global RGB vectors shaped [B, C].
        """
        if text_features.dim() != 3:
            raise ValueError(
                f"text_features must be [S, B, C], got {text_features.shape}"
            )
        if terrain_vectors.dim() != 2:
            raise ValueError(
                f"terrain_vectors must be [B, C], got {terrain_vectors.shape}"
            )
        if text_features.shape[1] != terrain_vectors.shape[0]:
            raise ValueError(
                "text feature batch size and terrain vector batch size differ: "
                f"{text_features.shape[1]} vs {terrain_vectors.shape[0]}"
            )
        if rgb_vectors is not None and rgb_vectors.dim() != 2:
            raise ValueError(
                f"rgb_vectors must be [B, C], got {rgb_vectors.shape}"
            )

        terrain = self.terrain_proj(terrain_vectors).unsqueeze(0)
        terrain = terrain.to(device=text_features.device, dtype=text_features.dtype)

        rgb_proj_out = None
        if self.use_rgb_prompt and rgb_vectors is not None:
            assert self.rgb_proj is not None
            rgb_proj_out = self.rgb_proj(rgb_vectors).unsqueeze(0)
            rgb_proj_out = rgb_proj_out.to(
                device=text_features.device, dtype=text_features.dtype
            )

        if self.mode == "add":
            delta = terrain.expand_as(text_features)
            if rgb_proj_out is not None:
                delta = delta + rgb_proj_out.expand_as(text_features)
        else:
            assert self.concat_proj is not None
            parts = [text_features, terrain.expand_as(text_features)]
            if rgb_proj_out is not None:
                parts.append(rgb_proj_out.expand_as(text_features))
            delta = self.concat_proj(torch.cat(parts, dim=-1))
        return text_features + torch.tanh(self.gate).to(text_features.dtype) * delta


class DEMSpatialPromptTokens(nn.Module):
    """SAM-DEM V5 (Scheme 3): location-aware spatial DEM prompt tokens.

    ``DEMPromptFusion`` collapses the whole terrain feature map to a single global
    vector before adding it to the text prompt, discarding *where* the salient
    terrain is. This module keeps the spatial layout: the deepest DEM feature map
    is adaptive-pooled to a ``grid_size × grid_size`` grid, each cell projected to
    ``dim`` and given a learned positional embedding, producing ``grid_size**2``
    extra prompt tokens that are concatenated onto the SAM3 prompt sequence (so the
    decoder cross-attention can attend to terrain *by region*).

    A scalar ``tanh`` gate is zero-initialised, so at training step 0 every token is
    exactly the zero vector — a no-op for both the mean-pool scoring head and the
    decoder cross-attention. This lets the module load ``strict=False`` onto a V4
    checkpoint (it is the only new parameter group) and learn its contribution on
    top of the warm-started weights, mirroring the gated-residual convention used
    by the other DEM fusion modules in this file.
    """

    def __init__(self, dim: int = 256, grid_size: int = 8) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Learned positional embedding, one per grid cell.
        self.pos = nn.Parameter(torch.zeros(grid_size * grid_size, 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        # Zero-init the final projection and the gate so the tokens are exactly 0 at
        # step 0 (warm-start safe). The gate still receives gradient via the nonzero
        # pos embedding, so it lifts off zero and the module starts learning.
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, dem_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dem_feat: [B, C, H, W] deepest DEM feature map from ``DEMEncoder``.
        Returns:
            Prompt tokens shaped [grid_size**2, B, C].
        """
        if dem_feat.dim() != 4:
            raise ValueError(f"dem_feat must be [B, C, H, W], got {dem_feat.shape}")
        pooled = F.adaptive_avg_pool2d(dem_feat, output_size=self.grid_size)
        tokens = pooled.flatten(2).permute(2, 0, 1)  # [G*G, B, C]
        tokens = self.proj(self.norm(tokens))        # zero at init
        tokens = tokens + self.pos.to(tokens.dtype)  # learned per-cell position
        return torch.tanh(self.gate).to(tokens.dtype) * tokens


class TextAdapter(nn.Module):
    """SAM-DEM V7 (Scheme 1): lightweight LoRA-style residual adapter on the text
    prompt tokens. ``txt' = txt + tanh(gate) * Up(GELU(Down(LN(txt))))`` with a
    low-rank bottleneck (rank ``r`` << dim), so it is parameter-cheap (LoRA-like).
    The up-projection and the scalar gate are zero-initialised, so the adapter is a
    strict no-op at step 0 — a warm-started model is bit-identical until it learns to
    re-shape the (now visually-phrased) text embedding for the landslide concept."""

    def __init__(self, dim: int = 256, rank: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """text_features: [S, B, C] -> same shape."""
        delta = self.up(F.gelu(self.down(self.norm(text_features))))
        return text_features + torch.tanh(self.gate).to(text_features.dtype) * delta


class DecoupledGatedPromptFusion(nn.Module):
    """SAM-DEM V7 (Scheme 2): decoupled, gated injection of DEM-terrain and RGB-image
    context into the text prompt tokens.

    Unlike ``DEMPromptFusion`` (which broadcast-*adds* a shared global vector to every
    text token), this keeps the modalities DECOUPLED: terrain and appearance each get
    their own projection and the text tokens attend to them through SEPARATE cross-
    attention blocks, each with its OWN ``tanh`` gate. The model can therefore weight
    "what the terrain says" against "what the image looks like" independently, and the
    text stream stays clean (residual). Both gates are zero-init -> no-op warm start."""

    def __init__(self, dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.txt_norm = nn.LayerNorm(dim)
        self.dem_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.rgb_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.dem_attn = nn.MultiheadAttention(dim, num_heads, batch_first=False)
        self.rgb_attn = nn.MultiheadAttention(dim, num_heads, batch_first=False)
        self.dem_gate = nn.Parameter(torch.zeros(1))
        self.rgb_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        text_features: torch.Tensor,
        terrain_vectors: torch.Tensor,
        rgb_vectors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """text_features: [S, B, C]; terrain/rgb vectors: [B, C]. Returns [S, B, C]."""
        out = text_features
        dem_kv = self.dem_proj(terrain_vectors).unsqueeze(0).to(out.dtype)  # [1,B,C]
        attn_dem, _ = self.dem_attn(self.txt_norm(out), dem_kv, dem_kv, need_weights=False)
        out = out + torch.tanh(self.dem_gate).to(out.dtype) * attn_dem
        if rgb_vectors is not None:
            rgb_kv = self.rgb_proj(rgb_vectors).unsqueeze(0).to(out.dtype)
            attn_rgb, _ = self.rgb_attn(self.txt_norm(out), rgb_kv, rgb_kv, need_weights=False)
            out = out + torch.tanh(self.rgb_gate).to(out.dtype) * attn_rgb
        return out


class VisualPromptAdapter(nn.Module):
    """SAM-DEM V8: project reference-image exemplar vectors into learnable visual
    prompt tokens. Inputs are pooled landslide-region features from a small bank of
    reference images; this module projects them and scales by a zero-init gate, so at
    step 0 the visual prompt contributes nothing (warm-start safe) and the model learns
    to use the exemplars on top of the warm-started weights."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, exemplars: torch.Tensor) -> torch.Tensor:
        """exemplars: [N, B, C] pooled reference vectors -> gated tokens [N, B, C]."""
        return torch.tanh(self.gate).to(exemplars.dtype) * self.proj(self.norm(exemplars))


class VerificationHead(nn.Module):
    """SAM-DEM V9 (Scheme 6): per-query landslide verification head (detect-then-verify).

    Scores each decoder query embedding as landslide / not-landslide. At inference its
    sigmoid multiplies the query objectness, so weakly-supported (false-positive) query
    regions are suppressed — structurally precision-oriented and recall-safe (it can only
    *down*-weight queries). A small MLP; the final layer is zero-init so at warm-start it
    outputs 0 (sigmoid 0.5, a near-neutral gate) until trained."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )
        # Zero weight + large positive bias => verify logit ~+4 (gate ~0.98 ~ 1) at init,
        # so the inference gate is a near no-op for the warm-started model until trained.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 4.0)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """queries: [B, Q, C] -> verification logits [B, Q, 1]."""
        return self.net(queries)


class CrossModalAlignHead(nn.Module):
    """SAM-DEM V7 (Scheme 4): projection heads that map pooled visual features and the
    pooled text-prompt embedding into a shared, L2-normalised space so a cross-modal
    alignment loss can pull the landslide text prompt toward the visual features of GT
    landslide pixels and push it away from background pixels (a CLIP-style region-text
    constraint that regularises the prompt to be visually discriminative)."""

    def __init__(self, dim: int = 256, proj_dim: int = 256) -> None:
        super().__init__()
        self.vis_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj_dim))
        self.txt_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj_dim))

    def encode_vis(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.vis_proj(v), dim=-1)

    def encode_txt(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.txt_proj(t), dim=-1)


class DEMCrossAttentionFusion(nn.Module):
    """Fuse RGB image features with DEM terrain features via cross attention."""

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        fuse_all_levels: bool = True,
        v3_refine: bool = False,
    ) -> None:
        super().__init__()
        self.fuse_all_levels = fuse_all_levels
        self.rgb_norm = nn.LayerNorm(dim)
        self.dem_norm = nn.LayerNorm(dim)
        # Shared 2D sinusoidal pos encoding for both modalities. Because DEM features are
        # resized to match RGB spatial shape before attention, identical pos enc at the same
        # (i, j) gives the cross-attn an explicit "same-pixel" prior — i.e. spatial alignment.
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=dim, normalize=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

        # SAM-DEM v3: a position-wise FFN applied to the attended terrain signal
        # before it is gated back into the RGB stream (standard transformer
        # attn→FFN block). Its own zero-init gate makes it a no-op at step 0.
        self.v3_refine = v3_refine
        if v3_refine:
            self.ffn_norm = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            self.ffn_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        rgb_features: List[torch.Tensor],
        dem_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if len(rgb_features) != len(dem_features):
            raise ValueError(
                "RGB and DEM feature lists must have the same length, "
                f"got {len(rgb_features)} and {len(dem_features)}"
            )

        fused = []
        start_fusing = 0 if self.fuse_all_levels else len(rgb_features) - 1
        for idx, (rgb_feat, dem_feat) in enumerate(zip(rgb_features, dem_features)):
            if idx < start_fusing:
                fused.append(rgb_feat)
                continue
            fused.append(self._fuse_single_level(rgb_feat, dem_feat))
        return fused

    def _fuse_single_level(
        self,
        rgb_feat: torch.Tensor,
        dem_feat: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_feat.shape[-2:] != dem_feat.shape[-2:]:
            dem_feat = F.interpolate(
                dem_feat,
                size=rgb_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        dem_feat = dem_feat.to(device=rgb_feat.device, dtype=rgb_feat.dtype)

        batch_size, channels, height, width = rgb_feat.shape

        # 2D sinusoidal pos enc, shared across modalities. Shape: [B, C, H, W] → tokens [B, HW, C].
        pos = self.pos_embed(rgb_feat).to(dtype=rgb_feat.dtype)
        pos_tokens = pos.flatten(2).transpose(1, 2)

        rgb_tokens = rgb_feat.flatten(2).transpose(1, 2)
        dem_tokens = dem_feat.flatten(2).transpose(1, 2)

        # Add pos enc to query and key (not value), so attention weights depend on (content + position)
        # but the aggregated DEM signal stays content-only.
        query = self.rgb_norm(rgb_tokens) + pos_tokens
        key = self.dem_norm(dem_tokens) + pos_tokens
        value = self.dem_norm(dem_tokens)

        attn_out, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )
        attn_out = self.out_proj(attn_out)
        if self.v3_refine:
            attn_out = attn_out + torch.tanh(self.ffn_gate).to(attn_out.dtype) * self.ffn(
                self.ffn_norm(attn_out)
            )
        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        return rgb_feat + torch.tanh(self.gate).to(rgb_feat.dtype) * attn_out


class DEMDecoderFusion(nn.Module):
    """Inject DEM terrain features into the pixel decoder's high-resolution output.

    Applied to the output of `PixelDecoder` immediately before the semantic and
    instance segmentation heads, this module gives the per-pixel mask logits
    direct access to terrain edges (slope discontinuities, ridge / scarp lines)
    that drive landslide boundary detection. DEM features are resized to the
    pixel-decoder output resolution, concatenated with the RGB-derived embedding,
    and projected through a small CNN. A scalar tanh gate (zero-initialised)
    preserves the base SAM3 mask head at step 0.
    """

    def __init__(self, dim: int = 256, dem_dim: int = 256, v3_refine: bool = False) -> None:
        super().__init__()
        self.dem_proj = nn.Conv2d(dem_dim, dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False),
            _gn(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            _gn(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

        # SAM-DEM v3: an extra gated residual refinement of the terrain-fused
        # pixel embedding, giving the per-pixel mask head more depth to resolve
        # landslide boundaries. Zero-init gate → no-op at step 0.
        self.v3_refine = v3_refine
        if v3_refine:
            self.refine = _SERefine(dim)
            self.refine_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pixel_embed: torch.Tensor,
        dem_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_embed: [B, C, H, W]  output of PixelDecoder.
            dem_feat:    [B, C_d, H_d, W_d]  highest-resolution DEM feature map
                         from DEMEncoder.
        Returns:
            Refined pixel_embed of the same shape.
        """
        if dem_feat.shape[-2:] != pixel_embed.shape[-2:]:
            dem_feat = F.interpolate(
                dem_feat,
                size=pixel_embed.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        dem_feat = dem_feat.to(device=pixel_embed.device, dtype=pixel_embed.dtype)
        dem_proj = self.dem_proj(dem_feat)
        delta = self.fuse(torch.cat([pixel_embed, dem_proj], dim=1))
        out = pixel_embed + torch.tanh(self.gate).to(pixel_embed.dtype) * delta
        if self.v3_refine:
            out = out + torch.tanh(self.refine_gate).to(out.dtype) * self.refine(out)
        return out
