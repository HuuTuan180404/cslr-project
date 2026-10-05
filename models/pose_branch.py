import torch
import torch.nn as nn
from models.mlp import MLP

from .hand_pose_encoder import HandEncoder
from .pose_encoder import PoseEncoder

from mamba_ssm import Mamba2


class MambaPoseEncoder(nn.Module):
    """
    Mamba-based temporal encoder.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """

    def __init__(
        self, input_dim, embed_dim, d_state, d_conv, expand, dropout, depth, pre_norm
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.pre_norm = pre_norm

        # Project concatenated pose features to Mamba dimension
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.mamba_layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(depth)
            ]
        )

        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(depth)]
        )

        self.dropout = nn.Dropout(dropout)

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args: x: (B, T, input_dim)

        Returns: x: (B, T, d_model)
        """

        # [B, T, input_dim]
        x = self.input_proj(x)

        for mamba, norm in zip(self.mamba_layers, self.norm_layers):
            residual = x

            if self.pre_norm:
                # Pre-Norm
                x = norm(x)
                x = mamba(x)
                x = self.dropout(x)
                x = residual + x

            else:
                # Post-Norm
                x = mamba(x)
                x = self.dropout(x)
                x = residual + x
                x = norm(x)

        # --------------------------------------------------
        # Final normalization if pre_norm
        # --------------------------------------------------
        if self.pre_norm:
            x = self.final_norm(x)

        return x


class PoseBranch(nn.Module):
    """
    Transformer-based encoder for pose sequences.

    Input: x: (B, T, input_dim)

    Output: x: (B, T, d_model)
    """

    def __init__(self, cfg):
        super().__init__()

        self.max_frames = cfg.dataset.max_frames
        self.use_rgb = cfg.model.use_rgb

        self.use_face_pose = cfg.pose.face.enabled
        self.use_body_pose = cfg.pose.body.enabled

        self.pre_norm = cfg.model.pre_norm

        pe_type = cfg.model.pe_type
        self.ctc_num_classes = cfg.model.t_branch.vocab_size + 3

        # ------------------------------------------------------------
        # PROJECTION: num_joints * channels -> embed_dim
        # ------------------------------------------------------------
        self.lh_pose_proj = nn.Linear(
            cfg.pose.left_hand.num_joints * cfg.pose.left_hand.channels,
            cfg.pose.left_hand.embed_dim,
        )
        self.rh_pose_proj = nn.Linear(
            cfg.pose.right_hand.num_joints * cfg.pose.right_hand.channels,
            cfg.pose.right_hand.embed_dim,
        )

        self.face_pose_proj = (
            nn.Linear(
                cfg.pose.face.num_joints * cfg.pose.face.channels,
                cfg.pose.face.embed_dim,
            )
            if self.use_face_pose
            else None
        )

        self.body_pose_proj = (
            nn.Linear(
                cfg.pose.body.num_joints * cfg.pose.body.channels,
                cfg.pose.body.embed_dim,
            )
            if self.use_body_pose
            else None
        )

        # ------------------------------------------------------------
        # ENCODER
        # ------------------------------------------------------------
        self.hand_encoder = HandEncoder(
            pose_embed_dim=cfg.pose.left_hand.embed_dim,
            pose_num_heads=cfg.pose.left_hand.num_heads,
            rgb_image_size=cfg.rgb.image_size,
            rgb_patch_size=cfg.rgb.patch_size,
            rgb_in_channels=cfg.rgb.channels,
            rgb_embed_dim=cfg.rgb.embed_dim,
            act=str(cfg.model.act),
            rgb_num_heads=cfg.rgb.num_heads,
            depth=cfg.model.p_branch.depth,
            dropout=cfg.model.dropout,
            pose_rgb_num_heads=cfg.model.pose_rgb_num_heads,
            pose_pose_num_heads=cfg.model.pose_pose_num_heads,
            mlp_ratio=cfg.model.mlp_ratio,
            use_rgb=cfg.model.use_rgb,
            pre_norm=self.pre_norm,
            pe_type=pe_type,
        )

        self.face_encoder = (
            PoseEncoder(
                embed_dim=cfg.pose.face.embed_dim,
                num_heads=cfg.pose.face.num_heads,
                depth=cfg.model.p_branch.depth,
                mlp_ratio=cfg.model.mlp_ratio,
                dropout=cfg.model.dropout,
                act=str(cfg.model.act),
                max_frames=cfg.dataset.max_frames,
                pre_norm=self.pre_norm,
                pe_type=pe_type,
            )
            if self.use_face_pose
            else None
        )

        self.body_encoder = (
            PoseEncoder(
                embed_dim=cfg.pose.body.embed_dim,
                num_heads=cfg.pose.body.num_heads,
                depth=cfg.model.p_branch.depth,
                mlp_ratio=cfg.model.mlp_ratio,
                dropout=cfg.model.dropout,
                act=str(cfg.model.act),
                max_frames=cfg.dataset.max_frames,
                pre_norm=self.pre_norm,
                pe_type=pe_type,
            )
            if self.use_body_pose
            else None
        )

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        self.full_pose_dim = cfg.pose.left_hand.embed_dim * 2
        if cfg.pose.face.enabled:
            self.full_pose_dim += cfg.pose.face.embed_dim
        if cfg.pose.body.enabled:
            self.full_pose_dim += cfg.pose.body.embed_dim

        # ------------------------------------------------------------
        # MAMBA
        # ------------------------------------------------------------
        self.mamba = MambaPoseEncoder(
            input_dim=self.full_pose_dim,
            embed_dim=cfg.model.p_branch.embed_dim,
            d_state=cfg.model.p_branch.d_state,
            d_conv=cfg.model.p_branch.conv_kernel,
            expand=cfg.model.p_branch.expand,
            dropout=cfg.model.dropout,
            depth=cfg.model.p_branch.depth,
            pre_norm=cfg.model.pre_norm,
        )

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        self.ctc_head = nn.Linear(cfg.model.p_branch.embed_dim, self.ctc_num_classes)

        self.final_proj = nn.Linear(
            cfg.model.p_branch.embed_dim, cfg.model.p_branch.proj_dim
        )

    def forward(
        self,
        lh_pose,
        rh_pose,
        lh_rgb=None,
        rh_rgb=None,
        face=None,
        body=None,
        input_frames=None,
    ):
        # (B, L, J, 2) -> (B, L, J*2)
        lh_pose = lh_pose.flatten(start_dim=-2)
        rh_pose = rh_pose.flatten(start_dim=-2)
        face = face.flatten(start_dim=-2) if face is not None else None
        body = body.flatten(start_dim=-2) if body is not None else None

        # ------------------------------------------------------------
        # PROJECTION: num_joints * channels -> embed_dim
        # ------------------------------------------------------------
        lh_pose = self.lh_pose_proj(lh_pose)
        rh_pose = self.rh_pose_proj(rh_pose)

        face = self.face_pose_proj(face) if self.face_pose_proj is not None else face
        body = self.body_pose_proj(body) if self.body_pose_proj is not None else body

        # ------------------------------------------------------------
        # ENCODER
        # ------------------------------------------------------------
        lh_pose, rh_pose = self.hand_encoder(lh_pose, rh_pose, lh_rgb, rh_rgb)

        if (self.face_encoder is not None) and (face is not None):
            face = self.face_encoder(face)

        if (self.body_encoder is not None) and (body is not None):
            body = self.body_encoder(body)

        # ------------------------------------------------------------
        # CONCAT
        # ------------------------------------------------------------
        features = [lh_pose, rh_pose]
        if face is not None:
            features.append(face)

        if body is not None:
            features.append(body)
        fused = torch.cat(features, dim=-1)

        # ------------------------------------------------------------
        # MAMBA
        # ------------------------------------------------------------
        pose_features = self.mamba.forward(fused)

        # CTC logits
        ctc_logits = self.ctc_head(pose_features)  # [B,L,num_classes]

        pose_features = self.masked_mean_pooling(pose_features, input_frames)  # [B,L,D]
        pose_features = self.final_proj(pose_features)  # [B, proj_D]

        return ctc_logits, pose_features

    def masked_mean_pooling(self, x, lengths) -> torch.Tensor:
        """
        Masked Mean Pooling theo temporal dimension.

        Args:
            x:
                Tensor shape [B, L, D]
            lengths:
                Tensor shape [B], số frame hợp lệ của mỗi sample.

        Returns:
            pooled:
                Tensor shape [B, D]
        """
        B, L, D = x.shape

        # [L]
        frame_idx = torch.arange(L, device=x.device)

        # [B, L]
        mask = frame_idx.unsqueeze(0) < lengths.unsqueeze(1)

        # [B, L, 1]
        mask = mask.unsqueeze(-1).to(x.dtype)

        # Tổng các frame hợp lệ
        summed = (x * mask).sum(dim=1)  # [B, D]

        # Số frame hợp lệ
        count = mask.sum(dim=1).clamp_min(1.0)  # [B, 1]

        # Mean
        pooled = summed / count

        return pooled  # [B, D]
