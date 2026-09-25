import torch
import torch.nn as nn

from .hand_pose_encoder import HandEncoder
from .pose_encoder import PoseEncoder


class PoseBranch(nn.Module):
    """
    Transformer-based encoder for pose sequences.

    Input:
        x: (B, T, input_dim)

    Output:
        x: (B, T, d_model)
    """

    def __init__(self, config):
        super().__init__()

        self.max_seq_len = config.dataset.max_length
        self.use_rgb = config.model.use_rgb

        self.use_face_pose = config.pose.face.enabled
        self.use_body_pose = config.pose.body.enabled

        self.pre_norm = config.model.pre_norm

        # hand
        self.hand_encoder = HandEncoder(
            hand_pose_input_dim=config.pose.left_hand.num_joints
            * config.pose.left_hand.channels,
            pose_embed_dim=config.pose.left_hand.embed_dim,
            pose_num_heads=config.pose.left_hand.num_heads,
            rgb_image_size=config.rgb.image_size,
            rgb_patch_size=config.rgb.patch_size,
            rgb_in_channels=config.rgb.channels,
            rgb_embed_dim=config.rgb.embed_dim,
            act=str(config.model.act),
            rgb_num_heads=config.rgb.num_heads,
            depth=config.model.depth,
            dropout=config.model.dropout,
            pose_rgb_num_heads=config.model.pose_rgb_num_heads,
            pose_pose_num_heads=config.model.pose_pose_num_heads,
            mlp_ratio=config.model.mlp_ratio,
            use_rgb=config.model.use_rgb,
            pre_norm=self.pre_norm,
        )

        # face
        self.face_encoder = (
            PoseEncoder(
                input_dim=config.pose.face.num_joints * config.pose.face.channels,
                embed_dim=config.pose.face.embed_dim,
                num_heads=config.pose.face.num_heads,
                depth=config.model.depth,
                mlp_ratio=config.model.mlp_ratio,
                dropout=config.model.dropout,
                act=str(config.model.act),
                max_seq_len=config.dataset.max_length,
                pre_norm=self.pre_norm,
            )
            if config.pose.face.enabled
            else None
        )

        # body
        self.body_encoder = (
            PoseEncoder(
                input_dim=config.pose.body.num_joints * config.pose.body.channels,
                embed_dim=config.pose.body.embed_dim,
                num_heads=config.pose.body.num_heads,
                depth=config.model.depth,
                mlp_ratio=config.model.mlp_ratio,
                dropout=config.model.dropout,
                act=str(config.model.act),
                max_seq_len=config.dataset.max_length,
                pre_norm=self.pre_norm,
            )
            if config.pose.body.enabled
            else None
        )

    def forward(
        self,
        lh_pose,
        rh_pose,
        lh_rgb: torch.Tensor | None,
        rh_rgb: torch.Tensor | None,
        face: torch.Tensor | None,
        body: torch.Tensor | None,
    ):
        # (B, L, J, 2) -> (B, L, J*2)
        lh_pose = lh_pose.flatten(start_dim=-2)
        rh_pose = rh_pose.flatten(start_dim=-2)
        face = face.flatten(start_dim=-2) if face is not None else None
        body = body.flatten(start_dim=-2) if body is not None else None

        lh_pose, rh_pose = self.hand_encoder(lh_pose, rh_pose, lh_rgb, rh_rgb)

        if (self.face_encoder is not None) and (face is not None):
            face = self.face_encoder(face)

        if (self.body_encoder is not None) and (body is not None):
            body = self.body_encoder(body)

        # Concat
        features = [lh_pose, rh_pose]
        if face is not None:
            features.append(face)

        if body is not None:
            features.append(body)

        fused = torch.cat(features, dim=-1)

        return fused
