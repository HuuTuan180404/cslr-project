import torch
import torch.nn as nn

from .pose_branch import PoseBranch
from .text_branch import TextBranch

from utils.utils import get_config, get_activation


class DualBranchCSLRModel(nn.Module):
    """
    Sketch model cho bài toán Continuous Sign Language Recognition (CSLR).

    Architecture:
        - Image/Pose branch: PoseBranch (pose + rgb)
        - Text branch: TextBranch (Mamba + projection)
        - Fusion head: concat image/text embeddings -> classifier

    Đây là bản phác thảo để bạn sửa tiếp theo, không cần phải tối ưu ngay.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        dropout = get_config(config, "model", "dropout")
        act = get_config(config, "model", "act")

        # ------------------------------------------------------------
        # 1. Image branch
        # ------------------------------------------------------------
        self.image_encoder = PoseBranch(config)

        # Tính chiều đầu ra của image encoder theo cấu hình hiện có
        # - left hand: pose left embed dim
        # - right hand: pose right embed dim
        # - body/face nếu enable
        left_dim = get_config(config, "pose", "left_hand", "embed_dim")
        right_dim = get_config(config, "pose", "right_hand", "embed_dim")

        face_dim = 0
        if config.pose.face.enabled:
            face_dim = get_config(config, "pose", "face", "embed_dim")

        body_dim = 0
        if config.pose.body.enabled:
            # if get_config(config, "pose", "body", "enabled"):
            body_dim = get_config(config, "pose", "body", "embed_dim")

        self.pose_dim = left_dim + right_dim + face_dim + body_dim

        # ------------------------------------------------------------
        # 2. Text branch
        # ------------------------------------------------------------
        self.vocab_size = get_config(config, "model", "text_branch", "vocab_size")
        self.text_projection_dim = get_config(
            config, "model", "text_branch", "proj_dim"
        )

        self.text_encoder = TextBranch(config)

        # ------------------------------------------------------------
        # 3. Projection head cho visual branch
        # ------------------------------------------------------------
        project_dim = get_config(config, "model", "pose_branch", "proj_dim")
        self.image_proj = nn.Sequential(
            nn.Linear(self.pose_dim, project_dim),
            get_activation(act),
            nn.Dropout(dropout),
            nn.Linear(project_dim, project_dim),
        )

        # ------------------------------------------------------------
        # 4. Classifier / fusion head (phác thảo)
        # ------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.text_projection_dim + project_dim, 256),
            get_activation(act),
            nn.Dropout(dropout),
            nn.Linear(256, self.vocab_size),
        )

    def forward(
        self,
        lh_pose,
        rh_pose,
        lh_rgb=None,
        rh_rgb=None,
        face=None,
        body=None,
        gloss_ids=None,
        text_attention_mask=None,
    ):
        # ------------------------------------------------------------
        # 1. Visual features
        # ------------------------------------------------------------
        image_features = self.image_encoder.forward(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body
        )
        # image_features: [B, T, D_img]

        # Pool temporal dimension để lấy biểu diễn toàn video
        image_context = image_features.mean(dim=1)  # [B, D_img]
        z_img = self.image_proj(image_context)  # [B, proj_dim]

        # ------------------------------------------------------------
        # 2. Text features
        # ------------------------------------------------------------
        if gloss_ids is not None and text_attention_mask is not None:
            text_dict = self.text_encoder.forward(
                input_ids=gloss_ids, attention_mask=text_attention_mask
            )
            z_text = text_dict["z_text"]  # [B, proj_text]
            token_features = text_dict["token_features"]
            sentence_features = text_dict["sentence_features"]
        else:
            z_text = None
            token_features = None
            sentence_features = None

        # ------------------------------------------------------------
        # 3. Fusion
        # ------------------------------------------------------------
        if z_text is not None:
            fused = torch.cat([z_img, z_text], dim=-1)  # [B, D_img + D_text]
            logits = self.classifier(fused)
        else:
            logits = None

        return {
            "image_features": image_features,
            "image_context": image_context,
            "z_img": z_img,
            "token_features": token_features,
            "sentence_features": sentence_features,
            "z_text": z_text,
            "logits": logits,
        }


# Alias ngắn gọn để tiện import
CSLRModel = DualBranchCSLRModel
