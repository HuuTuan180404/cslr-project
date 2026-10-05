import torch
import torch.nn as nn
import torch.nn.functional as F

from .pose_branch import PoseBranch
from .text_branch import TextBranch

from utils.utils import get_config
from models.utils import get_activation
from models.mlp import MLP


class CSLR(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.config = cfg

        # ------------------------------------------------------------
        # 1. Pose branch
        # ------------------------------------------------------------
        self.pose_branch = PoseBranch(cfg)
        self.full_pose_dim = self.pose_branch.full_pose_dim

        # ------------------------------------------------------------
        # 2. Text branch
        # ------------------------------------------------------------
        self.vocab_size = cfg.model.t_branch.vocab_size
        # CTC classes: PAD (0), glosses (1..vocab_size), UNK, then BLANK.
        self.ctc_num_classes = self.vocab_size + 3
        self.text_projection_dim = cfg.model.t_branch.proj_dim

        self.text_encoder = TextBranch(cfg)

        # ------------------------------------------------------------
        # 3. Projection head cho visual branch
        # ------------------------------------------------------------
        self.pose_proj = nn.Linear(
            cfg.model.p_branch.embed_dim, cfg.model.p_branch.proj_dim
        )

        # ------------------------------------------------------------
        # 4. Classifier / fusion head (phác thảo)
        # ------------------------------------------------------------
        self.ctc_logits = nn.Linear(cfg.model.p_branch.embed_dim, self.ctc_num_classes)

        self.contrastive_logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / cfg.loss.contrastive.temperature))
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
        input_frames=None,
    ):
        # ------------------------------------------------------------
        # 1. Visual features
        # ------------------------------------------------------------
        ctc_logits, pose_features = self.pose_branch.forward(
            lh_pose, rh_pose, lh_rgb, rh_rgb, face, body, input_frames
        )  # (B, T, num_classes), [B, proj_D]

        # ------------------------------------------------------------
        # 2. Text features
        # ------------------------------------------------------------
        z_text = None
        if gloss_ids is not None and text_attention_mask is not None:
            z_text = self.text_encoder.forward(
                input_ids=gloss_ids, attention_mask=text_attention_mask
            )  # [B, proj_D]

        # ------------------------------------------------------------
        # normalize
        # ------------------------------------------------------------
        z_pose = F.normalize(pose_features, dim=-1)
        z_text = F.normalize(z_text, dim=-1) if z_text is not None else z_text

        contrastive_logits = None
        if z_text is not None:
            similarity = z_pose @ z_text.T
            contrastive_logits = similarity * self.contrastive_logit_scale.exp()

        return {"contrastive_logits": contrastive_logits, "ctc_logits": ctc_logits}


# Alias ngắn gọn để tiện import
CSLRModel = CSLR
