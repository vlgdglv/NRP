import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from utils.logger import get_logger
from utils import is_rank0

logger = get_logger(__name__)


def load_janus_full(model_path, checkpoint_path=None):
    from model.janus_arch.models import MultiModalityCausalLM

    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )

    if checkpoint_path is not None:
        if is_rank0():
            logger.info(f"Loading full checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_rank0():
        logger.info(f"Janus full finetune: {trainable}/{total} params trainable ({100*trainable/total:.1f}%)")

    return model


class JanusFullModel(nn.Module):
    """Training wrapper for full-parameter Janus finetuning.

    Unlike RowExpertModel which uses disable_adapter() for teacher logits,
    this uses pre-computed teacher logits from the dataset (use_teacher=True)
    or trains with CE loss only.
    """
    def __init__(
        self,
        base_model,
        use_ce: bool = True,
        ce_weight: float = 1.0,
        use_kd: bool = False,
        kd_weight: float = 1.0,
        kd_temp: float = 1.0,
        use_topk_mass: bool = False,
        topk_mass_topk: int = 64,
        topk_mass_weight: float = 1.0,
        use_topk_cover: bool = False,
        topk_cover_weight: float = 1.0,
        topk_cover_temp: float = 1.0,
        topk_cover_topk: int = 128,
        image_latent_width: int = 24,
        image_latent_height: int = 24,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)
        self.use_ce = use_ce
        self.ce_weight = ce_weight
        self.use_kd = use_kd
        self.kd_weight = kd_weight
        self.kd_temp = kd_temp
        self.use_topk_mass = use_topk_mass
        self.topk_mass_topk = topk_mass_topk
        self.topk_mass_weight = topk_mass_weight
        self.use_topk_cover = use_topk_cover
        self.topk_cover_weight = topk_cover_weight
        self.topk_cover_temp = topk_cover_temp
        self.topk_cover_topk = topk_cover_topk
        self.image_latent_width = image_latent_width
        self.image_latent_height = image_latent_height
        self.offset = image_latent_width - 1

    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        teacher_token=None,
        teacher_logits=None,
        teacher_hidden_states=None,
        target_row_ids=None,
        target_col_ids=None,
        row_valid_mask=None,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )

        logits = outputs.logits
        loss = 0.0
        output_dict = {"logits": logits}

        if labels is not None:
            if self.use_ce:
                shift_logits = logits.view(-1, logits.shape[-1])
                shift_labels = labels.view(-1).long()
                ce_loss = CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_labels)
                loss = loss + self.ce_weight * ce_loss
                output_dict["ce_loss"] = ce_loss.detach()

            needs_teacher = (self.use_kd or self.use_topk_mass or self.use_topk_cover)
            if needs_teacher and teacher_logits is None:
                raise ValueError(
                    "Full finetuning requires pre-computed teacher logits "
                    "(use --use_teacher with teacher data). "
                    "No adapter to disable for online teacher."
                )

            if needs_teacher and teacher_logits is not None:
                student_logits = logits
                if self.offset > 0:
                    pad = torch.zeros_like(teacher_logits[:, :self.offset, :])
                    aligned_teacher = torch.cat([teacher_logits[:, self.offset:, :], pad], dim=1)
                else:
                    aligned_teacher = teacher_logits

                valid_mask = (labels != -100)

                if self.use_kd:
                    s = student_logits[valid_mask]
                    t = aligned_teacher[valid_mask]
                    T = float(self.kd_temp)
                    log_p = F.log_softmax(s / T, dim=-1)
                    p_t = F.softmax(t / T, dim=-1)
                    kd_loss = F.kl_div(log_p, p_t, reduction="batchmean") * (T * T)
                    output_dict["kd_loss"] = kd_loss.detach()
                    loss = loss + self.kd_weight * kd_loss

                if self.use_topk_cover:
                    s = student_logits[valid_mask]
                    t = aligned_teacher[valid_mask]
                    T = float(self.topk_cover_temp)
                    topk_idx = torch.topk(t, k=self.topk_cover_topk, dim=-1).indices
                    p_s = F.softmax(s / T, dim=-1)
                    p_s_topk = torch.gather(p_s, dim=-1, index=topk_idx).sum(dim=-1)
                    tc_loss = -torch.log(p_s_topk.clamp_min(1e-8)).mean()
                    output_dict["tpkc_loss"] = tc_loss.detach()
                    loss = loss + self.topk_cover_weight * tc_loss

                if self.use_topk_mass:
                    k = self.topk_mass_topk
                    t_logits = aligned_teacher.detach()
                    _, t_topk_idx = torch.topk(t_logits, k=k, dim=-1)
                    t_prob = F.softmax(t_logits, dim=-1).detach()
                    t_topk_prob = torch.gather(t_prob, dim=-1, index=t_topk_idx)
                    s_logprob = F.log_softmax(student_logits, dim=-1)
                    s_topk_logprob = torch.gather(s_logprob, dim=-1, index=t_topk_idx)
                    topk_ce = -(t_topk_prob * s_topk_logprob).sum(dim=-1)
                    tm_loss = topk_ce[valid_mask].mean()
                    output_dict["topk_mass_loss"] = tm_loss.detach()
                    loss = loss + self.topk_mass_weight * tm_loss

        output_dict["loss"] = loss
        return output_dict
