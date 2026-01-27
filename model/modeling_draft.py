import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class RowExpertModel(nn.Module):
    def __init__(self, 
        base_model,
        use_ce: bool = True,
        ce_weight: float = 1.0,
        use_kd: bool = True,
        kd_weight: float = 1.0, 
        kd_temp: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)
        self.use_ce = use_ce
        self.ce_weight = ce_weight
        self.use_kd = use_kd
        self.kd_weight = kd_weight
        self.kd_temp = kd_temp
        
    def forward(self, input_ids, attention_mask, labels=None):
        
        teacher_logits = None
        if self.use_kd:
            was_training = self.base_model.training
            self.base_model.eval()
            with self.base_model.disable_adapter():
                with torch.no_grad():
                    teacher_outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=False,
                    )
                    teacher_logits = teacher_outputs.logits
            if was_training:
                self.base_model.train()
            
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )

        logits = outputs.logits
        loss = 0.0
        output_dict = {}
        output_dict["logits"] = logits
        
        if labels is not None:
            if self.use_ce:
                shift_logits = logits.view(-1, logits.shape[-1])
                shift_labels = labels.view(-1)

                loss_fn = CrossEntropyLoss()
                ce_loss = loss_fn(shift_logits, shift_labels)
                loss = loss + self.ce_weight * ce_loss
                output_dict["ce_loss"] = ce_loss.item()
            
            if self.use_kd and teacher_logits is not None:
                invalid = -100
                valid_mask = (labels != invalid)

                student_logits = logits[valid_mask]
                teacher_logits = teacher_logits[valid_mask]
                # V = self.config.vocab_size
                # allowed = torch.zeros(V, device=teacher_logits.device, dtype=torch.bool)
                # allowed[:8198] = True
                # neg_inf = torch.finfo(teacher_logits.dtype).min
                # vocab_mask = torch.where(allowed, torch.zeros((), device=teacher_logits.device, dtype=teacher_logits.dtype), neg_inf)
                # student_logits = student_logits + vocab_mask
                # teacher_logits = teacher_logits + vocab_mask

                T = float(self.kd_temp)
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T * T)
                output_dict["kd_loss"] = kd_loss.item()
                
                loss = loss + self.kd_weight * kd_loss
        output_dict["loss"] = loss
        return output_dict