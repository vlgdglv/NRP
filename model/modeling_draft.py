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
        
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels=None, 
        teacher_token=None, 
        teacher_logits=None
    ):    
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
                shift_labels = labels.view(-1).long()

                loss_fn = CrossEntropyLoss()
                ce_loss = loss_fn(shift_logits, shift_labels)
                loss = loss + self.ce_weight * ce_loss
                output_dict["ce_loss"] = ce_loss.item()
            
            # sparse kd
            if self.use_kd:
                assert teacher_token is not None and teacher_logits is not None
                invalid = -100
                valid_mask = (labels != invalid).view(-1) 
                s_logits_v = logits.view(-1, logits.size(-1))[valid_mask]
                t_indices_v = teacher_token.view(-1, teacher_token.size(-1))[valid_mask].long()
                t_values_v = teacher_logits.view(-1, teacher_logits.size(-1))[valid_mask]
                
                student_topk_logits = torch.gather(s_logits_v, dim=-1, index=t_indices_v)

                T = float(self.kd_temp)
                log_p_student = F.log_softmax(student_topk_logits / T, dim=-1)
                p_teacher = F.softmax(t_values_v / T, dim=-1)

                
                kd_loss = F.kl_div(
                    log_p_student,
                    p_teacher,
                    reduction="batchmean",
                ) * (T * T)
                output_dict["kd_loss"] = kd_loss.item()
                
                loss = loss + self.kd_weight * kd_loss
            # print("kd_loss: ", kd_loss.item(), "ce_loss: ", ce_loss.item())
        output_dict["loss"] = loss
        return output_dict