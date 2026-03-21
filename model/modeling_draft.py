import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

class RowExpertModel(nn.Module):
    def __init__(self, 
        base_model,
        use_ce: bool = True,
        ce_weight: float = 1.0,
        use_kd: bool = False,
        use_tv: bool = False,
        use_acc: bool = False,
        use_topk_cover: bool = False,
        kd_weight: float = 1.0, 
        kd_temp: float = 1.0,
        tv_weight: float = 1.0, 
        tv_temp: float = 1.0,
        acc_weight: float = 1.0, 
        acc_temp: float = 1.0,
        topk_cover_weight: float = 1.0,
        topk_cover_temp: float = 1.0,
        topk_cover_topk: int = 128,
        use_mse: bool = False,
        mse_weight: float = 1.0,
        image_latent_width: int = 49,
        image_latent_height: int = 48,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", None)
        self.use_ce = use_ce
        self.ce_weight = ce_weight
        self.use_kd = use_kd
        self.kd_weight = kd_weight
        self.kd_temp = kd_temp
        self.use_tv = use_tv
        self.tv_weight = tv_weight
        self.tv_temp = tv_temp
        self.use_acc = use_acc
        self.acc_weight = acc_weight
        self.acc_temp = acc_temp

        self.use_topk_cover = use_topk_cover
        self.topk_cover_weight = topk_cover_weight
        self.topk_cover_temp = topk_cover_temp
        self.topk_cover_topk = topk_cover_topk
        
        self.use_mse = use_mse
        self.mse_weight = mse_weight
        
        self.image_latent_width = image_latent_width
        self.image_latent_height = image_latent_height
        self.offset = image_latent_width - 1
        
    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels=None, 
        teacher_token=None, 
        teacher_logits=None,
        teacher_hidden_states=None,
    ):    
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=self.use_mse ,
        )

        logits = outputs.logits
        loss = 0.0
        output_dict = {}
        output_dict["logits"] = logits
        student_logits = logits
        student_hidden_states = outputs.hidden_states[-1] if self.use_mse else None
        
        
        if labels is not None:
            if self.use_ce:
                shift_logits = logits.view(-1, logits.shape[-1])
                shift_labels = labels.view(-1).long()

                loss_fn = CrossEntropyLoss()
                ce_loss = loss_fn(shift_logits, shift_labels)
                loss = loss + self.ce_weight * ce_loss
                output_dict["ce_loss"] = ce_loss.detach()
            
            needs_teacher_logits = (
                self.use_kd or 
                self.use_tv or 
                self.use_acc or 
                self.use_topk_cover
                ) and teacher_logits is None
            needs_teacher_hidden = self.use_mse and teacher_hidden_states is None
            
            if needs_teacher_logits or needs_teacher_hidden:
                # infer logits online
                was_training = self.base_model.training
                self.base_model.eval()
                with self.base_model.disable_adapter():
                    with torch.no_grad():
                        teacher_outputs = self.base_model(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            use_cache=False,
                            output_hidden_states=self.use_mse,
                        )
                        teacher_logits = teacher_outputs.logits
                        if self.use_mse:
                            teacher_hidden_states = teacher_outputs.hidden_states[-1]
                if was_training:
                    self.base_model.train()
                    
                if self.offset > 0:
                    aligned_student_logits = student_logits
                    pad_logits = torch.zeros_like(teacher_logits[:, :self.offset, :])
                    aligned_teacher_logits = torch.cat([teacher_logits[:, self.offset:, :], pad_logits], dim=1)
                    
                    if self.use_mse:
                        aligned_student_hidden = student_hidden_states
                        pad_hidden = torch.zeros_like(teacher_hidden_states[:, :self.offset, :])
                        aligned_teacher_hidden = torch.cat([teacher_hidden_states[:, self.offset:, :], pad_hidden], dim=1)
                        
            # not sparse kd anymore
            if self.use_kd:
                invalid = -100
                valid_mask = (labels != invalid)
                student_logits = aligned_student_logits[valid_mask]
                teacher_logits = aligned_teacher_logits[valid_mask]
                # s_logits_v = logits.view(-1, logits.size(-1))[valid_mask]
                # t_indices_v = teacher_token.view(-1, teacher_token.size(-1))[valid_mask].long()
                # t_values_v = teacher_logits.view(-1, teacher_logits.size(-1))[valid_mask]
                # student_topk_logits = torch.gather(s_logits_v, dim=-1, index=t_indices_v)

                T = float(self.kd_temp)
                log_p_student = F.log_softmax(student_logits / T, dim=-1)
                p_teacher = F.softmax(teacher_logits / T, dim=-1)

                kd_loss = F.kl_div(
                    log_p_student,
                    p_teacher,
                    reduction="batchmean",
                ) * (T * T)
                output_dict["kd_loss"] = kd_loss.detach()
                loss = loss + self.kd_weight * kd_loss
                # kd_self = F.kl_div(
                #     p_teacher,
                #     p_teacher,
                #     reduction="batchmean",
                # ) * (T * T)
                # print("kd_loss: ", kd_loss.item(), "kd_self: ", kd_self.item())
        
        
            if self.use_tv:
                invalid = -100
                valid_mask = (labels != invalid)
                student_logits = aligned_student_logits[valid_mask]
                teacher_logits = aligned_teacher_logits[valid_mask]
                
                T = float(self.tv_temp)
                p_student = F.softmax(student_logits / T, dim=-1)
                p_teacher = F.softmax(teacher_logits / T, dim=-1)

                tv_loss = 0.5 * torch.abs(p_student - p_teacher).sum(dim=-1).mean()
                # tv_self = 0.5 * torch.abs(p_teacher - p_teacher).sum(dim=-1).mean()
                output_dict["tv_loss"] = tv_loss.detach()
                loss = loss + self.tv_weight * tv_loss
                # print("tv_loss: ", tv_loss.item(), "tv_self: ", tv_self.item())

            if self.use_acc:
                invalid = -100
                valid_mask = (labels != invalid)
                student_logits = aligned_student_logits[valid_mask]
                teacher_logits = aligned_teacher_logits[valid_mask]
                
                T = float(self.acc_temp)
                p_student = F.softmax(student_logits / T, dim=-1)
                log_p_teacher = F.log_softmax(teacher_logits / T, dim=-1)

                acc_loss = -(p_student * log_p_student).sum(dim=1).mean()
                output_dict["acc_loss"] = acc_loss.detach()
                loss = loss + self.acc_weight * acc_loss

            if self.use_topk_cover:
                invalid = -100
                valid_mask = (labels != invalid)
                student_logits = aligned_student_logits[valid_mask]
                teacher_logits = aligned_teacher_logits[valid_mask]
                
                T =float(self.topk_cover_temp)
                topk_idx = torch.topk(teacher_logits, k=self.topk_cover_topk, dim=-1).indices
                p_student = F.softmax(student_logits / T, dim=-1)
                p_student_topk = torch.gather(p_student, dim=-1, index=topk_idx).sum(dim=-1)

                topk_cover_loss = -torch.log(p_student_topk.clamp_min(1e-8)).mean()
                output_dict["tpkc_loss"] = topk_cover_loss.detach()
                loss = loss + self.topk_cover_weight * topk_cover_loss
            
        
            if self.use_mse:
                invalid = -100
                valid_mask = (labels != invalid)
                
                s_hidden_v = aligned_student_hidden[valid_mask]
                t_hidden_v = aligned_teacher_hidden[valid_mask]
                
                mse_loss = F.mse_loss(s_hidden_v, t_hidden_v)
                # mse_loss = F.smooth_l1_loss(s_hidden_v, t_hidden_v, beta=1.0)
                output_dict["mse_loss"] = mse_loss.detach()
                loss = loss + self.mse_weight * mse_loss


        output_dict["loss"] = loss
        return output_dict