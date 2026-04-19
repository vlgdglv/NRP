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
        use_gumbel: bool = False,
        gumbel_weight: float = 1.0,
        gumbel_tau: float = 1.0,
        # Two-stage repairable draft training
        refine_mode: str = "none",  # "none", "deterministic_soft_topk", "soft_gumbel", "straight_through_hard"
        refine_weight: float = 0.1,
        refine_tau: float = 1.0,
        refine_topk: int = 128,
        refine_full_sequence: bool = False,
        use_topk_mass: bool = False,
        topk_mass_topk: int = 64,
        topk_mass_weight: float = 1.0,
        use_row_rel: bool = False,
        row_rel_weight: float = 1.0,

        use_runtime_vocab_mask: bool = False,
        runtime_vocab_mask_mode: str | None = None,
        lumina_image_token_min: int = 4,
        lumina_image_token_max: int = 8195,
        lumina_eol_token_id: int = 8803,
        lumina_eoi_token_id: int = 8196,
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
        self.use_acc_v0 = False
        
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
        
        self.use_gumbel = use_gumbel
        self.gumbel_weight = gumbel_weight
        self.gumbel_tau = gumbel_tau

        # Two-stage repairable draft: backward compat
        if use_gumbel and refine_mode == "none":
            refine_mode = "soft_gumbel"
            refine_weight = gumbel_weight
            refine_tau = gumbel_tau
        self.refine_mode = refine_mode
        self.refine_weight = refine_weight
        self.refine_tau = refine_tau
        self.refine_topk = refine_topk
        self.refine_full_sequence = refine_full_sequence

        self.use_topk_mass = use_topk_mass
        self.topk_mass_topk = topk_mass_topk
        self.topk_mass_weight = topk_mass_weight 

        self.use_row_rel = use_row_rel
        self.row_rel_weight = row_rel_weight

        self.use_runtime_vocab_mask = use_runtime_vocab_mask
        self.runtime_vocab_mask_mode = runtime_vocab_mask_mode

        self.lumina_image_token_min = lumina_image_token_min
        self.lumina_image_token_max = lumina_image_token_max
        self.lumina_eol_token_id = lumina_eol_token_id
        self.lumina_eoi_token_id = lumina_eoi_token_id

    def save_pretrained(self, output_dir):
        self.base_model.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls,
                       checkpoint_path,
                       base_model,
                       **kwargs):
        """
        Create RowExpertModel from a previously saved checkpoint.

        Args:
            checkpoint_path: Path to the saved RowExpertModel checkpoint
            base_model: Base model with LoRA already loaded
            **kwargs: Additional arguments for RowExpertModel initialization
        """
        # For now, just create a new instance with the provided base_model
        # Future enhancement: could save/load loss function configurations
        return cls(base_model=base_model, **kwargs)

    def load_lora_checkpoint(self, checkpoint_path, strict=True):
        """
        Load LoRA weights from checkpoint into existing model.

        Args:
            checkpoint_path: Path to LoRA checkpoint
            strict: Whether to enforce strict loading
        """
        try:
            from peft import PeftModel
            # This would replace the current LoRA weights
            # Note: This method assumes base_model is already a PeftModel
            if hasattr(self.base_model, 'load_adapter'):
                self.base_model.load_adapter(checkpoint_path, "default", is_trainable=True)
            else:
                raise ValueError("Base model is not a PEFT model - cannot load LoRA checkpoint")
        except Exception as e:
            if strict:
                raise e
            else:
                print(f"Warning: Failed to load LoRA checkpoint: {e}")

    def _compute_draft_probs(self, logits):
        """Convert student logits to a soft probability distribution for draft embedding."""
        if self.refine_mode == "deterministic_soft_topk":
            p = F.softmax(logits / self.refine_tau, dim=-1)
            topk_vals, topk_idx = p.topk(self.refine_topk, dim=-1)
            p_masked = torch.zeros_like(p).scatter_(-1, topk_idx, topk_vals)
            return p_masked / p_masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        elif self.refine_mode == "soft_gumbel":
            return F.gumbel_softmax(logits / self.refine_tau, tau=1.0, hard=False)
        elif self.refine_mode == "straight_through_hard":
            return F.gumbel_softmax(logits / self.refine_tau, tau=1.0, hard=True)
        else:
            raise ValueError(f"Unknown refine_mode: {self.refine_mode}")

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
            output_hidden_states=True,
        )

        logits = outputs.logits
        loss = 0.0
        output_dict = {}
        output_dict["logits"] = logits
        student_logits = logits
        student_hidden_states = outputs.hidden_states[-1]
        
        
        if labels is not None:
            if False:
                valid_mask = (labels != -100)
                logits_v = logits[valid_mask]  # [N, V]
                prob = F.softmax(logits_v, dim=-1)
                # lumina visual token range: [4, 8195]
                visual_prob_mass = prob[:, :8199].sum(dim=-1)  # [N]

                output_dict["teacher_visual_mass_mean"] = visual_prob_mass.mean()
                output_dict["teacher_visual_mass_min"] = visual_prob_mass.min()
                output_dict["teacher_visual_mass_p10"] = visual_prob_mass.quantile(0.1)
                print(output_dict["teacher_visual_mass_mean"], output_dict["teacher_visual_mass_min"], output_dict["teacher_visual_mass_p10"] )
                
            if self.use_ce:
                shift_logits = logits.view(-1, logits.shape[-1])
                shift_labels = labels.view(-1).long()

                loss_fn = CrossEntropyLoss(ignore_index=-100)
                ce_loss = loss_fn(shift_logits, shift_labels)
                loss = loss + self.ce_weight * ce_loss
                output_dict["ce_loss"] = ce_loss.detach()
                
            if self.refine_mode != "none":
                B, L = input_ids.shape
                device = input_ids.device
                W = self.image_latent_width
                H = self.image_latent_height

                # Draft probabilities from student logits
                draft_probs = self._compute_draft_probs(student_logits)  # [B, L, V_student]

                # Embed draft probs: match the inference embedding pathway
                # Janus: token → gen_embed(16384,8) → gen_aligner(8→2048)
                #   so soft draft: draft_probs @ gen_embed.weight → gen_aligner(...)
                # Lumina: token → embed_tokens(65536,4096), unified vocab
                #   so soft draft: draft_probs @ embed_tokens.weight
                if hasattr(self.base_model, 'gen_embed') and hasattr(self.base_model, 'gen_aligner'):
                    # Janus path: gen_embed → gen_aligner
                    raw_embeds = draft_probs @ self.base_model.gen_embed.weight  # [B, L, 8]
                    all_draft_embeds = self.base_model.gen_aligner(raw_embeds)    # [B, L, 2048]
                    with torch.no_grad():
                        real_embeds = self.base_model.language_model.get_input_embeddings()(input_ids)  # [B, L, D]
                else:
                    # Lumina/unified path: LLM input embeddings
                    V_student = draft_probs.shape[-1]
                    embed_weight = self.base_model.get_input_embeddings().weight.to(device=draft_probs.device, dtype=draft_probs.dtype) # [V_full, D]
                    all_draft_embeds = draft_probs @ embed_weight[:V_student]     # [B, L, D]
                    with torch.no_grad():
                        real_embeds = self.base_model.get_input_embeddings()(input_ids)  # [B, L, D]
                if self.refine_full_sequence:
                    # Experiment A baseline: replace ALL positions (broken behavior)
                    hybrid_embeds = all_draft_embeds
                else:
                    # Correct: real embeddings for prefix, draft only at target rows
                    # For Janus: real prefix uses LLM embeddings (same as training forward)
                    # Draft rows use gen_embed+gen_aligner (same as inference)

                    # Build shifted draft: student logits at row r → embedding for row r+1
                    shifted_draft = torch.zeros_like(real_embeds)
                    draft_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
                    for b in range(B):
                        valid_pos = (labels[b] != -100).nonzero(as_tuple=False)
                        if valid_pos.numel() == 0:
                            continue
                        img_begin = valid_pos[0].item()
                        for r in range(H - 1):
                            src_s = img_begin + r * W
                            tgt_s = img_begin + (r + 1) * W
                            shifted_draft[b, tgt_s:tgt_s + W] = all_draft_embeds[b, src_s:src_s + W]
                            draft_mask[b, tgt_s:tgt_s + W] = True

                    m = draft_mask.unsqueeze(-1)  # [B, L, 1]
                    hybrid_embeds = torch.where(m, shifted_draft, real_embeds)

                # Refine forward: frozen base model on hybrid input
                was_training = self.base_model.training
                self.base_model.eval()
                with self.base_model.disable_adapter():
                    refine_outputs = self.base_model(
                        inputs_embeds=hybrid_embeds,
                        attention_mask=None,
                        use_cache=False,
                        output_hidden_states=False,
                    )
                    refine_logits = refine_outputs.logits
                if was_training:
                    self.base_model.train()

                valid_mask = (labels != -100)
                refine_loss = F.cross_entropy(
                    refine_logits[valid_mask], labels[valid_mask].long()
                )
                output_dict["refine_loss"] = refine_loss.detach()
                loss = loss + self.refine_weight * refine_loss
            
            needs_teacher_logits = (
                self.use_kd or 
                self.use_tv or 
                self.use_acc or 
                self.use_topk_cover or
                self.use_topk_mass or 
                self.use_row_rel
                ) and teacher_logits is None
            needs_teacher_hidden = (
                self.use_mse or
                self.use_row_rel
                ) and teacher_hidden_states is None
            
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
                            output_hidden_states=needs_teacher_hidden,
                        )
                        teacher_logits = teacher_outputs.logits
                        if needs_teacher_hidden:
                            teacher_hidden_states = teacher_outputs.hidden_states[-1]
                if was_training:
                    self.base_model.train()
                    
                if self.offset > 0:
                    aligned_student_logits = student_logits
                    pad_logits = torch.zeros_like(teacher_logits[:, :self.offset, :])
                    aligned_teacher_logits = torch.cat([teacher_logits[:, self.offset:, :], pad_logits], dim=1)
                    
                    if needs_teacher_hidden:
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

            if self.use_acc_v0:
                invalid = -100
                valid_mask = (labels != invalid)
                student_logits = aligned_student_logits[valid_mask]
                teacher_logits = aligned_teacher_logits[valid_mask]
                
                T = float(self.acc_temp)
                p_student = F.softmax(student_logits / T, dim=-1)

                acc_loss = -(p_student * log_p_student).sum(dim=1).mean()
                output_dict["acc_loss"] = acc_loss.detach()
                loss = loss + self.acc_weight * acc_loss

            if self.use_acc:
                invalid = -100
                valid_mask = (labels != invalid)
                T = float(self.acc_temp)
                
                student_prob = F.softmax(aligned_student_logits / T, dim=-1)
                teacher_logprob = F.log_softmax(aligned_teacher_logits / T, dim=-1)

                acc_per_pos = -(student_prob * teacher_logprob).sum(dim=-1)
                acc_loss = acc_per_pos[valid_mask].mean()

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

            if self.use_topk_mass:
                invalid = -100
                valid_mask = (labels != invalid)
                k = self.topk_mass_topk
                T = float(getattr(self, "topk_mass_temp", 1.0))
                teacher_logits = aligned_teacher_logits.detach()
                student_logits = aligned_student_logits
                # Strongly recommended: mask non-visual tokens before topk/logsoftmax
                # teacher_logits = mask_non_visual(teacher_logits)
                # student_logits = mask_non_visual(student_logits)
                teacher_topk_logits, teacher_topk_idx = torch.topk(teacher_logits, k=k, dim=-1)

                # teacher_topk_prob = F.softmax(teacher_topk_logits / T, dim=-1).detach()
                teacher_global_prob = F.softmax(teacher_logits / T, dim=-1).detach()
                teacher_topk_prob = torch.gather(teacher_global_prob, dim=-1, index=teacher_topk_idx)
                
                student_logprob = F.log_softmax(student_logits / T, dim=-1)
                student_topk_logprob = torch.gather(student_logprob, dim=-1, index=teacher_topk_idx)

                topk_ce_per_pos = -(teacher_topk_prob * student_topk_logprob).sum(dim=-1)
                topk_mass_loss = topk_ce_per_pos[valid_mask].mean()

                with torch.no_grad():
                    student_topk_mass = torch.logsumexp(student_topk_logprob, dim=-1).exp()
                    student_top1 = student_logits.argmax(dim=-1, keepdim=True)
                    student_top1_in_teacher_topk = (student_top1 == teacher_topk_idx).any(dim=-1)

                output_dict["topk_mass_loss"] = topk_mass_loss.detach()
                output_dict[f"student_mass_on_teacher_top{k}"] = (
                    student_topk_mass[valid_mask].mean().detach()
                )
                output_dict[f"student_top1_in_teacher_top{k}"] = (
                    student_top1_in_teacher_topk[valid_mask].float().mean().detach()
                )

                loss = loss + self.topk_mass_weight * topk_mass_loss
            
            # if self.use_topk_mass:
            #     invalid = -100
            #     valid_mask = (labels != invalid)
            #     k = self.topk_mass_topk
            #     T = float(getattr(self, "topk_mass_temp", 1.0))

            #     teacher_topk_idx = aligned_teacher_logits.topk(k, dim=-1).indices

            #     student_logprob = F.log_softmax(aligned_student_logits, dim=-1)
            #     student_topk_logprob = torch.gather(
            #         student_logprob, dim=-1, index=teacher_topk_idx,
            #     ) #[B, T, K]
            #     topk_mass_logprob = torch.logsumexp(student_topk_logprob, dim=-1)
            #     topk_mass_loss = -topk_mass_logprob[valid_mask].mean()

            #     output_dict["topk_mass_loss"] = topk_mass_loss.detach()
            #     output_dict[f"student_mass_on_teacher_top{k}"] = topk_mass_logprob[valid_mask].exp().mean().detach()

            #     teacher_topk_idx = aligned_teacher_logits.topk(k, dim=-1).indices
            #     teacher_topk_logits, teacher_topk_idx = torch.topk(aligned_teacher_logits / T, k=k, dim=-1)

            #     teacher_topk_prob = F.softmax(teacher_topk_logits / T, dim=-1)

            #     student_logprob = F.log_softmax(aligned_student_logits / T, dim=-1)
            #     student_topk_logprob = torch.gather(
            #         student_logprob, dim=-1, index=teacher_topk_idx,
            #     ) #[B, T, K]
                
            #     topk_mass_per_pos = -(teacher_topk_prob * student_topk_logprob).sum(dim=-1)
            #     topk_mass_loss = topk_mass_per_pos[valid_mask].mean()

            #     # No weighted:
            #     # with torch.no_grad():
            #     #     topk_mass_logprob = torch.logsumexp(student_topk_logprob, dim=-1)
            #     #     output_dict[f"student_mass_on_teacher_top{k}"] = topk_mass_logprob[valid_mask].exp().mean().detach()

            #     output_dict["topk_mass_loss"] = topk_mass_loss.detach()
            #     output_dict[f"student_mass_on_teacher_top{k}"] = topk_mass_per_pos[valid_mask].exp().mean().detach()
                
            #     loss = loss + self.topk_mass_weight * topk_mass_loss
            
            if self.use_mse:
                invalid = -100
                valid_mask = (labels != invalid)
                
                s_hidden_v = aligned_student_hidden[valid_mask]
                t_hidden_v = aligned_teacher_hidden[valid_mask]
                
                mse_loss = F.mse_loss(s_hidden_v, t_hidden_v)
                # mse_loss = F.smooth_l1_loss(s_hidden_v, t_hidden_v, beta=1.0)
                output_dict["mse_loss"] = mse_loss.detach()
                loss = loss + self.mse_weight * mse_loss

            if self.use_row_rel:
                assert target_row_ids is not None
                assert row_valid_mask is not None

                rel_losses = []
                B = input_ids.shape[0]
                for b in range(B):
                    valid_rows = target_row_ids[b][row_valid_mask[b]].unique()
                    valid_rows = valid_rows[valid_rows>=0]

                    for rid in valid_rows.tolist():
                        row_mask_b = row_valid_mask[b] & (target_row_ids[b] == rid)

                        if row_mask_b.sum() < 2: continue

                        hs = aligned_student_hidden[b][row_mask_b]
                        ht = aligned_teacher_hidden[b][row_mask_b]

                        hs = F.normalize(hs, p=2, dim=-1)
                        ht = F.normalize(ht, p=2, dim=-1)

                        Rs = hs @ hs.T
                        Rt = ht @ ht.T

                        rel_losses.append(F.mse_loss(Rs, Rt))
                if len(rel_losses) > 0:
                    row_rel_loss = torch.stack(rel_losses).mean()
                else:
                    row_rel_loss = aligned_student_logits.new_zeros(())
                
                output_dict["row_rel_loss"] = row_rel_loss.detach()
                loss = loss + self.row_rel_weight * row_rel_loss


        output_dict["loss"] = loss
        return output_dict

    def _maybe_apply_runtime_vocab_mask(self, logits, target_ids):
        if (not self.use_runtime_vocab_mask) or (self.runtime_vocab_mask_mode is None):
            return logits
        if target_ids is None:
            return logits

        if self.runtime_vocab_mask_mode == "lumina":
            return self._apply_lumina_runtime_vocab_mask(logits, target_ids)

        raise ValueError(f"Unknown runtime_vocab_mask_mode: {self.runtime_vocab_mask_mode}")


    def _apply_lumina_runtime_vocab_mask(self, logits, target_ids):
        """
        Mimic Lumina runtime logits processor at training time.

        Rules:
        1) normal image positions: only allow visual codebook ids [4, 8195]
        2) EOL positions: force eol token
        3) EOI positions: force eoi token

        logits: [N, V] or [B, T, V]
        target_ids: [N] or [B, T]
        """
        if logits.dim() not in (2, 3):
            raise ValueError(f"logits must be 2D or 3D, got shape={logits.shape}")

        orig_shape = logits.shape
        vocab_size = orig_shape[-1]

        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = target_ids.reshape(-1)

        masked = flat_logits.clone()
        neg_large = torch.finfo(masked.dtype).min

        # ignore_index positions: leave unchanged
        ignore_mask = flat_targets == -100

        # 1) normal visual token positions
        visual_pos_mask = (
            (flat_targets >= self.lumina_image_token_min) &
            (flat_targets <= self.lumina_image_token_max)
        ) & (~ignore_mask)

        if visual_pos_mask.any():
            if self.lumina_image_token_min > 0:
                masked[visual_pos_mask, :self.lumina_image_token_min] = neg_large
            if self.lumina_image_token_max + 1 < vocab_size:
                masked[visual_pos_mask, self.lumina_image_token_max + 1:] = neg_large

        # # 2) force EOL
        # eol_pos_mask = (flat_targets == self.lumina_eol_token_id)
        # if eol_pos_mask.any():
        #     masked[eol_pos_mask, :] = neg_large
        #     masked[eol_pos_mask, self.lumina_eol_token_id] = 0.0

        # # 3) force EOI
        # eoi_pos_mask = (flat_targets == self.lumina_eoi_token_id)
        # if eoi_pos_mask.any():
        #     masked[eoi_pos_mask, :] = neg_large
        #     masked[eoi_pos_mask, self.lumina_eoi_token_id] = 0.0

        return masked.view(orig_shape)