import torch


def is_janus(base_model):
    """Detect Janus by presence of prepare_gen_img_embeds (works through PEFT wrapping)."""
    return hasattr(base_model, "prepare_gen_img_embeds")


def build_inputs_embeds(base_model, input_ids, image_mask=None):
    """Build the correct training-time input embeddings.

    - Janus: text positions go through `language_model.get_input_embeddings()`,
      image positions go through `prepare_gen_img_embeds = gen_aligner(gen_embed(.))`.
      This is the SAME pathway used at inference; routing image VQ codes through the
      LLM embedding table is wrong (verified empirically: CE 11.6 vs 1.05 on greedy
      teacher-forced image tokens).
    - Lumina (Chameleon): unified vocab — `base_model.get_input_embeddings()` is correct
      for both text and image tokens. `image_mask` is ignored.

    Args:
        base_model: PEFT or raw model
        input_ids: [B, L] int
        image_mask: [B, L] bool, true at positions holding image VQ codes (Janus only)

    Returns:
        inputs_embeds: [B, L, D]
    """
    if is_janus(base_model):
        text_embed = base_model.language_model.get_input_embeddings()
        embeds = text_embed(input_ids)
        if image_mask is not None and image_mask.any():
            img_ids = input_ids[image_mask]
            img_embeds = base_model.prepare_gen_img_embeds(img_ids).to(embeds.dtype)
            embeds = embeds.clone()
            embeds[image_mask] = img_embeds
        return embeds
    return base_model.get_input_embeddings()(input_ids)


def build_glancing_inputs(base_model, input_ids, labels, reveal_ratio, image_mask=None):
    """GLAT: with probability `reveal_ratio`, replace input embeddings at target
    positions with the embedding of the *label* token at that position. Bidirectional
    within-row attention then lets unrevealed positions glance at the revealed targets.

    Returns:
        hybrid_embeds: [B, L, D]
        glancing_labels: [B, L] — labels at revealed positions set to -100 (no CE on them)
    """
    input_embeds = build_inputs_embeds(base_model, input_ids, image_mask)

    if reveal_ratio <= 0.0:
        return input_embeds, labels.clone()

    valid = (labels != -100)
    rand = torch.rand(valid.shape, device=valid.device, dtype=torch.float32)
    reveal_mask = valid & (rand < reveal_ratio)

    if not reveal_mask.any():
        return input_embeds, labels.clone()

    label_safe = labels.clamp(min=0)
    if is_janus(base_model):
        # Labels at image positions are image VQ codes -> route through gen_embed/gen_aligner
        label_embeds_full = base_model.prepare_gen_img_embeds(label_safe).to(input_embeds.dtype)
    else:
        label_embeds_full = base_model.get_input_embeddings()(label_safe).to(input_embeds.dtype)

    m = reveal_mask.unsqueeze(-1)
    hybrid_embeds = torch.where(m, label_embeds_full, input_embeds)

    glancing_labels = labels.clone()
    glancing_labels[reveal_mask] = -100

    return hybrid_embeds, glancing_labels
