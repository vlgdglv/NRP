import torch.nn as nn
from torch.nn import CrossEntropyLoss


class RowExpertModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits.view(-1, logits.shape[-1])
            shift_labels = labels.view(-1)

            loss_fn = CrossEntropyLoss()
            loss = loss_fn(shift_logits, shift_labels)
        
        return {"loss": loss, "logits": logits}