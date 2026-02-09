import os
import torch

def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def rollback_kv_cache(past_key_value, rollback_length):
    for layer_indx in range(len(past_key_value.key_cache)):
        past_key_value.key_cache[layer_indx] = past_key_value.key_cache[layer_indx][..., :-rollback_length, :]
        past_key_value.value_cache[layer_indx] = past_key_value.value_cache[layer_indx][..., :-rollback_length, :]


def snapshot_kv_cache(past_key_value, idx: int, inclusive: bool = False):
    if idx < 0:
        raise ValueError(f"idx must be >= 0, got {idx}")

    prefix_len = idx + 1 if inclusive else idx
    new_cache = past_key_value.__class__()  # DynamicCache() in most cases

    if not hasattr(new_cache, "key_cache"):
        new_cache.key_cache = []
    if not hasattr(new_cache, "value_cache"):
        new_cache.value_cache = []

    new_cache.key_cache = []
    new_cache.value_cache = []

    for layer_indx in range(len(past_key_value.key_cache)):
        k = past_key_value.key_cache[layer_indx]
        v = past_key_value.value_cache[layer_indx]

        seq_len = k.shape[-2]
        if prefix_len > seq_len:
            raise ValueError(
                f"prefix_len ({prefix_len}) > seq_len ({seq_len}) at layer {layer_indx}"
            )
        new_cache.key_cache.append(k[..., :prefix_len, :].clone())
        new_cache.value_cache.append(v[..., :prefix_len, :].clone())

    # if hasattr(new_cache, "seen_tokens"):
    #     try:
    #         new_cache.seen_tokens = prefix_len
    #     except Exception:
    #         pass
    # if hasattr(new_cache, "_seen_tokens"):
    #     try:
    #         new_cache._seen_tokens = prefix_len
    #     except Exception:
    #         pass

    return new_cache