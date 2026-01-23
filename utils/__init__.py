import os

def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0

def rollback_kv_cache(past_key_value, rollback_length):
    for layer_indx in range(len(past_key_value.key_cache)):
        past_key_value.key_cache[layer_indx] = past_key_value.key_cache[layer_indx][..., :-rollback_length, :]
        past_key_value.value_cache[layer_indx] = past_key_value.value_cache[layer_indx][..., :-rollback_length, :]