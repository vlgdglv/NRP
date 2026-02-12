
def check_image_for_lumina(batch, W=49, H=48):
    img_length = W * H
    for sample in batch:
        token_ids = sample["input_ids"]
        assert token_ids[-1] == 8710    # end of sequence
        assert token_ids[-2] == 8196    # end of image
        prompt_len = len(token_ids) - 2 - 3 - img_length
        assert token_ids[prompt_len-1] == 8710 # end of sequence (text prompt end)
        assert token_ids[prompt_len] == 8197 # start of image
        for idx in range(prompt_len+2, prompt_len+2+img_length, W):
            assert token_ids[idx+W] == 8803 # 


def check_image_for_emu3(batch, W=91, H=90):
    img_length = W * H
    for sample in batch:
        token_ids = sample["input_ids"]
        assert token_ids[-1] == 151853    # end of sequence
        assert token_ids[-2] == 151847    # end of image
        prompt_len = len(token_ids) - 2 - img_length
        assert token_ids[prompt_len-1] == 151851 # image token
        for idx in range(prompt_len-1, prompt_len-1+img_length, W):
            assert token_ids[idx+W] == 151846 # eol

def check_image_for_janus(batch, W=24, H=24):
    return True

lumina_img_token_config = {
    "eoi_token_id": 8196,
    "boi_token_id": 8197,
    "eol_token_id": 8803,
    "eos_token_id": 8710,
    "img_token_id": -1,
    "token_check_func": check_image_for_lumina
}

emu3_img_token_config = {
    "eoi_token_id": 151847,
    "boi_token_id": 151852,
    "eol_token_id": 151846,
    "eos_token_id": 151853,
    "img_token_id": -1,
    "token_check_func": check_image_for_emu3
}

janus_img_token_config = {
    "eoi_token_id": 151847,
    "boi_token_id": 151852,
    "eol_token_id": 151846,
    "eos_token_id": 151853,
    "img_token_id": -1,
    "token_check_func": check_image_for_janus
}

