import os
import copy
import json
import logging
import random
from typing import Any, Callable, Dict, List, Tuple, Union

from PIL import Image
import torch
from abc import ABC, abstractmethod
import logging

import model.lumina_arch.chameleon_vae as chameleon_vae
from .convertsation import Conversation
from .data_reader import read_general
from .tokenizer import Tokenizer


logger = logging.getLogger(__name__)



def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


class LabelAllZeroError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"LabelAllZeroError: {self.message}"


class ItemProcessorBase(ABC):
    @abstractmethod
    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        raise NotImplementedError

    def predict_item_token_length(self, data_item: dict) -> int:
        """
        estimate the token length of the data item for gathering items of similar lengths into a batch
        """
        return 1


class MMConvItemProcessor(ItemProcessorBase):
    def __init__(
        self,
        transform: Dict[str, Callable[[Any], Dict]],
        media_symbols: List[str],
        tokenizer: str | Tokenizer,
        conv_template,
    ):
        self.transform = transform
        logger.info(f"transform:\n{self.transform}")

        self.media_symbols = media_symbols
        logger.info(f"media_symbols:\n{self.media_symbols}")

        if isinstance(tokenizer, str):
            self.tokenizer = Tokenizer(model_path=tokenizer)
        else:
            self.tokenizer = copy.deepcopy(tokenizer)

        # todo should not already exist
        self.tokenizer.tokenizer.add_tokens(media_symbols)
        self.d_media_symbol2token = {}
        self.d_media_token2symbol = {}
        for media_symbol in media_symbols:
            tokenized_symbol = self.tokenizer.encode(media_symbol, bos=False, eos=False)
            assert len(tokenized_symbol) == 1
            self.d_media_symbol2token[media_symbol] = tokenized_symbol[0]
            self.d_media_token2symbol[tokenized_symbol[0]] = media_symbol

        # implicit_at_beginning means media without explict location specification are arranged right after bos token
        # if false, then these medias are arranged at the beginning of the first question
        self.implicit_at_beginning = False
        self.conv_template = conv_template

    def collect_and_process_media(self, data_item):
        """
        this function receives a raw piece of data (e.g. read from `.json` data file),
        and returns d_media, containing the prepared media readily usable by model
        YOU MAY OVERRIDE THIS FUNCTION TO SUPPORT COMPLEX LOADING OF VARIOUS FORMS OF DATA
        """
        d_media = {}
        for media_symbol in self.media_symbols:
            if media_symbol in data_item:
                l_media = data_item[media_symbol]  # a list of media paths
            elif media_symbol.lstrip("<|").rstrip("|>") in data_item:
                l_media = data_item[media_symbol.lstrip("<|").rstrip("|>")]
            else:
                l_media = []
            if not isinstance(l_media, list):  # data with only one media, in format {"image": image_name, ...}
                l_media = [l_media]

            d_media[media_symbol] = []
            for media in l_media:
                media = self.transform[media_symbol](media)
                assert isinstance(media, Dict)
                media["type"] = media_symbol
                d_media[media_symbol].append(media)

        return d_media

    def replace_media_token_with_media(
        self, tokens: List[int], labels: Union[List[int], None], d_media: Dict[str, List]
    ):
        d_media_counter = {key: 0 for key in d_media}
        for i, t in enumerate(tokens):
            if t in self.d_media_token2symbol:
                media_symbol = self.d_media_token2symbol[t]
                media = d_media[media_symbol][d_media_counter[media_symbol]]
                d_media_counter[media_symbol] += 1
                tokens[i] = media
                media["to_predict"] = labels[i] > 0

        assert all([d_media_counter[key] == len(d_media[key]) for key in d_media])

        if labels is not None:
            return tokens, labels
        else:
            return tokens

    @staticmethod
    def insert_implicit_media_symbol_in_q1(conv_list: List[Dict], d_media: Dict):
        """
        Add the media tokens to the beginning of the first instruction from
        human. This logic may be more reasonable. However, it is incompatible
        with old-version Accessory models, which are trained with image tokens
        inserted directly behind the first token (<bos>).
        :param conv_list: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :param d_media: a dict of media for all media types
        """
        conv_list = copy.deepcopy(conv_list)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = "".join([_["value"] for _ in conv_list if _["value"] is not None]).count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(
                    l_media
                ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv_list[0]["value"] = (media_symbol + " ") * len(l_media) + conv_list[0]["value"]

        return conv_list

    @staticmethod
    def insert_implicit_media_symbol_at_beginning(conv: str, d_media: Dict):
        """
        Legacy versions of LLaMA2-Accessory handled media in a non-interleaved
        manner, where image tokens are inserted directly behind the first token,
        namely <bos>. To support interleaved media comprehension and generation,
        Accessory now supports the explicit specification of media occurrence,
        which is achieved by adding media symbols, e.g. <image>, within the
        conversations. On the other hand, for media without explicit
        specification, this function realizes the legacy behavior to arrange
        them at the beginning of the conversation.
        :param conv: conversation
        :param d_media: a dict of media for all media types, for determining how
        many media tokens need to be inserted
        """
        conv = copy.deepcopy(conv)

        for media_symbol, l_media in d_media.items():
            media_symbol_count = conv.count(media_symbol)
            if media_symbol_count > 0:
                assert media_symbol_count == len(
                    l_media
                ), f"{media_symbol_count} {media_symbol} exists in text, but {len(l_media)} actual media are given"
            else:
                conv = (media_symbol + " ") * len(l_media) + conv

        return conv

    def preprocess_item(self, data_item):
        return data_item

    def add_speaker_and_signal(self, source: List):
        """
        Given source instruction and response pieces, return the text containing the complete conversation,
        and the list of values that the model should learn to predict during training
        :param source: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]
        :return: `conversation`: string containing the complete conversation;
                 `to_predict_list`: the list of values that the model should learn to predict during training
        """
        conv = self.conv_template()

        for i, sentence in enumerate(source):
            from_str = sentence["from"]
            if i % 2 == 0:
                assert from_str.lower() in ["human"]
                role = conv.roles[0]
            elif i % 2 == 1:
                assert from_str.lower() in ["gpt", "assistant"]
                role = conv.roles[1]
            else:
                raise ValueError(f"unknown dialog role: {from_str.lower()}")

            value = sentence["value"]

            conv.append_message(role, value)

        processed = conv.process()
        conversation, pieces = processed["conv"], processed["pieces"]

        return conversation, pieces

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        data_item = self.preprocess_item(data_item)

        d_media = self.collect_and_process_media(data_item)

        source = data_item["conversations"]

        # implicit_at_beginning means media without explict location specification are arranged right after bos token
        # if false, then these medias are arranged at the beginning of the first question
        if not self.implicit_at_beginning:
            source = self.insert_implicit_media_symbol_in_q1(source, d_media)

        conversation, pieces = self.add_speaker_and_signal(source)

        if self.implicit_at_beginning:
            conversation = self.insert_implicit_media_symbol_at_beginning(conversation, d_media)

        # dialog does not need eos
        tokens = self.tokenizer.encode(conversation, bos=True, eos=False)
        labels = [-100 for _ in tokens]

        # check special token num as expected
        for media_symbol, l_media in d_media.items():
            media_token = self.d_media_symbol2token[media_symbol]
            media_token_count = tokens.count(media_token)
            assert media_token_count == len(l_media), (
                f"{media_token_count} {media_token} (for {media_symbol}) exists in tokenized conversation, "
                f"but {len(l_media)} actual media are given"
            )

        check_pos = 0
        for i, p in enumerate(pieces):
            if i == 0:
                tokenized_value = self.tokenizer.encode(p["data"], bos=True, eos=False)
            else:
                tokenized_value = self.tokenizer.encode_wo_prefix_space(p["data"])

            assert (
                tokens[check_pos : check_pos + len(tokenized_value)] == tokenized_value
            ), "inconsistent complete conversation and corresponding piece after tokenization"

            if p["predict"]:
                labels[check_pos : check_pos + len(tokenized_value)] = tokenized_value

            check_pos = check_pos + len(tokenized_value)

        if training_mode and all([_ <= 0 for _ in labels]):  # nothing to predict
            raise LabelAllZeroError()

        # labels will be processed later by the model
        tokens, labels = self.replace_media_token_with_media(tokens, labels, d_media)

        assert len(tokens) == len(labels)

        if training_mode:
            return tokens, labels
        else:
            return tokens

    def predict_item_token_length(self, data_item: dict) -> int:
        """
        estimate the length of each item
        """

        if "conversations" in data_item:
            return sum([len(_["value"]) for _ in data_item["conversations"]])
        else:
            return 1


class FlexARItemProcessor(MMConvItemProcessor):
    image_start_token = "<racm3:break>"  # fixed tokens for start and end, so can hardcode
    image_end_token = "<eoss>"
    full_sub_sep_token = "<reserved08796>"
    sub_sub_sep_token = "<reserved08797>"
    sub_skip_token = "<reserved08798>"
    new_line_token = "<reserved08799>"

    def __init__(
        self,
        tokenizer="Alpha-VLLM/Lumina-mGPT-7B-768",
        conv_template=Conversation,
        target_size=512,
        vae_tokenizer_path="./ckpts/chameleon/tokenizer/text_tokenizer.json"
    ):

        super().__init__(
            {
                "<|image|>": self.process_image,
            },
            ["<|image|>"],
            tokenizer,
            conv_template,
        )

        self.patch_size = 32
        self.crop_size_list = generate_crop_size_list((target_size // self.patch_size) ** 2, self.patch_size)
        logger.info("List of crop sizes:")
        for i in range(0, len(self.crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in self.crop_size_list[i : i + 6]]))

        #  TODO
        #  currently still use the original image tokenizer provided by Meta rather than transformers
        #  because the transformers implementation does not contain the vae decoder
        self.chameleon_ori_vocab = chameleon_vae.VocabInfo(
            json.load(open(os.path.join(vae_tokenizer_path, "text_tokenizer.json"), encoding="utf8"))["model"]["vocab"]
        )
        self.chameleon_ori_translation = chameleon_vae.VocabTranslation(self.chameleon_ori_vocab, device="cuda")
        self.chameleon_ori_image_tokenizer = chameleon_vae.ImageTokenizer(
            cfg_path=os.path.join(vae_tokenizer_path, "vqgan.yaml"),
            ckpt_path=os.path.join(vae_tokenizer_path, "vqgan.ckpt"),
            device="cuda",
        )

    @staticmethod
    def get_n_grids_token(n_grids):
        return f"<reserved{8800 + n_grids:05d}>"

    def token2id(self, token: str) -> int:
        return self.tokenizer.tokenizer.vocab[token]

    @torch.no_grad()
    def process_image(self, image) -> Dict:
        if isinstance(image, Image.Image):
            pass
        else:
            image = Image.open(read_general(image))

        image = var_center_crop(image, crop_size_list=self.crop_size_list)

        w_grids, h_grids = image.size[0] // self.patch_size, image.size[1] // self.patch_size

        image_toks = self.chameleon_ori_translation.convert_img2bp2(
            self.chameleon_ori_image_tokenizer.img_tokens_from_pil(image)
        ).view(-1)

        full_image_toks = image_toks.reshape(image.size[1] // 16, image.size[0] // 16)
        new_line_id = self.token2id(self.new_line_token)

        full_image_toks = torch.cat(
            (
                full_image_toks,
                torch.ones(image.size[1] // 16, 1, device=full_image_toks.device, dtype=full_image_toks.dtype)
                * new_line_id,
            ),
            dim=1,
        ).flatten()

        result_toks = [
            self.token2id(self.image_start_token),
            self.token2id(self.get_n_grids_token(h_grids)),
            self.token2id(self.get_n_grids_token(w_grids)),
            *full_image_toks.tolist(),
            self.token2id(self.image_end_token),
        ]

        return {"input_ids": result_toks, "labels": result_toks}

    def process_item(self, item, training_mode=False, out_flatten=True):
        if not out_flatten:
            return super().process_item(item, training_mode=training_mode)

        if training_mode:
            tokens, labels = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            modified_labels_item = []
            for i, (token_or_media, ori_label) in enumerate(zip(tokens, labels)):
                if isinstance(token_or_media, int):
                    token = token_or_media
                    input_tokens_item.append(token)
                    modified_labels_item.append(ori_label)
                else:
                    input_tokens_item += token_or_media["input_ids"]
                    if ori_label <= 0:  # in the prompt part
                        modified_labels_item += [-100] * len(token_or_media["input_ids"])
                    else:
                        modified_labels_item += token_or_media["labels"]

            return input_tokens_item, modified_labels_item
        else:
            tokens = super().process_item(item, training_mode=training_mode)
            input_tokens_item = []
            for i, token_or_media in enumerate(tokens):
                if isinstance(token_or_media, int):
                    input_tokens_item.append(token_or_media)
                else:
                    input_tokens_item += token_or_media["input_ids"]

            return input_tokens_item

    def decode_image(self, tokens: List[int]) -> Image.Image:
        if tokens[0] == self.token2id(self.image_start_token):
            tokens = tokens[1:]
        if tokens[-1] == self.token2id(self.image_end_token):
            tokens = tokens[:-1]

        h_grids, w_grids = tokens[0] - 8804, tokens[1] - 8804
        tokens = tokens[2:]
        h, w = h_grids * self.patch_size, w_grids * self.patch_size
        h_latent_dim, w_latent_dim = h_grids * 2, w_grids * 2

        for i in range(len(tokens)):
            if (i + 1) % (w_latent_dim + 1) != 0:
                tokens[i] = self.chameleon_ori_translation.bpe2img[tokens[i]]

        assert len(tokens) == h_latent_dim * (w_latent_dim + 1)
        tokens = torch.tensor(tokens, dtype=torch.int64).cuda()

        tokens = tokens.view(h_latent_dim, w_latent_dim + 1)[:, :-1].flatten()

        return self.chameleon_ori_image_tokenizer.pil_from_img_toks(tokens, h_latent_dim, w_latent_dim)