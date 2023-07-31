import os
from typing import Any, Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizer, AddedToken
from transformers import logging

logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"monolingual_vocab_file": "dict.txt"}

# PRETRAINED_VOCAB_FILES_MAP = {
#     "vocab_file": {
#         "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/sentencepiece.bpe.model",
#     },
#     "monolingual_vocab_file": {
#         "vinai/bartpho-syllable": "https://huggingface.co/vinai/bartpho-syllable/resolve/main/dict.txt",
#     },
# }

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {"vinai/bartpho-syllable": 1024}


class CustomTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    # pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        monolingual_vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )

        # self.vocab_file = vocab_file
        self.monolingual_vocab_file = monolingual_vocab_file

        # Load the reduced vocab
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        with open(monolingual_vocab_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                token = line.strip().split()[0]
                if token not in self.fairseq_tokens_to_ids:
                    self.fairseq_tokens_to_ids[token] = len(self.fairseq_tokens_to_ids)
        self.fairseq_tokens_to_ids["<mask>"] = len(self.fairseq_tokens_to_ids)

        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, d):
        self.__dict__ = d

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An BARTPho sequence has the following format:

        - single sequence: ``<s> X </s>``
        - pair of sequences: ``<s> A </s></s> B </s>``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BARTPho does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        return len(self.fairseq_ids_to_tokens)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def split_num(self, sent):
        out = ""
        for i in range(len(sent)):
            char = sent[i]
            pre_char = "o" if i == 0 else sent[i - 1]
            if char in "0123456789" and pre_char in "0123456789,.":
                out += " " + char + " "
            else:
                out += char
        return out

    def split_word(self, sent, lang="vi"):
        out = ""
        if lang == "vi":
            split_char = "`'~!@#$%^&*()_-+={[}]:;<,>.?/" + '"' + "|\\"
        else:
            split_char = "~!@#$%^&*()_-+={[}]:;<,>.?/" + '"' + "|\\"
        for char in sent:
            if char in split_char:
                out += " " + char + " "
            else:
                out += char
        return out

    def drop_double_space(self, sent):
        while "  " in sent:
            sent = sent.replace("  ", " ")
        return sent.strip()

    def _tokenize(self, text: str) -> List[str]:
        return self.drop_double_space(self.split_num(self.split_word(text, "ba"))).split(" ")

    def _decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        spaces_between_special_tokens: bool = True,
        **kwargs
    ) -> str:
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", False)

        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        # To avoid mixing byte-level and unicode for byte-level BPT
        # we need to build string separately for added tokens and byte-level tokens
        # cf. https://github.com/huggingface/transformers/issues/1133
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            if token in self.added_tokens_encoder:
                if current_sub_text:
                    sub_texts.append(self.convert_tokens_to_string(current_sub_text))
                    current_sub_text = []
                sub_texts.append(token)
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        if spaces_between_special_tokens:
            text = " ".join(sub_texts)
        else:
            text = "".join(sub_texts)
        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        else:
            return self.fairseq_tokens_to_ids["<unk>"]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.fairseq_ids_to_tokens[index]

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = " ".join(tokens).strip()
        return out_string

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_monolingual_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_dict"],
        )
        if self.monolingual_vocab_file is None or os.path.abspath(self.monolingual_vocab_file) != os.path.abspath(out_monolingual_vocab_file):
            # copyfile(self.monolingual_vocab_file, out_monolingual_vocab_file)
            with open(out_monolingual_vocab_file, "w", encoding="utf8") as f:
                token_id = list(self.fairseq_tokens_to_ids.items())
                token_id.sort(key=lambda item: item[1])
                for token, value in token_id:
                    f.write(token + " " + str(value) + "\n")

        return None, out_monolingual_vocab_file

    def train(self, data_files):
        bana_vocab = dict()
        for file_path in data_files:
            with open(file_path, "r", encoding="utf8") as f:
                for line_ in f:
                    line_ = line_.replace("\n", "")
                    tokens = self.tokenize(line_)
                    for token in tokens:
                        if token not in bana_vocab:
                            bana_vocab[token] = 0
                        bana_vocab[token] += 1

        for token in bana_vocab.keys():
            self.fairseq_tokens_to_ids[token] = len(self.fairseq_tokens_to_ids)


def main():
    tokenizer = CustomTokenizer.from_pretrained("ba_tokenizer")
    ids = tokenizer.encode("Trö jeân pôm,minh sônaêm kung thu yoêk ñei khoang 60  trieâu ñoâng")
    print(tokenizer.tokenize("Trö jeân pôm,minh sônaêm kung thu yoêk ñei khoang 60  trieâu ñoâng"))
    print(tokenizer.convert_ids_to_tokens(ids))
    print(tokenizer.decode(ids, skip_special_tokens=True))
    print(tokenizer(["Trö jeân pôm ,  minh sônaêm kung thu yoêk ñei khoang 6 0  trieâu ñoâng"]))


if __name__ == "__main__":
    main()
