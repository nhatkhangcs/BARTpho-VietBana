from abc import ABC
import os
import string
import random

from torch.utils.data import Dataset
from transformers import AutoTokenizer
from GraphTranslation.pipeline.translation import TranslationPipeline


class ViBaDataset(Dataset, ABC):
    def __init__(self, data_folder, mode, tokenizer_path: str):
        super(ViBaDataset, self).__init__()
        self.data = self.load_data(data_folder, mode)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.phrase_chunking = TranslationPipeline()
        self.phrase_chunking.add_check_valid_anchor_func(self.check_valid_anchor)

    @staticmethod
    def check_valid_anchor(word):
        return word.is_punctuation or word.is_ner or word.is_end_sent or word.is_end_paragraph or word.is_conjunction

    @staticmethod
    def load_data(data_folder, mode, return_index=False):
        if mode == "train":
            vi_data_files = [item for item in os.listdir(data_folder) if
                             ".vi" in item and "test" not in item and "valid" not in item]
            ba_data_files = [item for item in os.listdir(data_folder) if
                             ".ba" in item and "test" not in item and "valid" not in item]
            vi_data_files.sort()
            ba_data_files.sort()
            data_name = []
            vi_data = []
            ba_data = []
            for vi_data_file in vi_data_files:
                ba_data_file = vi_data_file.replace(".vi", ".ba")
                if ba_data_file not in ba_data_files:
                    continue
                vi_data_path = os.path.join(data_folder, vi_data_file)
                ba_data_path = os.path.join(data_folder, ba_data_file)
                _vi_data = [item.replace("\n", "").strip() for item in
                            open(vi_data_path, "r", encoding="utf8").readlines()]
                _ba_data = [item.replace("\n", "").strip() for item in
                            open(ba_data_path, "r", encoding="utf8").readlines()]
                data_name += [f"{vi_data_file}___{i}" for i in range(len(_vi_data))]
                vi_data += _vi_data
                ba_data += _ba_data

        else:
            vi_data_file = [item for item in os.listdir(data_folder) if mode in item and ".vi" in item][0]
            ba_data_file = [item for item in os.listdir(data_folder) if mode in item and ".ba" in item][0]
            vi_data_path = os.path.join(data_folder, vi_data_file)
            ba_data_path = os.path.join(data_folder, ba_data_file)
            vi_data = [item.replace("\n", "").strip() for item in open(vi_data_path, "r", encoding="utf8").readlines()]
            ba_data = [item.replace("\n", "").strip() for item in open(ba_data_path, "r", encoding="utf8").readlines()]
            data_name = [f"{vi_data_file}___{i}" for i in range(len(vi_data))]

        if return_index:
            output = list(zip(data_name, vi_data, ba_data))
        else:
            output = list(zip(vi_data, ba_data))

        if mode == "valid":
            return output[:100]
        elif mode == "test":
            return output[:-10]
        return output

    def __len__(self):
        return len(self.data)

    def norm_text(self, line):
        for n in string.digits:
            line = line.replace(n, f" {n} ")
        for p in string.punctuation + ",.\\/:;–…":
            line = line.replace(p, f" ")
        while "  " in line:
            line = line.replace("  ", " ")
        line = line.strip()
        return line

    def __getitem__(self, idx):
        vi, ba = self.data[idx]
        new_vi = None
        new_ba = None
        if random.random() > 0.8:
            new_vi = vi
            new_ba = ba
        else:
            mapped_chunks, not_mapped_chunks = self.phrase_chunking.extract_chunks(vi, ba)
            if len(mapped_chunks) > 0:
                mapped_src_chunks, mapped_dst_chunks = list(map(list, zip(*mapped_chunks)))
            else:
                mapped_src_chunks, mapped_dst_chunks = [], []
            not_mapped_src_chunks, not_mapped_dst_chunks = not_mapped_chunks
            src_chunks = mapped_src_chunks + not_mapped_src_chunks
            dst_chunks = mapped_dst_chunks + not_mapped_dst_chunks
            if len(src_chunks) != len(dst_chunks) or len(src_chunks) == 0 or len(dst_chunks) == 0:
                new_vi = vi
                new_ba = ba
            count = 0
            while (new_vi is None or new_ba is None) and count < 5:
                count += 1
                chosen_item = random.randint(0, len(src_chunks) - 1)
                vi_chunk = src_chunks[chosen_item]
                ba_chunk = dst_chunks[chosen_item]
                if vi_chunk is not None and ba_chunk is not None:
                    new_vi = vi_chunk.text
                    new_ba = ba_chunk.text
            if new_vi is None or new_ba is None:
                new_vi = vi
                new_ba = ba
        vi = new_vi
        ba = new_ba

        vi = self.norm_text(vi)
        ba = self.norm_text(ba)
        vi_tokenize = self.tokenizer(vi, truncation=True)
        ba_tokenize = self.tokenizer(ba, truncation=True)
        return {
            "input_ids": vi_tokenize.input_ids,
            "attention_mask": vi_tokenize.attention_mask,
            "labels": ba_tokenize.input_ids
        }

    @staticmethod
    def get_datasets(data_folder, tokenizer_path):
        train_dataset = ViBaDataset(data_folder, "train", tokenizer_path)
        valid_dataset = ViBaDataset(data_folder, "valid", tokenizer_path)
        test_dataset = ViBaDataset(data_folder, "test", tokenizer_path)
        return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    dataset = ViBaDataset(data_folder="../data", mode="train", tokenizer_path="../pretrained/bartpho_syllable")
    print(dataset[0])
