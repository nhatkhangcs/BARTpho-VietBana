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

    @staticmethod
    def load_data(data_folder, mode):
        vi_data_file = [item for item in os.listdir(data_folder) if mode in item and ".vi" in item][0]
        ba_data_file = [item for item in os.listdir(data_folder) if mode in item and ".ba" in item][0]
        vi_data_path = os.path.join(data_folder, vi_data_file)
        ba_data_path = os.path.join(data_folder, ba_data_file)
        vi_data = [item.replace("\n", "").strip() for item in open(vi_data_path, "r", encoding="utf8").readlines()]
        ba_data = [item.replace("\n", "").strip() for item in open(ba_data_path, "r", encoding="utf8").readlines()]
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
            line = line.replace(p, f" {p} ")
        while "  " in line:
            line = line.replace("  ", " ")
        line = line.strip()
        return line

    def __getitem__(self, idx):
        vi, ba = self.data[idx]
        mapped_chunks, not_mapped_chunks = self.phrase_chunking.extract_chunks(vi, ba)
        mapped_src_chunks, mapped_dst_chunks = list(map(list, zip(*mapped_chunks)))
        not_mapped_src_chunks, not_mapped_dst_chunks = not_mapped_chunks
        src_chunks = mapped_src_chunks + not_mapped_src_chunks
        dst_chunks = mapped_dst_chunks + not_mapped_dst_chunks
        chosen_item = random.randint(0, len(src_chunks) - 1)

        vi = src_chunks[chosen_item].text
        ba = dst_chunks[chosen_item].text

        vi = self.norm_text(vi)
        ba = self.norm_text(ba)
        with self.tokenizer.as_target_tokenizer():
            vi_tokenize = self.tokenizer(vi, truncation=True)
            ba_tokenize = self.tokenizer(ba, truncation=True)
            print(self.tokenizer.tokenize(vi_tokenize))
            print(self.tokenizer.tokenize(ba_tokenize))
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

