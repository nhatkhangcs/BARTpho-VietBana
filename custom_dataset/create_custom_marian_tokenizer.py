import os
from tqdm import tqdm
import string

from transformers import AutoTokenizer


DATA_FOLDER = "data/new_all"
ba_files = [item for item in os.listdir(DATA_FOLDER) if ".ba" in item]
ba_files = [os.path.join(DATA_FOLDER, item) for item in ba_files]

ba_tokens = set()
l = None
for ba_file in ba_files:
    data = open(ba_file, "r", encoding="utf8").readlines()
    data = [item.replace("\n", "").strip() for item in data]
    for line in tqdm(data):
        if l is None:
            l = line
        for n in string.digits:
            line = line.replace(n, f" {n} ")
        for p in string.punctuation + ",.\\/:;–":
            line = line.replace(p, f" {p} ")
        while "  " in line:
            line = line.replace("  ", " ")
        line = line.strip()
        words = line.split(" ")
        for word in words:
            ba_tokens.add(word)

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-vi-en")
print(tokenizer.get_vocab()["<unk>"])
ba_token_list = list(ba_tokens)
ba_token_list.sort()
tokenizer.add_tokens(ba_token_list, special_tokens=True)
tokenizer.save_pretrained("pretrained/marian")

print(tokenizer.tokenize(l))
a = tokenizer.encode(l)
b = tokenizer.decode(a)
print(b)
