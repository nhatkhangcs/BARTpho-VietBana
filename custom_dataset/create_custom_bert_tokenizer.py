import os
from tqdm import tqdm
import string

from transformers import AutoTokenizer, MBartModel, BartTokenizer, BartphoTokenizer, PhobertTokenizer


DATA_FOLDER = "data/all"
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
        for p in string.punctuation:
            line = line.replace(p, f" {p} ")
        while "  " in line:
            line = line.replace("  ", " ")
        line = line.strip()
        words = line.split(" ")
        for word in words:
            ba_tokens.add(word)

syllable_tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base")
print(syllable_tokenizer.get_vocab()["<unk>"])
syllable_tokenizer.add_tokens(list(ba_tokens), special_tokens=True)
syllable_tokenizer.save_pretrained("pretrained/phobert-base")

print(syllable_tokenizer.tokenize(l))
a = syllable_tokenizer.encode(l)
b = syllable_tokenizer.batch_decode([a], skip_special_tokens=True)
print(b)
