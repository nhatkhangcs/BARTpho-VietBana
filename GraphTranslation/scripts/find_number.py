import string
vi_dict_path = "data/dictionary/vi_0504_w.txt"
ba_dict_path = "data/dictionary/bana_0504_w.txt"


vi_number = {"một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "trăm", "ngàn", "nghìn",
             "chục", "triệu", "tỷ"}

for vi_word, ba_word in zip(open(vi_dict_path, "r", encoding="utf8"), open(ba_dict_path, "r", encoding="utf8")):
    vi_word = vi_word.replace("\n", " ").strip()
    ba_word = ba_word.replace("\n", " ").strip()
    if len(vi_word) == 0 or len(ba_word) == 0:
        continue

    if len(set(vi_word.split()).intersection(vi_number)) or vi_word.isnumeric():
        print(f"{vi_word} || {ba_word}")

