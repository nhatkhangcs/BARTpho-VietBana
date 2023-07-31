vi_path = 'data/all/kriem.vi'
ba_path = 'data/all/kriem.ba'

vi_data = open(vi_path, "r", encoding="utf8").readlines()
ba_data = open(ba_path, "r", encoding="utf8").readlines()
vi_data = [item.strip() for item in vi_data]
ba_data = [item.strip() for item in ba_data]

new_vi_words = []
new_ba_words = []
for vi_word, ba_word in zip(vi_data, ba_data):
    vi_words = [item.strip() for item in vi_word.split(",")]
    ba_words = [item.strip() for item in ba_word.split(",")]
    for vi in vi_words:
        for ba in ba_words:
            new_vi_words.append(vi)
            new_ba_words.append(ba)

open("data/all/norm_kriem.vi", "w", encoding="utf8").write("\n".join(new_vi_words))
open("data/all/norm_kriem.ba", "w", encoding="utf8").write("\n".join(new_ba_words))