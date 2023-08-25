def add_word_to_dict(vie, bah):
    # append new word to dict.ba and dict.vi
    for bah in bah:
        with open("data/dictionary/dict.ba", "a", encoding="utf-8") as f:
            f.write(bah + "\n")
    for vie in vie:
        with open("data/dictionary/dict.vi", "a", encoding="utf-8") as f:
            f.write(vie + "\n")