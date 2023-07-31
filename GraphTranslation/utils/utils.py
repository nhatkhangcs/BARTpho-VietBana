import string
import json
from hashlib import md5
from nltk import ngrams

punctuation = string.punctuation
punctuation = punctuation.replace(" ", "").replace("_", "").replace("'", "")

s1 = list(
    u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ') + [
         "ĭ", "ŏ", "ŭ", "ĭ", "ŏ", "ê̆", "ơ̆", "ĕ"]
s0 = list(
    u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy') + [
         "i", "o", "u", "i", "o", "ê", "o", "e"]
s1_s0_mapping = {c1: c0 for c1, c0 in zip(s1, s0)}
s1_sorted = [s for s in s1]
s1_sorted.sort(key=lambda s: len(s), reverse=True)


def remove_accents(input_str):
    s = input_str
    for c in s1_sorted:
        s = s.replace(c, s1_s0_mapping[c])
    return s


def norm_word(word):
    word = word.strip()
    if len(word) == 1:
        return word
    for c in punctuation:
        if c in word:
            # word = word.split(c)[0]
            word = word.replace(c, " ")
    word = " ".join(word.split()).replace("_", " ").strip()
    return word


def generate_id(data):
    json_string = json.dumps(data, sort_keys=True, indent=4)
    return md5(json_string.encode("utf-8")).hexdigest()


def check_number(data):
    data = data.replace(".", "").replace(",", "").replace(" ", "")
    for c in string.punctuation:
        data = data.replace(c, "")
    return data.isnumeric()


def jaccard_distance(a: set, b: set):
    return len(a.intersection(b)) / len(a.union(b))


def word_distance(a: str, b: str, n_gram=3, mode="jaccard"):
    if mode == "jaccard":
        a_set = set(gram_set for k in range(1, n_gram + 1) for gram_set in ngrams(a, k))
        b_set = set(gram_set for k in range(1, n_gram + 1) for gram_set in ngrams(b, k))
        return jaccard_distance(a_set, b_set)
    elif mode == "hamming":
        _a_r = remove_accents(a)
        _b_r = remove_accents(b)
        _a = norm_word(_a_r).lower()
        _b = norm_word(_b_r).lower()
        min_length = min(len(_a), len(_b))
        max_length = max(len(_a), len(_b))
        if min_length <= 3:
            if _a == _b:
                return 0
            return 1e9
        output = sum(_a[i] != _b[i] for i in range(min_length)) + max_length - min_length
        return output
    return 0


def norm_space_punctuation(text):
    for c in string.punctuation:
        if c in "'" + '"':
            continue
        if f" {c}" in text:
            text = text.replace(f" {c}", c)
        if f"{c}{c}" in text:
            text = text.replace(f"{c}{c}", c)
    return text
