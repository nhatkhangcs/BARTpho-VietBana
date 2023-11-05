import os
import json

from GraphTranslation.common.languages import Languages
from GraphTranslation.utils.utils import norm_word
from objects.singleton import Singleton


class Config(metaclass=Singleton):
    # Binh Dinh
    BinhDinh = "BinhDinh"

    # Gia Lai
    GiaLai = "GiaLai"

    # KonTum
    KonTum = "KonTum"

    dst_words_paths = "dictionary/dict.ba"
    src_words_paths = "dictionary/dict.vi"
    
    src_monolingual_paths = [
        "parallel_corpus/train.vi", "parallel_corpus/valid.vi"]
    dst_monolingual_paths = [
        "parallel_corpus/train.ba", "parallel_corpus/valid.ba"]
    
    src_mono_test_paths = ["parallel_corpus/test.vi"]
    dst_mono_test_paths = ["parallel_corpus/test.ba"]

    parallel_paths = [("parallel_corpus/train.vi", "parallel_corpus/train.ba"),
                      ("parallel_corpus/valid.vi", "parallel_corpus/valid.ba"),
                      ("dictionary/dict.vi", "dictionary/dict.ba")]

    src_custom_ner_path = "GraphTranslation/data/custom_ner/vi_ner.json"
    dst_custom_ner_path = "GraphTranslation/data/custom_ner/ba_ner.json"

    # ----------------------------------------- #

    src_syn_path = "data/synonyms/vi_syn_data_1.json"
    dst_syn_path = None

    graph_cache_path = "data/cache/graph.json"
    graph_cache_path_1 = "data/cache/graph_1.json"
    activate_path = "data/cache/activation.txt"
    logging_folder = "logs"
    # vncorenlp_host = "http://172.28.0.23"

    # ----------------------------------------- #


    vncorenlp_host = "http://localhost"
    vncorenlp_port = 9000
    cache_size = 10000
    _dst_words = None
    _src_words = None
    _src_dst_mapping = None
    _src_custom_ner = None
    _dst_custom_ner = None
    src_syn_words = {}
    dst_syn_words = {}
    max_gram = 4
    bow_window_size = 2

    #@property
    def src_custom_ner(self):
        if self._src_custom_ner is None and os.path.exists(self.src_custom_ner_path):
            custom_ner = json.load(
                open(self.src_custom_ner_path, "r", encoding="utf8"))
            self._src_custom_ner = custom_ner
        return self._src_custom_ner

    #@property
    def dst_custom_ner(self):
        if self._dst_custom_ner is None and os.path.exists(self.dst_custom_ner_path):
            custom_ner = json.load(
                open(self.dst_custom_ner_path, "r", encoding="utf8"))
            self._dst_custom_ner = custom_ner
        return self._dst_custom_ner

    def load_syn_word_set(self):
        def load_syn_word_set(language: Languages):
            if language == Languages.SRC:
                path = self.src_syn_path
            else:
                path = self.dst_syn_path
            if path is not None and os.path.exists(path):
                syn_data = json.load(open(path, "r", encoding="utf8"))
                syn_data = {norm_word(word): [norm_word(w) for w_set in syn_data[word]["syn"].values() for w in w_set]
                            for word in syn_data}

                if language == Languages.SRC:
                    self.src_syn_words = syn_data
                else:
                    self.dst_syn_words = syn_data
        load_syn_word_set(Languages.SRC)
        load_syn_word_set(Languages.DST)

    @staticmethod
    def upper_start_chars(text):
        return " ".join([item.capitalize() for item in text.split()])

    def load_src_dst_dict(self, area):
        full_path_dst = self.dst_words_paths
        full_path_src = self.src_words_paths
        if area == self.BinhDinh:
            full_path_dst = "data/" + self.BinhDinh + "/" + full_path_dst
            full_path_src = "data/" + self.BinhDinh + "/" + full_path_src
        elif area == self.GiaLai:
            full_path_dst = "data/" + self.GiaLai + "/" + full_path_dst
            full_path_src = "data/" + self.GiaLai + "/" + full_path_src
        else:
            full_path_dst = "data/" + self.KonTum + "/" + full_path_dst
            full_path_src = "data/" + self.KonTum + "/" + full_path_src
        #print(full_path_dst, full_path_src)
        if self._dst_words is None or self._src_words is None or self._src_dst_mapping is None:
            all_dst_words = []
            all_src_words = []
            dst_words_path = full_path_dst
            src_words_path = full_path_src
            # for dst_words_path, src_words_path in zip(full_path_dst, full_path_src):
            dst_words = [item.replace("\n", "").strip()
                         for item in open(dst_words_path, "r", encoding="utf8").readlines()]
            #print(len(dst_words))
            dst_words = [item for item in dst_words if len(item) > 0]
            #print(len(dst_words))
            dst_words += [self.upper_start_chars(w) for w in dst_words]
            #print(len(dst_words))
            dst_words += [w.lower() for w in dst_words]
            #print(len(dst_words))
            src_words = [item.replace("\n", "").strip()
                         for item in open(src_words_path, "r", encoding="utf8").readlines()]
            #print(len(src_words))
            src_words = [item for item in src_words if len(item) > 0]
            #print(len(src_words))
            src_words += [self.upper_start_chars(w) for w in src_words]
            #print(len(src_words))
            src_words += [w.lower() for w in src_words]
            #print(len(src_words))
            if len(dst_words) != len(src_words):
                raise ValueError("Ba dict must be equal size to Vi dict")
            all_dst_words += dst_words
            all_src_words += src_words
            self._dst_words = all_dst_words
            self._src_words = all_src_words
            dictionary = set()
            for src, dst in zip(all_src_words, all_dst_words):
                dictionary.add((src, dst))
            self._src_dst_mapping = dictionary
            self.load_syn_word_set()

    #@property
    def dst_words(self, area):
        self.load_src_dst_dict(area)
        dst_dict = self.dst_syn_words
        dst_words = list(dst_dict.keys()) + \
            [w for w_list in dst_dict.values() for w in w_list]
        return self._dst_words + dst_words

    #@property
    def src_words(self, area):
        self.load_src_dst_dict(area)
        # syn_dict = self.src_syn_words
        # syn_words = list(syn_dict.keys()) + [w for w_list in syn_dict.values() for w in w_list]
        # return self._src_words + syn_words
        return self._src_words

    #@property
    def dst_word_set(self, area):
        return set(self.dst_words(area))

    #@property
    def src_word_set(self, area):
        return set(self.src_words(area))

    #@property
    def src_dst_mapping(self, area):
        self.load_src_dst_dict(area)
        #print(dict(self._src_dst_mapping))
        return self._src_dst_mapping
